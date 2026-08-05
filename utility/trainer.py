"""Backbone training loop with early stopping and checkpointing."""
import torch
from time import time, strftime, localtime
import utility.tools as tools
import utility.tester as tester
import os
import datetime


def training(model, args, dataset, device, logger):
    model.to(device)
    best_recall_epoch, best_recall_1, best_ndcg_1 = 0, 0, 0
    cnt = 0

    save_stem = getattr(args, 'save_name', None) or f"best_model_{model.model_name}"
    model_save_path = f"model_save/{save_stem}.pth"

    optim = torch.optim.Adam(model.parameters(), lr=float(args.learn_rate))
    logger.info(model)
    logger.info(optim)

    for epoch in range(int(args.train_epoch)):
        start_time = time()

        model.train()

        user_pos_neg_pairs = dataset.random_create_user_pos_neg()
        users = torch.Tensor(user_pos_neg_pairs[:, 0]).long()
        pos_items = torch.Tensor(user_pos_neg_pairs[:, 1]).long()
        neg_items = torch.Tensor(user_pos_neg_pairs[:, 2]).long()

        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)

        users, pos_items, neg_items = tools.shuffle(users, pos_items, neg_items)

        num_batch = len(users) // int(args.train_batch_size) + 1

        for batch_i, (batch_user, batch_pos, batch_neg) in \
                enumerate(tools.mini_batch(users, pos_items, neg_items, batch_size=int(args.train_batch_size))):

            loss_list = model(batch_user, batch_pos, batch_neg)

            total_loss = 0.

            if batch_i == 0:
                assert len(loss_list) > 1
                total_loss_list = [0.] * len(loss_list)

            for i in range(len(loss_list)):
                loss = loss_list[i]
                total_loss += loss
                total_loss_list[i] += loss.item()

            print('\t Step %d/%d: loss = %.8f' % (batch_i, num_batch, total_loss), end='\r')
            optim.zero_grad()
            total_loss.backward()
            optim.step()
        end_time = time()

        loss = round(sum(total_loss_list) / num_batch, 6)
        loss_strs = str(loss) + "=" + "+".join([str(round(i / num_batch, 6)) for i in total_loss_list])
        print("\t Epoch: %4d| train time %.3f| train loss: %s" % (epoch + 1, end_time - start_time, loss_strs))
        logger.info("\t Epoch: %4d| train time %.3f| train loss: %s" % (epoch + 1, end_time - start_time, loss_strs))

        if epoch % int(args.test_frequency) == 0:
            if int(args.sparsity_test) == 0:
                metric = getattr(args, 'early_stop_metric', 'recall10')
                if metric == 'train_ndcg5':
                    train_res = tester.testing_on(model, args, dataset, device,
                                                  dataset.train_dict, None, tag="train-ndcg5")
                    test_res = tester.testing_on(model, args, dataset, device,
                                                 dataset.test_dict, dataset.train_dict, tag="test-monitor")
                    cnt += 1
                    tr_ndcg5 = float(train_res['ndcg'][0])
                    if tr_ndcg5 > best_ndcg_1:
                        cnt = 0
                        best_recall_epoch = epoch + 1
                        best_ndcg_1 = tr_ndcg5
                        best_recall_1 = float(test_res['recall'][1])
                        torch.save(model.state_dict(), model_save_path)
                        print(f"\t [saved] epoch{epoch + 1} train_ndcg5={tr_ndcg5:.4f} "
                              f"(test R@10={best_recall_1:.4f})")
                    logger.info("\t [ep%d|train_ndcg5] train_ndcg5=%.4f | test recall=%s ndcg=%s"
                                % (epoch + 1, tr_ndcg5, str(test_res['recall']), str(test_res['ndcg'])))
                    if cnt > int(args.early_stop):
                        print("\t train ndcg@5 plateaued -> early stop")
                        break
                else:
                    if getattr(dataset, 'has_val', False):
                        model_results = tester.testing_on(model, args, dataset, device,
                                                          dataset.val_dict, dataset.train_dict, tag="val")
                    else:
                        model_results = tester.testing(model, args, dataset, device)
                    cnt += 1
                    _k2i = {'5': 0, '10': 1, '20': 2}
                    if metric.startswith('ndcg'):
                        _track_arr = model_results['ndcg']; _ki = metric[4:] or '10'
                    else:
                        _track_arr = model_results['recall']; _ki = metric.replace('recall', '') or '10'
                    _track_idx = _k2i.get(_ki, 1)
                    _track_val = _track_arr[_track_idx]
                    if _track_val > best_recall_1:
                        cnt = 0
                        best_recall_epoch = epoch + 1
                        best_recall_1 = _track_val
                        best_ndcg_1 = model_results['ndcg'][1]

                        torch.save(model.state_dict(), model_save_path)
                        print(f"\t Model parameters saved to {model_save_path} ({metric}={_track_val:.5f})")

                    print("\t Recall:" + str(model_results['recall']) + "\n\t NDCG:  " + str(model_results['ndcg']))
                    logger.info(
                        "\t Recall:" + str(model_results['recall']) + "\n" + "\t" * 7 + " NDCG:  " + str(
                            model_results['ndcg']))

                    if cnt > int(args.early_stop):
                        break
            else:
                result = tester.sparsity_test(dataset, args, model, device)
                if result[0]['ndcg'][1] > best_recall_1:
                    best_report_epoch = epoch + 1
                    best_report_recall = result[0]['ndcg'][1]
                print("\t level_1: recall:", result[0]['recall'], ',ndcg:',
                      result[0]['ndcg'])
                print("\t level_2: recall:", result[1]['recall'], ',ndcg:',
                      result[1]['ndcg'])
                print("\t level_3: recall:", result[2]['recall'], ',ndcg:',
                      result[2]['ndcg'])
                logger.info("\t level_1: recall:" + str(result[0]['recall']) + ',ndcg:' + str(result[0]['ndcg']))
                logger.info("\t level_2: recall:" + str(result[1]['recall']) + ',ndcg:' + str(result[1]['ndcg']))
                logger.info("\t level_3: recall:" + str(result[2]['recall']) + ',ndcg:' + str(result[2]['ndcg']))

    metric = getattr(args, 'early_stop_metric', 'recall10')
    if metric == 'train_ndcg5':
        if os.path.exists(model_save_path):
            model.load_state_dict(torch.load(model_save_path))
            print(f"\t Loaded best model (train ndcg@5 peak @epoch{best_recall_epoch}) for final test eval.")
        test_results = tester.testing_on(model, args, dataset, device,
                                         dataset.test_dict, dataset.train_dict, tag="TEST-FINAL")
        logger.info("\t [TEST-FINAL] Recall:" + str(test_results['recall']) + " NDCG:" + str(test_results['ndcg']))
        best_recall_1 = float(test_results['recall'][1])
        best_ndcg_1 = float(test_results['ndcg'][1])
    elif getattr(dataset, 'has_val', False):
        if os.path.exists(model_save_path):
            model.load_state_dict(torch.load(model_save_path))
            print(f"\t Loaded best model from {model_save_path} for final test eval.")
        excl = {}
        for u, its in dataset.train_dict.items():
            excl[u] = list(its)
        for u, its in dataset.val_dict.items():
            excl[u] = excl.get(u, []) + list(its)
        test_results = tester.testing_on(model, args, dataset, device,
                                         dataset.test_dict, excl, tag="TEST-FINAL")
        logger.info("\t [TEST-FINAL] Recall:" + str(test_results['recall']) + " NDCG:" + str(test_results['ndcg']))

    print("\t Model training process completed.")
    print("\t best recall epoch:" + str(best_recall_epoch))
    print("\t best recall:" + str(best_recall_1) + "\t best ndcg:" + str(best_ndcg_1))

    logger.info("\t Model training process completed.")
    logger.info("\t best recall epoch:" + str(best_recall_epoch))
    logger.info("\t best recall:" + str(best_recall_1) + "\t best ndcg:" + str(best_ndcg_1))

    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    result_folder = "results"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    file_path = os.path.join(result_folder, f"{model.model_name}_results_{current_time}.txt")
    with open(file_path, 'w') as file:
        file.write("Model Training Results for {}:\n".format(model.model_name))
        file.write("Best Recall Epoch: " + str(best_recall_epoch) + "\n")
        file.write("Best Recall: " + str(best_recall_1) + "\n")
        file.write("Best NDCG: " + str(best_ndcg_1) + "\n")
        file.write("File created at: " + current_time + "\n")

    handlers = logger.handlers

    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()
