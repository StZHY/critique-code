import torch
from time import time, strftime, localtime
import utility.tools as tools
import utility.tester as tester
import utility.critique_sample as cri_sample 
import os
import datetime


def critique_training(critique_model, args, dataset, device, logger):

    critique_model.to(device)
    history_sampled_neg = {}
    
    model_results = tester.testing(critique_model, args, dataset, device)
    print("\t Recall:" + str(model_results['recall']) + "\n\t NDCG:  " + str(model_results['ndcg']) + "\n\t HitRate:  " + str(model_results['hitrate']))
    logger.info("\t Recall:" + str(model_results['recall']) + "\n" + "\t" * 7 + " NDCG:  " + str(model_results['ndcg']) + "\n" + "\t" * 7 + " HitRate:  " + str(model_results['hitrate']))

    critique_model.supplement_average_items(dataset.train_dict)
    critique_optim = torch.optim.Adam(critique_model.parameters(), lr=float(args.critique_rate))
    best_recall_epoch, best_recall_0, best_ndcg_0 = 0, 0, 0
    cnt = 0

    logger.info(critique_model)
    logger.info(critique_optim)

    for round_num in range(int(args.critique_round_num)):
        
        dataset.load_critique_data(str(round_num))
        
        start_time = time()

        user_pos_pairs = dataset.create_u_pos_pairs()
        users = torch.Tensor(user_pos_pairs[:, 0]).long()
        pos_items = torch.Tensor(user_pos_pairs[:, 1]).long()

        bpr_neg_items, sampled_neg_items = cri_sample.weight_sample_bpr_neg_items(critique_model, args, dataset, device)

        if round_num == 0:
            history_sampled_neg = sampled_neg_items
        else:
            for user_id, neg_items in sampled_neg_items.items():
                if user_id in history_sampled_neg:
                    history_sampled_neg[user_id].extend(neg_items)
                else:
                    history_sampled_neg[user_id] = neg_items
                        
        cl_neg_items = cri_sample.sample_cl_neg_items(args, dataset, device, history_sampled_neg)
        
        for epoch in range(int(args.critique_epoch)):

            users = users.to(device)
            pos_items = pos_items.to(device)

            users, pos_items, bpr_neg_items, cl_neg_items = tools.shuffle(users, pos_items, bpr_neg_items, cl_neg_items)
            num_batch = len(users) // int(args.train_batch_size) + 1
            total_loss = 0.
            
            critique_model.train()

            for batch_user, batch_pos, batch_bpr_neg, batch_cl_neg in \
                tools.mini_batch(users, pos_items, bpr_neg_items, cl_neg_items, batch_size=int(args.train_batch_size)):

                loss = critique_model(batch_user, batch_pos, batch_bpr_neg, batch_cl_neg)
                total_loss += loss.item()
                
                critique_optim.zero_grad()
                loss.backward()
                critique_optim.step()
                
            end_time = time()
            avg_loss = round(total_loss / num_batch, 6)
            print("\t Round: %4d | Epoch: %4d | train time %.3f | train loss: %.6f" % (round_num+1, epoch + 1, end_time - start_time, avg_loss))
            logger.info("\t Round: %4d | Epoch: %4d | train time %.3f | train loss: %s" % (round_num+1, epoch + 1, end_time - start_time, avg_loss))


            if int(args.sparsity_test) == 0:
                model_results = tester.testing(critique_model, args, dataset, device)
                cnt += 1
                if model_results['ndcg'][0] > best_ndcg_0:
                    cnt = 0
                    best_ndcg_round = round_num + 1
                    best_recall_epoch = epoch + 1
                    best_recall_0 = model_results['recall'][0]
                    best_ndcg_0 = model_results['ndcg'][0]

                print("\t Recall:" + str(model_results['recall']) + "\n\t NDCG:  " + str(model_results['ndcg']) + "\n\t HitRate:  " + str(model_results['hitrate']))
                logger.info(
                    "\t Recall:" + str(model_results['recall']) + "\n" + "\t" * 7 + " NDCG:  " + str(
                        model_results['ndcg']) + "\n" + "\t" * 7 + " HitRate:  " + str(model_results['hitrate']))

                if cnt > int(args.early_stop):
                    break
            else:
                result = tester.sparsity_test(dataset, args, critique_model, device)
                if result[0]['ndcg'][1] > best_ndcg_0:
                    best_report_epoch = epoch + 1
                    best_report_ndcg = result[0]['ndcg'][1]
                print("\t level_1: recall:", result[0]['recall'], ',ndcg:',
                      result[0]['ndcg'])
                print("\t level_2: recall:", result[1]['recall'], ',ndcg:',
                      result[1]['ndcg'])
                print("\t level_3: recall:", result[2]['recall'], ',ndcg:',
                      result[2]['ndcg'])
                #                 print("\t level_4: recall:", result[3]['recall'],  ',ndcg:',
                #                       result[3]['ndcg'])
                logger.info("\t level_1: recall:" + str(result[0]['recall']) + ',ndcg:' + str(result[0]['ndcg']))
                logger.info("\t level_2: recall:" + str(result[1]['recall']) + ',ndcg:' + str(result[1]['ndcg']))
                logger.info("\t level_3: recall:" + str(result[2]['recall']) + ',ndcg:' + str(result[2]['ndcg']))
    #                 logger.info("\t level_4: recall:" + str(result[3]['recall']) + ',ndcg:' + str(result[3]['ndcg']))

    print("\t Model training process completed.")
    print("\t best recall epoch:" + str(best_recall_epoch))
    print("\t best ndcg round:" + str(best_ndcg_round))
    print("\t best recall:" + str(best_recall_0) + "\t best ndcg:" + str(best_ndcg_0))

    logger.info("\t Model training process completed.")
    logger.info("\t best recall epoch:" + str(best_recall_epoch))
    logger.info("\t best recall:" + str(best_recall_0) + "\t best ndcg:" + str(best_ndcg_0))

    """
    save file to result folder
    """
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    result_folder = "results"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    file_path = os.path.join(result_folder, f"{critique_model.model_name}_results_{current_time}.txt")
    with open(file_path, 'w') as file:
        file.write("Model Training Results for {}:\n".format(critique_model.model_name))
        file.write("Best Recall Epoch: " + str(best_recall_epoch) + "\n")
        file.write("Best Recall: " + str(best_recall_0) + "\n")
        file.write("Best NDCG: " + str(best_ndcg_0) + "\n")
        file.write("File created at: " + current_time + "\n")

    handlers = logger.handlers

    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()