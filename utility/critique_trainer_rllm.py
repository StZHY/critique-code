"""Critique training loop (residual-LLM): per-round data load, sampling, multi-task training, eval, save."""
import torch
from time import time
import utility.tools as tools
import utility.tester as tester
import utility.critique_sample_rllm as cri_sample
import os
import json
import datetime


def critique_training(critique_model, args, dataset, device, logger):
    """Critique training (residual-LLM): critique_round_num rounds x critique_epoch epochs per round."""
    critique_model.to(device)

    incr_gate = getattr(args, 'integ_incremental_gate', False)
    if incr_gate and hasattr(critique_model, 'set_active_round'):
        critique_model.set_active_round(None)

    model_results = tester.testing(critique_model, args, dataset, device)
    print("\t Recall:" + str(model_results['recall']) + "\n\t NDCG:  " + str(model_results['ndcg']) + "\n\t HitRate:  " + str(model_results['hitrate']))
    logger.info("\t Recall:" + str(model_results['recall']) + "\n" + "\t" * 7 + " NDCG:  " + str(model_results['ndcg']) + "\n" + "\t" * 7 + " HitRate:  " + str(model_results['hitrate']))

    def _metric_snapshot(tag, res):
        return {
            "round": tag,
            "recall": [float(x) for x in res['recall']],
            "ndcg": [float(x) for x in res['ndcg']],
            "hitrate": [float(x) for x in res['hitrate']],
        }
    round_metrics = [_metric_snapshot("baseline", model_results)]

    critique_model.supplement_average_items(dataset.train_dict)
    critique_optim = torch.optim.Adam(critique_model.parameters(), lr=float(args.critique_rate))
    best_recall_epoch, best_recall_0, best_ndcg_0 = 0, 0, 0
    cnt = 0

    logger.info(critique_model)
    logger.info(critique_optim)

    use_rllm = getattr(args, 'rllm_mode', 'none') != 'none'

    for round_num in range(int(args.critique_round_num)):

        dataset.load_critique_data(str(round_num))

        if incr_gate and hasattr(critique_model, 'set_active_round'):
            critique_model.set_active_round(round_num)

        start_time = time()

        user_pos_pairs = dataset.create_u_pos_pairs()
        users = torch.Tensor(user_pos_pairs[:, 0]).long()
        pos_items = torch.Tensor(user_pos_pairs[:, 1]).long()

        bpr_neg_items, sampled_neg_items = cri_sample.weight_sample_bpr_neg_items(critique_model, args, dataset, device)
        pos_cand_items = cri_sample.sample_pos_items(critique_model, args, dataset, device)
        pos_pool_ids, pos_pool_scores = cri_sample.sample_pos_pool(args, dataset, device)
        cri_pool_ids, cri_pool_scores = cri_sample.sample_critique_pool(args, dataset, device)
        if getattr(args, 'use_cer_margin', False):
            cer_ids, cer_g = cri_sample.sample_cri_neg_cl(args, dataset, device)
            critique_model.set_cri_neg_cl(cer_ids, cer_g)
        neg_gate = cri_sample.compute_neg_gate(args, bpr_neg_items, device)

        if round_num == 0:
            history_sampled_neg = sampled_neg_items
        else:
            for user_id, neg_items in sampled_neg_items.items():
                if user_id in history_sampled_neg:
                    history_sampled_neg[user_id].extend(neg_items)
                else:
                    history_sampled_neg[user_id] = neg_items

        cl_neg_items = cri_sample.sample_cl_neg_items(args, dataset, device, history_sampled_neg)

        users = users.to(device)
        bpr_neg_items = bpr_neg_items[users]
        cl_neg_items = cl_neg_items[users]
        pos_cand_items = pos_cand_items[users]
        pos_pool_ids = pos_pool_ids[users]
        pos_pool_scores = pos_pool_scores[users]
        cri_pool_ids = cri_pool_ids[users]
        cri_pool_scores = cri_pool_scores[users]
        if neg_gate is not None:
            neg_gate = neg_gate[users]

        neg_users_set = set(dataset.neg_train_dict.keys())
        pos_users_set = set(getattr(dataset, 'pos_train_dict', {}).keys())
        neg_mask = torch.tensor([1.0 if int(u) in neg_users_set else 0.0 for u in users.tolist()], device=device)
        pos_mask = torch.tensor([1.0 if int(u) in pos_users_set else 0.0 for u in users.tolist()], device=device)
        logger.info(f"\t Round {round_num}: participating users {len(users)} (neg={int(neg_mask.sum().item())} / pos={int(pos_mask.sum().item())}) | rllm={args.rllm_mode} λ={args.rllm_lambda}")

        cnt = 0
        use_gate = neg_gate is not None
        for epoch in range(int(args.critique_epoch)):

            users = users.to(device)
            pos_items = pos_items.to(device)

            if use_gate:
                users, pos_items, bpr_neg_items, cl_neg_items, pos_cand_items, pos_pool_ids, pos_pool_scores, \
                    cri_pool_ids, cri_pool_scores, neg_mask, pos_mask, neg_gate = \
                    tools.shuffle(users, pos_items, bpr_neg_items, cl_neg_items, pos_cand_items,
                                  pos_pool_ids, pos_pool_scores, cri_pool_ids, cri_pool_scores,
                                  neg_mask, pos_mask, neg_gate)
            else:
                users, pos_items, bpr_neg_items, cl_neg_items, pos_cand_items, pos_pool_ids, pos_pool_scores, \
                    cri_pool_ids, cri_pool_scores, neg_mask, pos_mask = \
                    tools.shuffle(users, pos_items, bpr_neg_items, cl_neg_items, pos_cand_items,
                                  pos_pool_ids, pos_pool_scores, cri_pool_ids, cri_pool_scores,
                                  neg_mask, pos_mask)
            num_batch = len(users) // int(args.train_batch_size) + 1
            total_loss = 0.

            critique_model.train()

            if use_gate:
                batches = tools.mini_batch(users, pos_items, bpr_neg_items, cl_neg_items, pos_cand_items,
                                           pos_pool_ids, pos_pool_scores, cri_pool_ids, cri_pool_scores,
                                           neg_mask, pos_mask, neg_gate, batch_size=int(args.train_batch_size))
            else:
                batches = tools.mini_batch(users, pos_items, bpr_neg_items, cl_neg_items, pos_cand_items,
                                           pos_pool_ids, pos_pool_scores, cri_pool_ids, cri_pool_scores,
                                           neg_mask, pos_mask, batch_size=int(args.train_batch_size))
            for batch in batches:
                batch_user, batch_pos, batch_bpr_neg, batch_cl_neg, batch_pos_cand, \
                    batch_pos_pool_ids, batch_pos_pool_scores, batch_cri_pool_ids, batch_cri_pool_scores, \
                    batch_neg_mask, batch_pos_mask = batch[:11]
                batch_neg_gate = batch[11] if use_gate else None

                loss = critique_model(batch_user, batch_pos, batch_bpr_neg, batch_cl_neg,
                                      batch_pos_cand, batch_neg_mask, batch_pos_mask, batch_neg_gate,
                                      pos_pool=batch_pos_pool_ids, pos_pool_scores=batch_pos_pool_scores,
                                      cri_pool=batch_cri_pool_ids, cri_pool_scores=batch_cri_pool_scores)
                total_loss += loss.item()

                critique_optim.zero_grad()
                loss.backward()
                critique_optim.step()

            end_time = time()
            avg_loss = round(total_loss / num_batch, 6)
            print("\t Round: %4d | Epoch: %4d | train time %.3f | train loss: %.6f" % (round_num + 1, epoch + 1, end_time - start_time, avg_loss))
            logger.info("\t Round: %4d | Epoch: %4d | train time %.3f | train loss: %s" % (round_num + 1, epoch + 1, end_time - start_time, avg_loss))

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
                logger.info("\t Recall:" + str(model_results['recall']) + "\n" + "\t" * 7 + " NDCG:  " + str(model_results['ndcg']) + "\n" + "\t" * 7 + " HitRate:  " + str(model_results['hitrate']))
                if cnt > int(args.early_stop):
                    break
            else:
                result = tester.sparsity_test(dataset, args, critique_model, device)
                if result[0]['ndcg'][1] > best_ndcg_0:
                    best_report_epoch = epoch + 1
                    best_report_ndcg = result[0]['ndcg'][1]
                logger.info("\t level_1: recall:" + str(result[0]['recall']) + ',ndcg:' + str(result[0]['ndcg']))

        round_res = tester.testing(critique_model, args, dataset, device)
        round_metrics.append(_metric_snapshot(round_num, round_res))
        logger.info("\t Round %d DONE: R@5/10/20 = %s | ndcg@5/10/20 = %s" %
                    (round_num, str(round_res['recall']), str(round_res['ndcg'])))

    print("\t Model training process completed.")
    logger.info("\t Model training process completed.")
    logger.info("\t best recall epoch:" + str(best_recall_epoch))
    logger.info("\t best recall:" + str(best_recall_0) + "\t best ndcg:" + str(best_ndcg_0))

    os.makedirs("model_save", exist_ok=True)
    log_tag = args.log if args.log and args.log != "None" else "default"
    critique_save_path = f"model_save/critique_{args.dataset}_{log_tag}.pth"
    torch.save(critique_model.state_dict(), critique_save_path)
    print(f"\t Critique model saved to {critique_save_path}")
    logger.info(f"\t Critique model saved to {critique_save_path}")

    round_metrics_path = f"results/round_metrics_{log_tag}.json"
    json.dump(round_metrics, open(round_metrics_path, "w", encoding="utf-8"), indent=2)
    print(f"\t Round metrics saved to {round_metrics_path} ({len(round_metrics)} rounds including baseline)")
    logger.info(f"\t Round metrics saved to {round_metrics_path}")
    for rm in round_metrics:
        print("\t round=%s | R@5/10/20=%.4f/%.4f/%.4f | ndcg@5/10/20=%.4f/%.4f/%.4f" %
              (rm['round'], rm['recall'][0], rm['recall'][1], rm['recall'][2],
               rm['ndcg'][0], rm['ndcg'][1], rm['ndcg'][2]))

    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    result_folder = "results"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    file_path = os.path.join(result_folder, f"{critique_model.model_name}_results_{current_time}.txt")
    with open(file_path, 'w') as file:
        file.write("Model Training Results for {}:\n".format(critique_model.model_name))
        file.write("rllm_mode: {} | rllm_lambda: {} | npc_alpha: {}\n".format(args.rllm_mode, args.rllm_lambda, args.npc_alpha))
        file.write("Best Recall Epoch: " + str(best_recall_epoch) + "\n")
        file.write("Best Recall: " + str(best_recall_0) + "\n")
        file.write("Best NDCG: " + str(best_ndcg_0) + "\n")

    handlers = logger.handlers
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()
