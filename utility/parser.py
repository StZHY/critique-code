"""Argument parser shared by backbone training and critique scripts."""
import argparse

def parse_args():
    parse = argparse.ArgumentParser(description="Run Reproduct")
    parse.add_argument('--seed', type=int, default=2025, help='random seed')
    parse.add_argument('--gpu', type=int, default=0, help='indicates which gpu to use')
    parse.add_argument('--cuda', type=bool, default=True, help='use gpu or not')
    parse.add_argument('--log', type=str, default='None', help='init log file name')
    parse.add_argument('--dataset_path', type=str, default='./dataset/', help='choice dataset')
    parse.add_argument('--dataset_type', type=str, default='.txt', help='choice dataset')
    parse.add_argument('--dataset', type=str, default='movielens-1m', help='choice dataset')
    parse.add_argument('--top_K', type=str, default='[5, 10, 20]')
    parse.add_argument('--train_epoch', type=int, default=600)
    parse.add_argument('--critique_round_num', type=int, default=5)
    parse.add_argument('--critique_epoch', type=int, default=12)
    parse.add_argument('--early_stop', type=int, default=10)
    parse.add_argument('--early_stop_metric', type=str, default='recall10',
                       choices=['recall5', 'recall10', 'recall20', 'ndcg5', 'ndcg10', 'ndcg20', 'train_ndcg5'])
    parse.add_argument('--embedding_size', type=int, default=64)
    parse.add_argument('--train_batch_size', type=int, default=2048)
    parse.add_argument('--test_batch_size', type=int, default=2048)
    parse.add_argument('--learn_rate', type=float, default=0.002)
    parse.add_argument('--critique_rate', type=float, default=0.002)
    parse.add_argument('--reg_lambda', type=float, default=0.0001)
    parse.add_argument('--gcn_layer', type=int, default=3)
    parse.add_argument('--test_frequency', type=int, default=1)
    parse.add_argument('--sparsity_test', type=int, default=0)
    parse.add_argument('--tau', type=float, default=0.28)
    parse.add_argument('--ssl_lambda', type=float, default=5.0)
    parse.add_argument('--encoder', type=str, default='MF')
    # critique train/eval caliber must match the backbone encoder (MF or GCN) to avoid a caliber mismatch
    parse.add_argument('--critique_encoder', type=str, default='MF', choices=['MF', 'GCN'])

    parse.add_argument('--num_bpr_neg', type=int, default=10)
    parse.add_argument('--num_cl_neg', type=int, default=1000)
    parse.add_argument('--critique_cl_lambda', type=float, default=1.0)
    parse.add_argument('--critique_pos_lambda', type=float, default=0.8)
    parse.add_argument('--critique_neg_lambda', type=float, default=1.0)
    parse.add_argument('--lambda_cf', type=float, default=0.8)
    parse.add_argument('--lambda_cross', type=float, default=0.0,
                       help='bilinear consistency coefficient between CF and critique scores (0 = legacy additive fusion)')
    parse.add_argument('--prd_pos_lambda', type=float, default=0.0,
                       help='preference-ranking distillation weight on positive candidates (0 = legacy)')
    parse.add_argument('--prd_tau', type=float, default=0.05,
                       help='PRD tie threshold: pairs with |score_i - score_j| > tau are counted')
    parse.add_argument('--prd_gap_weight', type=float, default=1.0,
                       help='PRD pair confidence weight power (1.0 = gap-weighted, 0.0 = uniform)')
    parse.add_argument('--force_neg_anchor', action='store_true', default=False)
    # diagnostic perturbations (EXP-M05/M09); all default off so legacy behavior is byte-identical
    parse.add_argument('--shuffle_neg_scores', action='store_true', default=False,
                       help='diagnostic: shuffle per-user neg scores to test signal authenticity')
    parse.add_argument('--reverse_neg_scores', action='store_true', default=False,
                       help='diagnostic: reverse neg scores to test signal direction')
    parse.add_argument('--neg_dropout', type=float, default=0.0,
                       help='diagnostic: randomly drop a fraction of neg candidates (0 = off)')
    parse.add_argument('--score_noise_std', type=float, default=0.0,
                       help='diagnostic: add Gaussian noise to neg scores (0 = off)')
    parse.add_argument('--perturb_seed', type=int, default=2025,
                       help='random seed for diagnostic perturbations')

    # in-training CF gating on neg items (per-neg weight derived from a precomputed backbone CF table)
    parse.add_argument('--neg_gate_mode', type=str, default='none',
                       choices=['none', 'linear', 'power', 'softmax', 'hard_topk', 'hard_thresh'],
                       help='in-training CF gating shape on neg items (none = legacy, no gating)')
    parse.add_argument('--neg_gate_gamma', type=float, default=5.0,
                       help='gating sharpness gamma (used by power/softmax modes)')
    parse.add_argument('--neg_gate_topk', type=int, default=5,
                       help='number of high-CF neg items kept in hard_topk mode')
    parse.add_argument('--neg_gate_cftable', type=str, default=None,
                       help='path to precomputed backbone CF table JSON (required when mode != none)')
    parse.add_argument('--neg_gate_invert', action='store_true', default=False,
                       help='invert gating direction (default False = high-CF neg kept)')

    # Non-Propagating Critique residual head options (backbone frozen variants)
    parse.add_argument('--npc_residual', action='store_true', default=False,
                       help='non-propagating critique: neg via a user-only residual head, detached from GCN')
    parse.add_argument('--npc_full', action='store_true', default=False,
                       help='NPC full: backbone frozen, all critique losses via the user-only residual head')
    parse.add_argument('--npc_prop', action='store_true', default=False,
                       help='NPC propagating residual: backbone frozen, residual re-propagated through aggregate()')
    parse.add_argument('--npc_prop_neg', action='store_true', default=False,
                       help='NPC polarity-aware: pos/CL via propagating head, neg via non-propagating head')
    parse.add_argument('--npc_alpha', type=float, default=1.0,
                       help='NPC residual (propagating head) weight; residual is zero-init, alpha scales it')
    parse.add_argument('--npc_alpha_neg', type=float, default=1.0,
                       help='NPC non-propagating neg head weight (decoupled from the propagating head)')

    parse.add_argument('--out', type=str, default=None,
                       help='generator output path override')
    parse.add_argument('--critique_round_path', type=str, default='./experiments/top200/critique_round/',
                       help='critique_round directory read by simulate')
    parse.add_argument('--backbone', type=str, default=None,
                       help='backbone weights path override; defaults to model_save/best_model_LightCCF.pth')
    parse.add_argument('--save_name', type=str, default=None,
                       help='backbone save-name override (no path, no .pth); defaults to best_model_LightCCF')

    return parse.parse_args()
