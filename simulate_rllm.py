"""Entry point: load frozen backbone, build the residual-LLM Critique model, run critique training."""
import torch
import numpy as np
import random
import logging
from datetime import datetime
import utility.parser_rllm as parser
import utility.tools as tools
import utility.critique_data_loader as data_loader
import utility.critique_trainer_rllm as critique_trainer
import os, sys
from Critique_rllm import Critique

os.chdir(sys.path[0])

def load_param():
    args = parser.parse_args()
    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("\t device:" + str(device) + str(args.gpu))
    return args, device

def load_log(args, model_name):
    if not os.path.exists('log/' + model_name):
        os.mkdir('log/' + model_name)
    if not os.path.exists('log/' + model_name + '/' + args.dataset):
        os.mkdir('log/' + model_name + '/' + args.dataset)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    if args.log == 'None':
        logfile = os.path.join('log/' + model_name + '/' + args.dataset, f'{timestamp}.log')
    else:
        logstr = str(args.log)
        logfile = os.path.join('log/' + model_name + '/' + args.dataset, f'{logstr}.log')
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.info("-----" * 10)
    logger.info(args)
    return logger


def load_models():
    model_name = "LightCCF"

    import_str = 'from ' + model_name + ' import Trainer'
    imported_module = {}
    exec(import_str, imported_module)
    Trainer = imported_module['Trainer']

    return model_name, Trainer


def load_dataset(args):
    dataset = data_loader.Data(args)
    return dataset


def build_user_fp_by_round(args):
    """Per-round FP sets for incremental gating: {r: {user: {item: score}}} where round r accumulates rounds 0..r."""
    R = int(args.critique_round_num)
    schedule = getattr(args, 'integ_gate_schedule', 'strength')

    if schedule == 'strength':
        import json as _json
        fp_f = getattr(args, 'integ_fp_file', None)
        if not fp_f or not os.path.exists(fp_f):
            raise FileNotFoundError(f"--integ_gate_schedule=strength requires --integ_fp_file (constrained dislike json): {fp_f}")
        raw = _json.load(open(fp_f, encoding='utf-8'))
        per_user = {}
        for u_str, lst in raw.items():
            u = int(u_str)
            pairs = [(int(e[0]), float(e[1]) if len(e) > 1 else 1.0) for e in lst]
            pairs.sort(key=lambda x: -x[1])
            per_user[u] = pairs
        by_round = {r: {} for r in range(R)}
        for u, pairs in per_user.items():
            n = len(pairs)
            for r in range(R):
                k = max(1, int(round(n * (r + 1) / R)))
                k = min(k, n)
                by_round[r][u] = {it: sc for it, sc in pairs[:k]}
        n_u = len(per_user)
        print(f"\t [rllm gate_schedule=strength] constrained dislike split into {R} rounds by descending score: {n_u} users")
        return by_round

    elif schedule == 'round_path':
        round_path = args.critique_round_path
        accum = {}
        by_round = {r: {} for r in range(R)}
        for r in range(R):
            neg_f = os.path.join(round_path, f"test_{r}_neg.txt")
            sc_f = os.path.join(round_path, f"test_{r}_neg_scores.txt")
            if os.path.exists(neg_f):
                scores_map = {}
                if os.path.exists(sc_f):
                    for line in open(sc_f):
                        p = line.split()
                        scores_map.setdefault(int(p[0]), []).extend(map(float, p[1:]))
                for line in open(neg_f):
                    p = line.split()
                    u = int(p[0])
                    items = list(map(int, p[1:]))
                    scs = scores_map.get(u, [1.0] * len(items))
                    for idx, it in enumerate(items):
                        s = scs[idx] if idx < len(scs) else 1.0
                        if it not in accum.setdefault(u, {}) or s > accum[u][it]:
                            accum[u][it] = s
            by_round[r] = {u: dict(d) for u, d in accum.items()}
        print(f"\t [rllm gate_schedule=round_path] test_{{0..r}}_neg accumulated over {R} rounds: round{R-1} "
              f"{len(by_round[R-1])} users accumulated")
        return by_round

    raise ValueError(f"Unknown integ_gate_schedule: {schedule}")


if __name__ == "__main__":

    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print('\t Loading parameter file...')
    args, device = load_param()
    print('-' * 80)

    print('\t Loading model ...')
    model_name, Trainer = load_models()
    print('-' * 80)

    print('\t Loading logger...')
    logger = load_log(args, model_name)
    print('-' * 80)
    print('\t Loading dataset file...')
    dataset = load_dataset(args)
    print('-' * 80)
    print('\t Loading recommender and run...')
    lightccf_trainer = Trainer(args, dataset, device, logger)
    LightCCF_model = lightccf_trainer.model

    model_save_path = args.backbone if args.backbone else f"model_save/best_model_LightCCF.pth"
    if not os.path.exists(model_save_path):
        logger.error(f"Error: Model file not found at {model_save_path}. Please train the model first.")

    print(f"Loading best model parameters from {model_save_path}...")
    state_dict = torch.load(model_save_path, map_location=device)
    LightCCF_model.load_state_dict(state_dict)

    print(f"LightCCF model successfully loaded from {model_save_path}.\n")

    incr_gate = getattr(args, 'integ_incremental_gate', False)
    user_fp_by_round = None
    if incr_gate:
        print('\t Building per-round FP set (per-round incremental gating, --integ_incremental_gate)...')
        user_fp_by_round = build_user_fp_by_round(args)
        last = user_fp_by_round[int(args.critique_round_num) - 1]
        n_fp_users = len(last)
        n_fp_items = sum(len(v) for v in last.values())
        print(f"\t Last round (round {int(args.critique_round_num)-1}) accumulated FP: {n_fp_users} users, "
              f"avg {n_fp_items/max(n_fp_users,1):.1f} FP/user")
    critique_model = Critique(
        args=args,
        original_user_embedding_weights=LightCCF_model.user_embedding.weight,
        original_item_embedding_weights=LightCCF_model.item_embedding.weight,
        user_fp_by_round=user_fp_by_round,
    )

    critique_trainer.critique_training(critique_model, args, dataset, device, logger)
