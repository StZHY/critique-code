import torch
import numpy as np
import random
import logging
from datetime import datetime
import utility.parser as parser
import utility.tools as tools
import utility.critique_data_loader as data_loader
import utility.critique_trainer as critique_trainer
import os, sys
from Critique import Critique

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


if __name__ == "__main__":
    
    """fix the random seed"""
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
    
    model_save_path = f"model_save/best_model_LightCCF.pth"
    if not os.path.exists(model_save_path):
        logger.error(f"Error: Model file not found at {model_save_path}. Please train the model first.")
    
    print(f"Loading best model parameters from {model_save_path}...")
    state_dict = torch.load(model_save_path, map_location=device)
    LightCCF_model.load_state_dict(state_dict)
    
    print(f"LightCCF loaded from {model_save_path}\n")
    
    critique_model = Critique(
        args=args,
        original_user_embedding_weights=LightCCF_model.user_embedding.weight,
        original_item_embedding_weights=LightCCF_model.item_embedding.weight,
    )
    
    critique_trainer.critique_training(critique_model, args, dataset, device, logger)