import torch
import random
import logging
import json
import numpy as np
from datetime import datetime
from LightCCF import Trainer
import utility.parser as parser
import utility.tools as tools
import utility.data_loader as data_loader
import os, sys
from collections import Counter, defaultdict 
from itertools import combinations

os.chdir(sys.path[0])

def load_param():
    args = parser.parse_args()
    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("\t device:" + str(device) + str(args.gpu))
    return args, device

def load_log(args, model_name):
    #if not os.path.exists('log/' + model_name):
        #os.mkdir('log/' + model_name)
    if not os.path.exists('log/' + model_name + '/' + args.dataset):
        os.makedirs('log/' + model_name + '/' + args.dataset)
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

def load_dataset(args):
    dataset = data_loader.Data(args)

    return dataset

def load_save_model(model, device):
    
    model_save_path = f"model_save/best_model_LightCCF.pth"
    if not os.path.exists(model_save_path):
        logger.error(f"Error: Model file not found at {model_save_path}. Please train the model first.")
        return

    print(f"Loading best model parameters from {model_save_path}...")
    state_dict = torch.load(model_save_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    print("Model parameters loaded successfully.")
    return model

def analyze_and_get_top_keyphrases(model, dataset, device, items2entity_file_path, num_keyphrases=5):

    model.eval()
    test_users_dict = dataset.test_dict
    all_positive_train = dataset.all_positive
    num_items = dataset.num_items
    
    # Load items to entities mapping
    mv_to_entities = defaultdict(set)
    try:
        with open(items2entity_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ')
                mv_id = int(parts[0])
                entities = [int(i) for i in parts[1:]]
                mv_to_entities[mv_id].update(entities)
    except FileNotFoundError:
        print(f"Error: File '{items2entity_file_path}' not found. Please check the path.")
        return {}

    user_top_keyphrases = defaultdict(list)
    
    for user_id_long in test_users_dict.keys():
        user_id = torch.tensor([user_id_long], dtype=torch.long).to(device)
        interacted_items_in_train = set(all_positive_train[user_id_long])

        with torch.no_grad():
            predictions = model.get_rating_for_test(user_id).squeeze(0)
        
        item_scores = list(zip(range(num_items), predictions.cpu().numpy()))
        sorted_scores = sorted(item_scores, key=lambda x: x[1], reverse=True)
        top_20_predicted_items = [
            item for item, score in sorted_scores 
            if item not in interacted_items_in_train and item not in test_users_dict.get(user_id_long, [])
        ][:20]

        top_items_entities = []
        for item_id in top_20_predicted_items:
            top_items_entities.extend(list(mv_to_entities.get(item_id, [])))
        entity_counts = Counter(top_items_entities)
        
        user_train_entities = set()
        for item_id in interacted_items_in_train:
            user_train_entities.update(mv_to_entities.get(item_id, []))
            
        sorted_entities = entity_counts.most_common()
        selected_keyphrases = []
        for entity, count in sorted_entities:
            if entity not in user_train_entities:
                selected_keyphrases.append(entity)

                if len(selected_keyphrases) >= num_keyphrases:
                    break

        if len(selected_keyphrases) < num_keyphrases:
            if selected_keyphrases:
                shortfall = num_keyphrases - len(selected_keyphrases)
                resampled_keyphrases = random.choices(selected_keyphrases, k=shortfall)
                selected_keyphrases.extend(resampled_keyphrases)
        
        if selected_keyphrases:
            user_top_keyphrases[user_id_long] = selected_keyphrases

    print("Top keyphrase analysis completed.")
    return user_top_keyphrases

def select_critiquing_by_keyphrase(user_keyphrases, dataset, entities2mv_file_path):
    
    # Load entities to items mapping
    entities_to_mv = defaultdict(list)
    try:
        with open(entities2mv_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ')
                entity_id = int(parts[0])
                mv_ids = [int(i) for i in parts[1:]]
                entities_to_mv[entity_id].extend(mv_ids)
    except FileNotFoundError:
        print(f"Error: File '{entities2mv_file_path}' not found. Please check the path.")
        return {}

    critiquing_with_keyphrases = defaultdict(list)
    
    selected_items_for_user = defaultdict(set)
    
    for u_id, keyphrases in user_keyphrases.items():
        interacted_items_in_train = set(dataset.all_positive[u_id])
        
        for keyphrase_id in keyphrases:
            candidate_items = [
                item for item in entities_to_mv.get(keyphrase_id, [])
                if item not in interacted_items_in_train and item not in dataset.test_dict.get(u_id, [])
            ]
            
            if candidate_items:
                available_candidates = [item for item in candidate_items if item not in selected_items_for_user[u_id]]
                
                if available_candidates:
                    selected_item = random.choice(available_candidates)
                else:
                    selected_item = random.choice(candidate_items)

                selected_items_for_user[u_id].add(selected_item)
                
                critiquing_with_keyphrases[u_id].append({'item_id': selected_item, 'keyphrase_id': keyphrase_id})
                
    print("False positive items selected.")
    return critiquing_with_keyphrases

def save_critiquing(critiquing_items, output_file_path, output_json_file_path):

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for u_id, item_info_list in critiquing_items.items():
                item_list = [item_info['item_id'] for item_info in item_info_list]
                if item_list:
                    items_str = ' '.join(map(str, item_list))
                    f.write(f"{u_id} {items_str}\n")
        print("False positive items saved successfully.")
        print(f"\n--- Saving false positive items to {output_json_file_path} ---")

        with open(output_json_file_path, 'w', encoding='utf-8') as f:
            json.dump(critiquing_items, f, ensure_ascii=False, indent=4)
        print("False positive items saved successfully.")
    except Exception as e:
        print(f"Error saving false positive items file: {e}")
    
def find_best_comparison_items(model, dataset, critiquing_items):

    model.eval()
    items_emb = model.item_embedding.weight.data.cpu().numpy()
    item_embeddings = {int(i): vec for i, vec in enumerate(items_emb)}
    train_users_dict = dataset.train_dict
    
    final_comparisons = defaultdict(list)
    
    for u_id, criticism_item_info_list in critiquing_items.items():
        user_pos_items = train_users_dict.get(u_id, [])
        if len(user_pos_items) < 2:
            continue
        
        pos_ids = [item_id for item_id in user_pos_items if item_id in item_embeddings]
        if len(pos_ids) < 2:
            continue
        
        pos_embeddings_matrix = np.array([item_embeddings[item_id] for item_id in pos_ids])
        dist_v1pos_v2pos = np.linalg.norm(pos_embeddings_matrix[:, np.newaxis, :] - pos_embeddings_matrix[np.newaxis, :, :], axis=2)

        for item_info in criticism_item_info_list:
            neg_item_id = item_info['item_id']
            best_S = float('inf')
            best_pair_ids = None
            
            if neg_item_id not in item_embeddings:
                continue
            vneg = item_embeddings[neg_item_id]
            dist_neg_pos = np.linalg.norm(pos_embeddings_matrix - vneg, axis=1)
            pos_item_indices_combinations = combinations(range(len(pos_ids)), 2)
            
            for i, j in pos_item_indices_combinations:
                S = dist_neg_pos[i] + dist_neg_pos[j] + dist_v1pos_v2pos[i][j]
                
                if S < best_S:
                    best_S = S
                    best_pair_ids = [pos_ids[i], pos_ids[j]]
            
            if best_pair_ids:
                final_comparisons[u_id].append([neg_item_id] + best_pair_ids)
                
    print("Comparison items found for all false positives.")
    return final_comparisons

def combine_and_save_data(comparison_items, critiquing_with_keyphrases, items2entity_file_path, entities2name_file_path, output_file_path):

    try:
        entities_2_name = {}
        with open(entities2name_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    entity_id, entity_name = int(parts[0]), parts[1]
                    entities_2_name[entity_id] = entity_name

    except FileNotFoundError as e:
        print(f"Error: Missing file - {e}")
        return
    
    combined_name_mapping = defaultdict(dict)
    
    for u_id, criticism_item_info_list in critiquing_with_keyphrases.items():
        if not criticism_item_info_list:
            continue
        
        user_map = []
        comparison_list_for_user = comparison_items.get(u_id, [])

        for item_info in criticism_item_info_list:
            neg_item_id = item_info['item_id']
            keyphrase_id = item_info['keyphrase_id']
            
            # Find the corresponding pos_items_ids in comparison_list
            pos_items_ids = None
            for comp_list in comparison_list_for_user:
                if comp_list[0] == neg_item_id:
                    pos_items_ids = comp_list[1:]
                    break
            
            if pos_items_ids:
                item_name = entities_2_name.get(neg_item_id, "")
                entity_name = entities_2_name.get(keyphrase_id, "")
                pos_items_names = [entities_2_name.get(p_id, "") for p_id in pos_items_ids]
                
                user_map.append({
                    "disliked_movie_id":neg_item_id,
                    "title": item_name,
                    "entity": entity_name,
                    "pos_items": pos_items_names
                })
        
        combined_name_mapping[str(u_id)] = user_map
        
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(combined_name_mapping, f, ensure_ascii=False, indent=4)
        print(f"Final combined file saved to '{output_file_path}'.")
    except Exception as e:
        print(f"Error saving file: {e}")


if __name__ == '__main__':
    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
    print('\t Analyzing prediction errors for test users...')
    print('-' * 80)

    print('\t Loading parameter file...')
    args, device = load_param()
    print('-' * 80)
    
    print('\t Loading logger...')
    logger = load_log(args, "false-pos")
    print('-' * 80)
    print('\t Loading dataset file...')
    dataset = load_dataset(args)
    print('-' * 80)
    print('\t Loading recommender and run...')

    print('\t Loading model ...')
    LightCCF_model = Trainer(args, dataset, device, logger)
    print('-' * 80)
    
    data_path = "./dataset/movielens-1m/"
    output_folder = "prepare_for_LLM/"
    if not os.path.exists(data_path + output_folder):
        os.makedirs(output_folder)
        
    items2entity_file_path = os.path.join(data_path, "mv2entities.txt")
    entities2name_file_path = os.path.join(data_path, "entities2name.txt")
    entities2mv_file_path = os.path.join(data_path, "entities2mv.txt")

    output_critique_file_path = os.path.join(data_path + output_folder, "user_comparsion_critique.json")
    critiquing_json_file_path = os.path.join(data_path + output_folder, "json_critiquing.json")
    critiquing_file_path = os.path.join(data_path + output_folder, "critiquing.txt")

    save_model = load_save_model(LightCCF_model.model, device)
    if save_model is None:
        sys.exit("Model not loaded. Exiting.")
        
    user_top_keyphrases = analyze_and_get_top_keyphrases(save_model, dataset, device, items2entity_file_path, num_keyphrases=5)
    critiquing_with_keyphrases = select_critiquing_by_keyphrase(user_top_keyphrases, dataset, entities2mv_file_path)
    save_critiquing(critiquing_with_keyphrases, critiquing_file_path, critiquing_json_file_path)
    comparison_items = find_best_comparison_items(save_model, dataset, critiquing_with_keyphrases)
    combine_and_save_data(comparison_items, critiquing_with_keyphrases, items2entity_file_path, entities2name_file_path, output_critique_file_path)
    
    print("\nProcess finished.")