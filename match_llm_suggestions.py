import os, sys
import json
import time
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from tqdm import tqdm


def create_movie_lookup(movie_data_path):

    try:
        movies_df = pd.read_csv(
            movie_data_path,
            sep='\t',
            header=None,
            engine='python',
            encoding='utf-8',
            names=['rehased_id', 'ID', 'fb_id', 'Title']
        )
    except Exception as e:
        print(f"error {e}")
        return None, None, None

    movies_df['Title'] = movies_df['Title'].apply(lambda x: re.sub(r'\[.*?\]', '', x).strip())
    
    vectorizer = TfidfVectorizer(stop_words='english')
    movie_vectors = vectorizer.fit_transform(movies_df['Title'])

    return movies_df, vectorizer, movie_vectors

def load_user_data(train_path):

    user_history = defaultdict(set)
    
    def parse_file(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(' ')
                    if len(parts) < 2:
                        continue
                    user_id = parts[0]
                    movie_ids = parts[1:]
                    user_history[user_id].update(movie_ids)
        except FileNotFoundError:
            print(f"canot find: {file_path}")
        except Exception as e:
            print(f"read {file_path} error: {e}")

    parse_file(train_path)

    return user_history

def find_best_match(suggested_title, movies_df, vectorizer, movie_vectors):
   
    suggested_vector = vectorizer.transform([suggested_title])

    cosine_similarities = cosine_similarity(suggested_vector, movie_vectors).flatten()

    best_match_index = cosine_similarities.argmax()
    best_match_score = cosine_similarities[best_match_index]

    if best_match_score > 0.8:
        matched_id = int(movies_df.iloc[best_match_index]['rehased_id'])
        return matched_id
    
    return None

def process_and_write_data(
    suggestions_data,
    user_error,
    num_rounds,
    data_path,
    base_filename
):
    all_round_ids = defaultdict(list)
    all_round_scores = defaultdict(list)

    try:
        for user_id, error_list in user_error.items():
            
            user_suggestions = suggestions_data.get(str(user_id), {})

            for round_idx in range(num_rounds):
                
                round_movie_ids = []
                round_movie_scores = []

                mv_id = user_error[int(user_id)][round_idx]
                round_movie_ids.append(mv_id)
                round_movie_scores.append('1.0')
                
                if round_idx < len(error_list):
                    disliked_movie = error_list[round_idx]
                    suggested_list = user_suggestions.get(disliked_movie, [])

                    for suggestion in suggested_list:
                        movie_id = suggestion.get("id")
                        movie_llm_score = suggestion.get("probability")
                        round_movie_ids.append(str(movie_id))
                        round_movie_scores.append(str(movie_llm_score))

                all_round_ids[round_idx].append(f"{user_id} {' '.join(round_movie_ids)}")
                all_round_scores[round_idx].append(f"{user_id} {' '.join(round_movie_scores)}")
                
        for round_count in range(num_rounds):
            output_filename = os.path.join(data_path, f"{base_filename}_{round_count}.txt")
            output_score_filename = os.path.join(data_path, f"{base_filename}_{round_count}_scores.txt")

            if round_count in all_round_ids:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    for line in all_round_ids[round_count]:
                        f.write(line + '\n')

                with open(output_score_filename, 'w', encoding='utf-8') as f:
                    for line in all_round_scores[round_count]:
                        f.write(line + '\n')
                
                print(f"{round_count+1} round data write in {base_filename}_{round_count}.txt and {base_filename}_{round_count}_scores.txt")
            else:
                print(f"error {round_count+1} round has no data")

    except Exception as e:
        print(f"error: {e}")

def main():
    user_critiquing_file = './dataset/movielens-1m/prepare_for_LLM/critiquing.txt'
    suggestions_input_file = './dataset/movielens-1m/LLM_request_back/new_llm_cri_suggestions.json'
    matched_output_file = './dataset/movielens-1m/LLM_request_back/matched_suggestions.json'
    movies_data_path = './dataset/movielens-1m/mv_id_name_align.txt'
    critique_round_data_path = './dataset/movielens-1m/critique_round/'
    train_data_path = './dataset/movielens-1m/train.txt'
    num_rounds = 5

    user_critiquing = {}
    with open(user_critiquing_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ')
            user_id = int(parts[0])
            user_critiquing[user_id] = parts[1:]
            
    movies_df, vectorizer, movie_vectors = create_movie_lookup(movies_data_path)
    if movies_df is None:
        return

    user_history = load_user_data(train_data_path)

    if not os.path.exists(suggestions_input_file):
        print(f"error: {suggestions_input_file}")
        return
    
    with open(suggestions_input_file, 'r', encoding='utf-8') as f:
        llm_suggestions = json.load(f)
    
    all_matched_suggestions = defaultdict(dict)
    
    for user_id, user_suggestions in tqdm(llm_suggestions.items(), ascii=True):

        matched_user_suggestions = {}
        user_movies_to_exclude = user_history.get(user_id, set())

        for disliked_movie, suggested_list in user_suggestions.items():
            matched_movies_with_prob = []
            if suggested_list is None:
                matched_movies_with_prob.append({'id': disliked_movie, 'probability': 1.0})
            else:
                for suggested_items in suggested_list:
                    
                    parts = suggested_items.split("::")
                    if len(parts) == 3:
                        suggested_id = parts[0]
                        suggested_title, probability = parts[1], parts[2]
                    else:
                        print(f"error: {user_id},{disliked_movie} has no score.")
                        return
                        
                    matched_id = find_best_match(suggested_title, movies_df, vectorizer, movie_vectors)
                    
                    if matched_id == suggested_id:
                        matched_id = int(suggested_id)
                    else:
                        continue

                    if matched_id and str(matched_id) not in user_movies_to_exclude:
                        matched_movies_with_prob.append({'id': matched_id, 'probability': probability})
                        
            if matched_movies_with_prob is None:
                matched_movies_with_prob.append({'id': disliked_movie, 'probability': 1.0})

            matched_user_suggestions[disliked_movie] = matched_movies_with_prob
        
        all_matched_suggestions[user_id] = matched_user_suggestions
            
    final_output = {
        "llm suggestions with IDs": dict(all_matched_suggestions)
    }
    
    with open(matched_output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
        
    print(f"finised")

    process_and_write_data(all_matched_suggestions, user_critiquing, num_rounds, critique_round_data_path, base_filename="train")

if __name__ == '__main__':
    main()