import os
import json
import sys
import time
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional

from volcenginesdkarkruntime import Ark

all_suggestions = defaultdict(dict)


def create_cached_context(client: Ark, dataset_path: str, model_id: str) -> Optional[str]:

    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            movie_lines = f.readlines()

        movie_list_str = ""
        for line in movie_lines:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                movie_id = parts[0]
                movie_name_and_year = parts[3]
                movie_list_str += f"{movie_id}::{movie_name_and_year}\n"

        if not movie_list_str:
            print("error", file=sys.stderr)
            return None

        system_prompt = f"""You are a movie recommendation engine with a specific, critical task. Your goal is to predict which movies a user will *dislike* based on their feedback.
                    <MasterMovieList>
                    {movie_list_str.strip()}
                    </MasterMovieList>

                    You will be given the user's taste profile in the user's message. Your task is to act as a filter. You must follow these reasoning steps precisely:

                    <ReasoningSteps>
                    1.  **Analyze the User's Dislike:** First, look at especially the '<CritiqueKeyword>'. This keyword is the most important clue. Identify the core reason for the user's dislike.
                    2.  **Understand the User's Likes:** Next, look at the two movies the user liked. These serve as a contrast. They show you what the user is *not* looking for. For example, if they dislike a slow drama but like action movies, you should find other slow movies, not action movies.
                    3.  **Scan and Filter the Master Movie List:** Go through the entire <MasterMovieList> provided above. For each movie, ask yourself: "Does this movie share the same negative traits identified in Step 1?".
                    4.  **Select, Score, and Constrain:** From your filtered list, you MUST select the **TOP 20 to 25** movies that are the absolute strongest matches. You must score them based on the following rubric.

                    <ScoringRubric>
                    - **High Confidence (0.8 - 1.0):** Assign this score if the movie is a *direct and strong match* for the user's dislike. For example, if the <CritiqueKeyword> is 'slow and boring', a high-confidence movie would be another notoriously slow-paced drama. It MUST also be a clear opposite of the user's liked movies.
                    - **Medium Confidence (0.5 - 0.7):** Assign this score for movies that are a *thematic match* but not a direct one. It might be in a similar genre or style that the user likely dislikes, but doesn't perfectly match the critique keyword.
                    - **Low Confidence (< 0.5):** DO NOT include movies with a low confidence score. Your final list should only contain medium to high confidence recommendations.
                    </ScoringRubric>
                    </ReasoningSteps>

                    **CRITICAL OUTPUT FORMATTING:**
                    - Your entire output MUST be a single, raw JSON object.
                    - Do NOT include any explanations, introductory text, or markdown formatting like ```json.
                    - The key of the JSON object MUST be the ID of the disliked movie from the user prompt.
                    - The value MUST be a list of strings, with each string formatted as 'mv_id::mv_name::score'.
                    - **The list MUST contain between 20 and 25 movies. DO NOT exceed this limit.**

                    **EXAMPLE:**
                    ---
                    **Hypothetical User Prompt:**
                    "
                    The ID of the movie that the user disliked is:
                    <DislikedMovieID>
                    25
                    </DislikedMovieID>
                    The title of the disliked movie is:
                    <DislikedMovieTitle>
                    Dead Man Walking (1995)
                    </DislikedMovieTitle>
                    The critique keyword for the disliked movie is: 'slow and boring'.
                    The two movies the user liked for comparison are: 'Star Wars: Episode IV - A New Hope (1977)' and 'The Matrix (1999)'.
                    "
                    ---
                    **Your CORRECT and COMPLETE output for this example would be:**
                    {{
                        "25": [
                            "123::AnotherSlowDrama (1998)::0.95",
                            "456::ThoughtProvokingFilm (2002)::0.90",
                            "789::CharacterStudy (1976)::0.88"
                        ]
                    }}
                    ---
                    """

        response = client.responses.create(
            model=model_id,
            input=[
                {
                    "role": "system",
                    "content": system_prompt
                }
            ],
            caching={"type": "enabled"},
            thinking={"type": "disabled"},
        )
        print(response.usage.model_dump_json())
        context_id = response.id
        print(f"New Context ID: {context_id}")
        
        return context_id

    except FileNotFoundError:
        print(f"error: cannot find '{dataset_path}'。", file=sys.stderr)
        return None
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return None


def main():

    start = datetime.now()

    data_path = "./dataset/movielens-1m/"
    movie_dataset_file = "mv_id_name_align.txt"
    input_file = "prepare_for_LLM/user_comparsion_critique.json"
    suggestions_output_file = "LLM_request_back/llm_cri_suggestions.json"
    context_id_file = "context_id.txt"
    model_endpoint_id = ""

    existing_context_id = None

    input_path = os.path.join(data_path, input_file)
    movie_dataset_path = os.path.join(data_path, movie_dataset_file)
    output_path = os.path.join(data_path, suggestions_output_file)
    context_id_path = os.path.join(data_path, context_id_file)

    print("Initializing Ark client...")
    client = Ark(
        base_url='XXXXXXXXXXXXXXXXXXXXXXXXXXXX',
        api_key=os.environ.get("ARK_API_KEY"),
    )

    context_id = existing_context_id
    if not context_id and os.path.exists(context_id_path):
        with open(context_id_path, 'r', encoding='utf-8') as f:
            context_id_from_file = f.read().strip()
            if context_id_from_file:
                context_id = context_id_from_file
                

    if not context_id:
        context_id = create_cached_context(client, movie_dataset_path, model_endpoint_id)
        
        with open(context_id_path, 'w', encoding='utf-8') as f:
            f.write(context_id)

    if not context_id:
        print("error: cannot create or load a valid context ID.", file=sys.stderr)
        return

    if not os.path.exists(input_path):
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content:
                    loaded_suggestions = json.loads(content)
                    global all_suggestions
                    all_suggestions = defaultdict(dict, loaded_suggestions)
        except json.JSONDecodeError:
            print("error")

    tasks_to_process = []
    user_ids = list(all_data.keys())
    users_to_process = user_ids[0:20]

    for user_id in users_to_process:
        user_data = all_data.get(user_id, {})

        for info in user_data:
            disliked_movie_id = info.get("disliked_movie_id", "")

            if user_id in all_suggestions and disliked_movie_id in all_suggestions[user_id]:
                continue
            
            tasks_to_process.append((user_id, info))

    if not tasks_to_process:
        print("finished")
        return
    
    print(f" {len(tasks_to_process)} new task to process.")

    for i, (user_id, info) in enumerate(tasks_to_process):
        disliked_movie_id = info.get("disliked_movie_id", "")
        movie_title = info.get("title", "")
        critique_keyword = info.get("entity", "")
        pos_items = info.get("pos_items", [])

        print(f"\n---process {i+1}/{len(tasks_to_process)} | user: {user_id}, movie ID: {disliked_movie_id} ---")

        prompt_text = f"""
            The ID of the movie that the user disliked is:
            <DislikedMovieID>
            {disliked_movie_id}
            </DislikedMovieID>
            The title of the disliked movie is:
            <DislikedMovieTitle>
            {movie_title}
            </DislikedMovieTitle>
            The critique keyword for the disliked movie is: '{critique_keyword}'.
            The two movies the user liked for comparison are: '{pos_items[0]}' and '{pos_items[1]}'.
            """

        try:
            completion = client.responses.create(
                model=model_endpoint_id,
                previous_response_id=context_id,
                input=[{"role": "user", "content": prompt_text}],
                caching={"type": "enabled"},
                thinking={"type": "disabled"}
            )

            content_str = completion.output[0].content[0].text
            print(f"back data: {content_str}")
            
            start_marker = '```json'
            end_marker = '```'

            suggestions_data = None

            cleaned_json_str = content_str.strip()
            if cleaned_json_str.startswith(start_marker):
                cleaned_json_str = cleaned_json_str[len(start_marker):]
            if cleaned_json_str.endswith(end_marker):
                cleaned_json_str = cleaned_json_str[:-len(end_marker)]
            
            suggestions_data = json.loads(cleaned_json_str.strip())


            if user_id not in all_suggestions:
                all_suggestions[user_id] = {}
            
            if isinstance(suggestions_data, dict):
                for key, new_value in suggestions_data.items():
                    if key in all_suggestions[user_id]:
                        all_suggestions[user_id][key].extend(new_value)
                    else:
                        all_suggestions[user_id].update(suggestions_data)

            else:
                all_suggestions[user_id][disliked_movie_id] = None 

        except json.JSONDecodeError as e:
            print(f"!!! JSON failed: {e}", file=sys.stderr)
            print(f"!!! error: {content_str}", file=sys.stderr)
            if user_id not in all_suggestions:
                all_suggestions[user_id] = {}
            all_suggestions[user_id][disliked_movie_id] = f"ERROR: JSONDecodeError - {str(e)}"

        except Exception as e:
            print(f"!!! error: {e}", file=sys.stderr)

            if user_id not in all_suggestions:
                all_suggestions[user_id] = {}
            all_suggestions[user_id][disliked_movie_id] = f"ERROR: {str(e)}"

        finally:

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_suggestions, f, ensure_ascii=False, indent=4)
            print(f"Progress saved to {output_path}")
            time.sleep(1)

    end = datetime.now()
    print(f"\nFinished processing all tasks.")
    print(f"Total execution time: {end - start}")


if __name__ == "__main__":
    if not os.environ.get("ARK_API_KEY"):
        print("error ARK_API_KEY", file=sys.stderr)
    else:
        main()
