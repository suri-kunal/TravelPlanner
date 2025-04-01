import os
import re
import sys
from travelplanner.agents.prompts import planner_agent_prompt, cot_planner_agent_prompt, react_planner_agent_prompt,react_reflect_planner_agent_prompt,reflect_prompt
# from utils.func import get_valid_name_city,extract_before_parenthesis, extract_numbers_from_filenames
import json
import time
from langchain_community.callbacks.manager import get_openai_callback

from tqdm import tqdm
from travelplanner.tools.planner.apis import Planner, ReactPlanner, ReactReflectPlanner
import openai
import argparse
from datasets import load_dataset

def load_line_json_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n'):
            unit = json.loads(line)
            data.append(unit)
    return data

def extract_numbers_from_filenames(directory):
    # Define the pattern to match files
    pattern = r'annotation_(\d+).json'

    # List all files in the directory
    files = os.listdir(directory)

    # Extract numbers from filenames that match the pattern
    numbers = [int(re.search(pattern, file).group(1)) for file in files if re.match(pattern, file)]

    return numbers


def catch_openai_api_error():
    error = sys.exc_info()[0]
    if error == openai.error.APIConnectionError:
        print("APIConnectionError")
    elif error == openai.error.RateLimitError:
        print("RateLimitError")
        time.sleep(60)
    elif error == openai.error.APIError:
        print("APIError")
    elif error == openai.error.AuthenticationError:
        print("AuthenticationError")
    else:
        print("API error:", error)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="validation")
    parser.add_argument("--create_subset", type=bool, default=False)
    parser.add_argument("--result_prefix", type=str, default="generated_plan_")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--strategy", type=str, default="direct")
    args = parser.parse_args()
    directory = f'{args.output_dir}/{args.set_type}'

    if args.set_type == 'train':
        query_data_list  = load_dataset('osunlp/TravelPlanner','train')['train']
    elif args.set_type == 'validation':
        query_data_list  = load_dataset('osunlp/TravelPlanner','validation')['validation']
    else:
        query_data_list  = load_dataset('osunlp/TravelPlanner','test')['test']
    result_prefix = args.result_prefix

    if args.create_subset:
        # Create a subset of the dataset
        query_data_list = query_data_list.train_test_split(test_size=0.2,shuffle=False,seed=42)["test"]
        result_prefix = f"sample_{args.result_prefix}"
    
    numbers = [i for i in range(1,len(query_data_list)+1)]

    if args.strategy == 'direct':
        planner = Planner(model_name=args.model_name, agent_prompt=planner_agent_prompt)
    elif args.strategy == 'cot':
        planner = Planner(model_name=args.model_name, agent_prompt=cot_planner_agent_prompt)
    elif args.strategy == 'react':
        planner = ReactPlanner(model_name=args.model_name, agent_prompt=react_planner_agent_prompt)
    elif args.strategy == 'reflexion':
        planner = ReactReflectPlanner(model_name=args.model_name, agent_prompt=react_reflect_planner_agent_prompt,reflect_prompt=reflect_prompt)


    with get_openai_callback() as cb:
        for number in tqdm(numbers[:]):
            
            query_data = query_data_list[number-1]
            reference_information = query_data['reference_information']
            while True:
                    if args.strategy in ['react','reflexion']:
                        planner_results, scratchpad  = planner.run(reference_information, query_data['query'])
                    else:
                        planner_results  = planner.run(reference_information, query_data['query'])
                    if planner_results != None:
                        break
            print(planner_results)
            # check if the directory exists
            if not os.path.exists(os.path.join(f'{args.output_dir}/{args.set_type}')):
                os.makedirs(os.path.join(f'{args.output_dir}/{args.set_type}'))
            if not os.path.exists(os.path.join(f'{args.output_dir}/{args.set_type}/{result_prefix}{number}.json')):
                result =  [{}]
            else:
                result = json.load(open(os.path.join(f'{args.output_dir}/{args.set_type}/{result_prefix}{number}.json')))
            if args.strategy in ['react','reflexion']:
                result[-1][f'{args.model_name}_{args.strategy}_sole-planning_results_logs'] = scratchpad 
            result[-1][f'{args.model_name}_{args.strategy}_sole-planning_results'] = planner_results
            # write to json file
            with open(os.path.join(f'{args.output_dir}/{args.set_type}/{result_prefix}{number}.json'), 'w') as f:
                json.dump(result, f, indent=4)
        print(cb)
