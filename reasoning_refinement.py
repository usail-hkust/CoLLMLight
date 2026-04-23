import argparse
from utils.LLMs import LLAMA_model
from utils.my_utils import calculate_reactive_action_reward, calculate_advance_reasoning_reward
import json
import re
from tqdm import tqdm
from collections import defaultdict
import random
from utils.prompts import *

# 4. Use the newly generated data and original data to construct a preference dataset, and also build a fine-tuning dataset.
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model_path', type=str, default="Your model path", help='Path to the model')
args = parser.parse_args()
model_path = args.model_path

root_path = './data/FinetuneData/'
init_file_path = root_path + 'SynTrain_sample_all.json'

with open(init_file_path, 'r') as file:
    init_data_list = json.load(file)

random.shuffle(init_data_list)
data_list = init_data_list
print('len data list', len(data_list))

# 1. Load the model
LLM = LLAMA_model(model=model_path)
sft_data = []
kto_data = []
dpo_data = []

# 2. Iterate through all data scenarios, extract incorrect signals, or signals with a reward below a certain threshold.
def run_batch_test(LLM, dataset_list, batch_size=100):
    """"""
    # Prepare all ATR prompts
    prompts_atr = [get_atr_user_prompt(data) for data in dataset_list]
    
    # Generate all ATR responses in batches of batch_size
    print("Generating ATR responses...")
    total_response_atr = []
    for i in tqdm(range(0, len(prompts_atr), batch_size)):
        responses_atr = LLM.batch_ask(prompts_atr[i:i+batch_size], system_prompt=SYSTEM_PROMPT)
        total_response_atr.extend(responses_atr)
        
    # Prepare all RA prompts
    prompts_ra = [get_ra_user_prompt(dataset_list[i], total_response_atr[i]) for i in range(len(dataset_list))]
    
    # Generate all RA responses in batches of batch_size
    print("Generating RA responses...")
    total_response_ra = []
    for i in tqdm(range(0, len(prompts_ra), batch_size)):
        responses_ra = LLM.batch_ask(prompts_ra[i:i+batch_size], system_prompt=SYSTEM_PROMPT)
        total_response_ra.extend(responses_ra)
    
    # Calculate rewards and save data
    print("Calculating rewards...")
    reward_ra_list = []
    reward_atr_list = []
    results_data = []
    for i, data in enumerate(dataset_list):
        reward_ra, used_atr = calculate_reactive_action_reward(total_response_ra[i], data['Best_Signal'])
        reward_atr = calculate_advance_reasoning_reward(total_response_atr[i], reward_ra, used_atr)
        reward_ra_list.append(reward_ra)
        reward_atr_list.append(reward_atr)
        results_data.append({
            'data': data,
            'response_atr': total_response_atr[i],
            'response_ra': total_response_ra[i],
            'reward_atr': reward_atr,
            'reward_ra': reward_ra,
            'prompt_atr': prompts_atr[i],
            'prompt_ra': prompts_ra[i]
        })
    avg_ra_reward = sum(reward_ra_list) / len(reward_ra_list)
    print('avg ra reward', avg_ra_reward)
    avg_atr_reward = sum(reward_atr_list) / len(reward_atr_list)
    print('avg atr reward', avg_atr_reward)
    
    return results_data

def kto_format_item(user_content, assistant_content, label):
    item = {}
    item['messages'] = []
    system_msg_item = {}
    system_msg_item['content'] = SYSTEM_PROMPT
    system_msg_item['role'] = 'system'
    user_msg_item = {}
    user_msg_item['content'] = user_content
    user_msg_item['role'] = 'user'
    assistant_msg_item = {}
    assistant_msg_item['content'] = assistant_content
    assistant_msg_item['role'] = 'assistant'
    item['messages'].append(system_msg_item)
    item['messages'].append(user_msg_item)
    item['messages'].append(assistant_msg_item)
    item['label'] = label
    return item

def dpo_format_item(user_content, assistant_content_good, assistant_content_bad, label_good):
    item = {}
    item['conversations'] = []
    # conver_item1 = {}
    # conver_item1['from'] = 'system'
    # conver_item1['value'] = SYSTEM_PROMPT
    conver_item2 = {}
    conver_item2['from'] = 'human'
    conver_item2['value'] = SYSTEM_PROMPT + '\n\n' + user_content
    # item['conversations'].append(conver_item1)
    item['conversations'].append(conver_item2)
    item['chosen'] = {}
    item['chosen']['from'] = 'gpt'
    item['rejected'] = {}
    item['rejected']['from'] = 'gpt'
    if label_good:
        item['chosen']['value'] = assistant_content_good
        item['rejected']['value'] = assistant_content_bad
    else:
        item['chosen']['value'] = assistant_content_bad
        item['rejected']['value'] = assistant_content_good
    return item

def get_kto_items(good_item, bad_item):
    kto_item_list = []
    kto_item1 = kto_format_item(good_item['prompt_atr'], good_item['response_atr'], good_item['reward_atr'] >= bad_item['reward_atr'])
    kto_item2 = kto_format_item(bad_item['prompt_atr'], bad_item['response_atr'], good_item['reward_atr'] < bad_item['reward_atr'])
    kto_item_list.append(kto_item1)
    kto_item_list.append(kto_item2)
    
    # ra item
    kto_item3 = kto_format_item(good_item['prompt_ra'], good_item['response_ra'], good_item['reward_ra'] >= bad_item['reward_ra'])
    kto_item4 = kto_format_item(bad_item['prompt_ra'], bad_item['response_ra'], good_item['reward_ra'] < bad_item['reward_ra'])
    kto_item_list.append(kto_item3)
    kto_item_list.append(kto_item4)
    return kto_item_list

def get_dpo_items(good_item, bad_item, LLM):
    dpo_item_list = []
    dpo_item_atr = dpo_format_item(good_item['prompt_atr'], good_item['response_atr'], bad_item['response_atr'], good_item['reward_atr'] > bad_item['reward_atr'])
    dpo_item_list.append(dpo_item_atr)
    prompt_ra_batch = [good_item['prompt_ra'], bad_item['prompt_ra']]
    response_ra_batch = LLM.batch_ask(prompt_ra_batch, system_prompt=SYSTEM_PROMPT)
    good_item_another_response_ra = response_ra_batch[0]
    good_item_another_ra_reward = calculate_reactive_action_reward(good_item_another_response_ra, good_item['data']['Best_Signal'])[0]
    bad_item_another_response_ra = response_ra_batch[1]
    bad_item_another_ra_reward = calculate_reactive_action_reward(bad_item_another_response_ra, bad_item['data']['Best_Signal'])[0]

    if good_item_another_ra_reward != good_item['reward_ra']:
        dpo_item_ra = dpo_format_item(good_item['prompt_ra'], good_item['response_ra'], good_item_another_response_ra, good_item['reward_ra'] > good_item_another_ra_reward)
        dpo_item_list.append(dpo_item_ra)
    if bad_item_another_ra_reward != bad_item['reward_ra']:
        dpo_item_ra = dpo_format_item(bad_item['prompt_ra'], bad_item_another_response_ra, bad_item['response_ra'], bad_item_another_ra_reward > bad_item['reward_ra'])
        dpo_item_list.append(dpo_item_ra)
    return dpo_item_list

def get_sft_items(good_item):
    sft_item_list = []
    item1 = {}
    item1['instruction'] = SYSTEM_PROMPT
    item1['input'] = good_item['prompt_atr']
    item1['output'] = good_item['response_atr']
    item2 = {}
    item2['instruction'] = SYSTEM_PROMPT
    item2['input'] = good_item['prompt_ra']
    item2['output'] = good_item['response_ra']
    sft_item_list.append(item1)
    sft_item_list.append(item2)
    return sft_item_list

def update_data(new_results_data, wrong_results_data, LLM):
    for i, item in enumerate(new_results_data):
        if item['reward_ra'] == 1:
            kto_items = get_kto_items(item, wrong_results_data[i])
            dpo_items = get_dpo_items(item, wrong_results_data[i], LLM)
            sft_items = get_sft_items(item)
            kto_data.extend(kto_items)
            dpo_data.extend(dpo_items)
            sft_data.extend(sft_items)

def save_data():
    num = 2
    with open(root_path + 'sft_refine_{}.json'.format(num), 'w') as f:
        json.dump(sft_data, f, indent=4)
    with open(root_path + 'kto_refine_{}.json'.format(num), 'w') as f:
        json.dump(kto_data, f, indent=4)
    with open(root_path + 'dpo_refine_{}.json'.format(num), 'w') as f:
        json.dump(dpo_data, f, indent=4)

# Iterate through and test all scenarios
results_data = run_batch_test(LLM, data_list)
# Get scenarios with incorrect signal selection
wrong_data_list = [item['data'] for item in results_data if item['reward_ra'] != 1]
wrong_results_data = [item for item in results_data if item['reward_ra'] != 1]
print('init wrong data len: ', len(wrong_data_list))

llm_params = {
    "temperature": 1.0
}
LLM = LLAMA_model(llm_params=llm_params, model=model_path)

for _ in tqdm(range(10)):
    new_results_data = run_batch_test(LLM, wrong_data_list)
    update_data(new_results_data, wrong_results_data, LLM)
    wrong_data_list = [item['data'] for item in new_results_data if item['reward_ra'] != 1]
    wrong_results_data = [item for item in new_results_data if item['reward_ra'] != 1]
    print('wrong data len: ', len(wrong_data_list))
    save_data()

## Save data