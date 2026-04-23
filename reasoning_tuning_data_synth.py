from utils.LLMs import GPT_model
from utils.prompts import *
from utils.utils import location_direction_dict
import json
import re
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import random

LLM = GPT_model(model='gpt-4')

root_path = './data/FinetuneData/'
init_file_path = root_path + 'SynTrain_sample_all.json'
total_data = []

with open(init_file_path, 'r') as file:
    init_data_list = json.load(file)

data_list = []
for data in init_data_list:
    if data['Timestep'] == 0 or len(data['Current_Observation']['non_empty_lanes']) == 0:
        continue
    data_list.append(data)

print('len data list', len(data_list))

# Shuffle data_list randomly
random.shuffle(data_list)
if len(data_list) > 1000:
    new_data_list = data_list[:1000]
    remaining_data = data_list[1000:]
    # Save the remaining part as env_rl_data.json
    with open(root_path + 'env_rl_data.json', 'w') as f:
        json.dump(remaining_data, f, indent=4)
    data_list = new_data_list
else:
    print("data_list length is less than 1000, no sampling will be performed.")


def extract_json(response):
    try:
        match = re.search(r"({.*})", response.strip(), re.DOTALL)
        if match:
            json_data = match.group(1)  # Extract the JSON body
            data = json.loads(json_data)
            return data
        else:
            print("No JSON data found in the response.")
            return ""
            
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Parse error: {e}")


def convert_dict_to_numbered_text(input_dict):
    numbered_lines = [
        f"{index + 1}. **{key}**:\n{value}\n"
        for index, (key, value) in enumerate(input_dict.items())
    ]
    
    # Join the text lines into a single paragraph
    return "\n".join(numbered_lines)

def gpt4_json_reponse_generate(prompt, system_prompt):
    retries = 0
    max_retries = 5
    while retries < max_retries:
        try:
            response = LLM.ask(prompt, system_prompt)
            # Try to parse the framework
            data = extract_json(response)
            # If parsing is successful, return the data
            return data
        
        except Exception as e:
            # Catch parsing failure exceptions
            retries += 1
            print(f"Parse error: {e}, retrying... ({retries}/{max_retries})")
    raise RuntimeError("Maximum retries reached. Failed to parse framework.")

def get_historical_observation_atr(timestep, Traffic_state_history):
    # get the history before timestep t-1
    h_w_size = 5
    
    history_state = Traffic_state_history[-h_w_size-1:-1]

    timestep_start = timestep - len(history_state)
    text = "## Historical Observation\n"
    text += "- Lanes include both signal-controlled lanes at the current intersection (e.g., NL, NT, SL, ST, EL, ET, WL, WT) and upstream/downstream lanes from neighboring intersections (e.g., SL's upstream lane (4, ST)). **Only lanes with vehicles are shown.**\n"
    text += "- Values are shown as: value or value(+change from previous timestep).\n\n"
    for i in range(len(history_state)):
        text += f"Timestep {timestep_start + i} signal: {history_state[i]['Signal']}\n"
        text += f"Timestep {timestep_start + i + 1} traffic states:\n"
        text += "|Lane|Cars Input|Cars Output|Queued Cars|Moving Cars|Average Waiting Time (mins)|Occupancy (%)|\n"
        signal_consequence = history_state[i]
        for lane in signal_consequence:
            if lane in ['Signal', 'Summary']:
                continue
            lane_data = signal_consequence[lane]
            exist = lane_data['Occupancy (%)'] + lane_data['Cars Input'] + lane_data['Cars Output']
            if not exist:
                continue
            text += f"|{lane}|{int(lane_data['Cars Input'])}|{int(lane_data['Cars Output'])}|{int(lane_data['Queued Cars'])}({'+' if lane_data['Queued Cars Change']>=0 else ''}{int(lane_data['Queued Cars Change'])})|{int(lane_data['Moving Cars'])}({'+' if lane_data['Moving Cars Change']>=0 else ''}{int(lane_data['Moving Cars Change'])})|{lane_data['Average Waiting Time (mins)']}({'+' if lane_data['Average Waiting Time Change (mins)']>=0 else ''}{lane_data['Average Waiting Time Change (mins)']})|{lane_data['Occupancy (%)']}({'+' if lane_data['Occupancy Change (%)']>=0 else ''}{lane_data['Occupancy Change (%)']})|\n"
        text += '\n'
    if len(history_state) > 0:
        return text
    else:
        return ""
    
def get_current_observation_atr(step_num, Traffic_state_history):
    signal = Traffic_state_history[-1]['Signal']
    text = "## Current Observation\n"
    text += f"Timestep {step_num} signal: {signal}\n"
    return text

def get_current_observation_ra(step_num, Traffic_state_history):
    signal = Traffic_state_history[-1]['Signal']
    text = "# Current Observation\n"
    text += "- Lanes include both signal-controlled lanes at the current intersection (e.g., NL, NT, SL, ST, EL, ET, WL, WT) and upstream/downstream lanes from neighboring intersections (e.g., SL's upstream lane (4, ST)). **Only lanes with vehicles are shown.**\n"
    text += "- Values are shown as: value or value(+change from previous timestep).\n\n"
    text += f"Timestep {step_num-1} signal: {signal}\n"
    current_state = Traffic_state_history[-1]
    text += f"Timestep {step_num} traffic states:\n"
    text += "|Lane|Cars Input|Cars Output|Queued Cars|Moving Cars|Average Waiting Time (mins)|Occupancy (%)|\n"
    for lane in current_state:
        if lane in ['Signal', 'Summary']:
            continue
        lane_data = current_state[lane]
        exist = lane_data['Occupancy (%)'] + lane_data['Cars Input'] + lane_data['Cars Output']
        if not exist:
            continue
        text += f"|{lane}|{int(lane_data['Cars Input'])}|{int(lane_data['Cars Output'])}|{int(lane_data['Queued Cars'])}({'+' if lane_data['Queued Cars Change']>=0 else ''}{int(lane_data['Queued Cars Change'])})|{int(lane_data['Moving Cars'])}({'+' if lane_data['Moving Cars Change']>=0 else ''}{int(lane_data['Moving Cars Change'])})|{lane_data['Average Waiting Time (mins)']}({'+' if lane_data['Average Waiting Time Change (mins)']>=0 else ''}{lane_data['Average Waiting Time Change (mins)']})|{lane_data['Occupancy (%)']}({'+' if lane_data['Occupancy Change (%)']>=0 else ''}{lane_data['Occupancy Change (%)']})|\n"
    text += '\n'
    return text

def get_local_signal_priority_text(signal_value_dict):
    text = "# Local Signal Priority\n"
    text += "This rank only considers your own intersection and assumes the downstream allows the release.\n"
    text += "|Rank|Signal|Waiting Time Reduction|\n"
    for i, p in enumerate(sorted(signal_value_dict, key=signal_value_dict.get, reverse=True)):
        text += "|{}|{}|{:.1f} mins|\n".format(i+1, p, signal_value_dict[p]/60)
    text += '\n'
    return text

def get_atr_user_prompt(data):
    t0 = data['Timestep']-1
    t1 = data['Timestep']
    historical_observation = get_historical_observation_atr(t0, data['Traffic_state_history'])
    current_observation = get_current_observation_atr(t1, data['Traffic_state_history'])
    prompt = get_advance_traffic_reasoning_prompt(t0, t1, historical_observation, current_observation)
    return prompt

def get_atr_user_prompt_adaptive(data, complex_level):
    t0 = data['Timestep']-1
    t1 = data['Timestep']
    historical_observation = get_historical_observation_atr(t0, data['Traffic_state_history'])
    current_observation = get_current_observation_atr(t1, data['Traffic_state_history'])
    prompt = get_advance_traffic_reasoning_prompt_adaptive(t0, t1, historical_observation, current_observation, complex_level)
    return prompt

def get_ra_user_prompt(data, atr_results):
    t0 = data['Timestep']-1
    t1 = data['Timestep']
    current_observation_text = get_current_observation_ra(t1, data['Traffic_state_history'])
    local_signal_priority = get_local_signal_priority_text(data['Signal_Rank'])
    prompt = get_reactive_action_prompt(t0, t1, advance_traffic_reasoning_results=atr_results, current_observation=current_observation_text, local_signal_priority=local_signal_priority)
    return prompt

def critic_neighbor_check(data):
    neighboring_critic_exist = False
    if len(data['Traffic_state_history']) > 1:
        traffic_state = data['Traffic_state_history'][-2]
        for lane in traffic_state:
            if "upstream" in lane or "downstream" in lane:
                if traffic_state[lane]['Occupancy (%)'] >= 40:
                    neighboring_critic_exist = True
                    break

    return neighboring_critic_exist

def get_atr_results_by_LLM(data):
    neighboring_critic_exist = critic_neighbor_check(data)
    
    if neighboring_critic_exist:
        complex_level = 'complex'
        atr_prompt = get_atr_user_prompt_adaptive(data, complex_level='complex')
    else:
        complex_level = 'easy'
        atr_prompt = get_atr_user_prompt_adaptive(data, complex_level='easy')
    atr_results = LLM.ask(atr_prompt, system_prompt=SYSTEM_PROMPT)

    return atr_results, complex_level

def get_ra_results_by_LLM(ra_user_prompt):
    ra_results = LLM.ask(ra_user_prompt, system_prompt=SYSTEM_PROMPT)
    return ra_results

for i, data in enumerate(tqdm(data_list)):
    new_item = {}
    if data['Timestep'] == 0:
        continue
    
    atr_user_prompt = get_atr_user_prompt(data)
    atr_results, complex_level = get_atr_results_by_LLM(data)
    ra_user_prompt = get_ra_user_prompt(data, atr_results)
    ra_results = get_ra_results_by_LLM(ra_user_prompt)

    total_data.append({
        'atr_user_prompt': atr_user_prompt,
        'atr_output': atr_results,
        'ra_user_prompt': ra_user_prompt,
        'ra_output': ra_results,
        'data': data,
        'complex_level': complex_level
    })

    print('total consume: {}'.format(LLM.total_consume))
    
    # Save total_data
    with open(root_path + 'syn_rt_data.json', 'w') as f:
        json.dump(total_data, f, indent=4)