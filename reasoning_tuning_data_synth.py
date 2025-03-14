from utils.LLMs import GPT_model
import json
import re
from tqdm import tqdm
from collections import defaultdict

LLM = GPT_model(model = 'gpt-4')


root_path = './data/FinetuneData/'
init_file_path = root_path + 'SynTrain_sample.json'
# exist_data_path = root_path + 'old_data_ft_aql2.json'
new_data_phase1 = []
new_data = []

# exist_data_list = []
# with open(exist_data_path, 'r') as f:
#     exist_data = json.load(f)
# for item in exist_data:
#     exist_data_list.append(item['data'])

with open(init_file_path, 'r') as file:
    init_data_list = json.load(file)

# data_list = []
# for item in init_data_list:
#     if item not in exist_data_list:
#         data_list.append(item)

# data_list = data_list[-10:]

data_list = init_data_list
print('len data list', len(data_list))


def extract_json(response):
    try:
        match = re.search(r"({.*})", response.strip(), re.DOTALL)
        if match:
            json_data = match.group(1)  # 提取 JSON 正文部分
            data = json.loads(json_data)
            return data
        else:
            print("No JSON data found in the response.")
            return ""
        
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Parse error: {e}")


def convert_dict_to_numbered_text(input_dict):  
    """  
    将字典转换为带序号的文本段落，关键词加粗并换行  
    
    参数:  
    input_dict (dict): 输入的字典  
    
    返回:  
    str: 带序号的文本段落  
    """  
    # 使用enumerate生成带序号的文本行  
    numbered_lines = [  
        f"{index + 1}. **{key}**:\n{value}\n"   
        for index, (key, value) in enumerate(input_dict.items())  
    ]  
    
    # 将文本行连接成一个段落  
    return "\n".join(numbered_lines)  

    

def gpt4_json_reponse_generate(prompt, system_prompt):
    retries = 0
    max_retries = 5
    while retries < max_retries:
        try:
            response = LLM.ask(prompt, system_prompt)
            # print(response)
            # 尝试解析框架
            data = extract_json(response)
            # print(data)
            # 如果解析成功，返回框架
            return data
        
        except Exception as e:
            # 捕获解析失败的异常
            retries += 1
            print(f"Parse error: {e}, retrying... ({retries}/{max_retries})")
    raise RuntimeError("Maximum retries reached. Failed to parse framework.")

def get_current_observation_text(current_observation):
    empty_lanes = current_observation['empty_lanes']
    text = "#### **Intersection Lanes - Controlled by You:** \n"
    text += "|Lane|Queued Cars|Moving Cars|Average Waiting Time (mins)|Occupancy|\n"
    non_empty_lanes_observation = current_observation['non_empty_lanes']
    for lane in non_empty_lanes_observation:
        lane_info = non_empty_lanes_observation[lane]
        text += "|{}|{}|{}|{:.1f}|{:.1f}%|\n".format(lane, lane_info['queue_car_num'], lane_info['coming_car_num'], lane_info['avg_wait_time'], lane_info['occupancy'])
    text += '{} are empty.\n'.format(empty_lanes)
    text += '\n'
    
    information_text = "#### **Nearby Upstream and Downstream Lanes - Controlled by Other Intersections near You:** \n"
    data_title_text = "We use (inter_id, lane) to represent these lanes. For instance, (1, 'NL') represents the NL lane at Intersection 1.\n"
    data_title_text += "|Relation|Queued Cars|Moving Cars|Average Waiting Time (mins)|Occupancy|\n"
    upstream_down_stream_text = ''
    for lane in current_observation['non_empty_lanes']:
        for direc in ['upstream', 'downstream']:
            if direc in current_observation['up_down_stream_view'][lane]:
                stream_lanes_data = current_observation['up_down_stream_view'][lane][direc]
                for stream_lane in stream_lanes_data:
                    stream_lane_data = stream_lanes_data[stream_lane]
                    downstream_inter_lane = (stream_lane_data['inter_id'], stream_lane)
                    not_empty = len(stream_lane_data['veh2pos'])
                    if not_empty:
                        upstream_down_stream_text += "|Your {}'s {} lane {}|{}|{}|{:.1f}|{:.1f}%|\n".format(lane, direc, downstream_inter_lane, stream_lane_data['queue_len'], sum(stream_lane_data['cells']), stream_lane_data['avg_wait_time']/60, stream_lane_data['occupancy']*100)

    if len(upstream_down_stream_text) == 0:
        upstream_down_stream_text = "All nearby upstream and downstream lanes are in good state with low occupancy.\n"
        text = text + information_text + upstream_down_stream_text
    else:
        text = text + information_text + data_title_text + upstream_down_stream_text
    text += '\n'
    # long distance
    no_empty_lanes = list(current_observation['non_empty_lanes'].keys())
    long_distance_text = get_long_distance_text(current_observation['long_distance_info'], no_empty_lanes)
    if len(long_distance_text) != 0:
        text = text + long_distance_text
    text += '\n'
    
    return text


def get_long_distance_text(long_distance_info, no_empty_lanes):
    long_distance_exist = False 
    long_distance_text = "#### **Long-distance upstream and downstream information:** \n"
    long_distance_text += "|Relation|The number of lanes whose occupancy exceeds half|Queued Cars|Average Waiting Time (mins)|Average Occupancy|\n"
    for lane in no_empty_lanes:
        if len(long_distance_info[lane]['exist']) > 0:
            for direc in long_distance_info[lane]['exist']:
                long_distance_exist = True
                long_distance_text += "|{}' {}|{}|{}|{:.1f}|{:.1f}%| \n".format(lane, direc, long_distance_info[lane][direc]['lane_num'], long_distance_info[lane][direc]['total_queue_num'], long_distance_info[lane][direc]['average_waiting_time']/60, long_distance_info[lane][direc]['average_occupancy']*100)

    if not long_distance_exist:
        return ""
    else:
        return long_distance_text
    

def get_historical_observation(data):
    timestep = data['Timestep']
    history_state = data['Traffic_state_history']
    timestep_start = timestep - len(history_state)
    text = "### Historical Observation"
    for i in range(len(history_state)):
        text += f"Timestep: {timestep_start + i}\n"
        text += f"Signal: {history_state[i]['Signal']}\n"
        text += "|Lane|Cars Input|Cars Output|Queued Cars Change|Queued Cars|Moving Cars Change|Moving Cars|Average Waiting Time Change (mins)|Average Waiting Time (mins)|Occupancy Change (%)|Occupancy (%)|\n"
        signal_consequence = history_state[i]
        for lane in signal_consequence:
            if lane in ['Signal', 'Summary']:
                continue
            lane_data = signal_consequence[lane]
            exist = lane_data['Occupancy (%)'] + lane_data['Cars Input'] + lane_data['Cars Output']
            if not exist:
                continue
            text += f"|{lane}|{lane_data['Cars Input']}|{lane_data['Cars Output']}|{lane_data['Queued Cars Change']}|{lane_data['Queued Cars']}|{lane_data['Moving Cars Change']}|{lane_data['Moving Cars']}|{lane_data['Average Waiting Time Change (mins)']}|{lane_data['Average Waiting Time (mins)']}|{lane_data['Occupancy Change (%)']}|{lane_data['Occupancy (%)']}|\n"
        text += '\n'
    memory_text = get_memory_text(data)
    text += memory_text
    text += '\n'
    return text

def get_memory_text(data):
    recent_memory = data['Current_Observation']['memory']['recent_memory']
    lane_memories = data['Current_Observation']['memory']['lane_memories']
    similar_lane_memory = data['Current_Observation']['memory']['similar_lane_memory']
    no_empty_lanes = list(data['Current_Observation']['non_empty_lanes'].keys())
    prompt = "#### **Past Lane Activation Data (Memory):**\n"
    prompt += "There were some lane activation cases and results in the past. Based on these data, you can better estimate the influence of each signal in the current situation.\n"
    ## recent_memory
    memory_exist = False
    recent_memory = False
    if recent_memory:
        recent_memory_txt = "In the most recent period, you selected signal {}. Here are the results:\n".format(recent_memory['signal'])
        if len(recent_memory['pos']) > 0:
            recent_memory = True
            for i in range(len(recent_memory['pos'])):
                lane, memory_idx  = recent_memory['pos'][i]
                lane_memory = lane_memories[i]
                recent_memory_txt += "- Your {} lane ({:.1f}% occupancy, {} queue cars, {} moving cars before) has released {} cars".format(lane, lane_memory['occupancy_before']*100, lane_memory['queue_num_before'], lane_memory['moving_num_before'], len(lane_memory['release_results']))
                downstream_ct = defaultdict(int)
                for veh in lane_memory['release_results']:
                    if lane_memory['release_results'][veh]:
                        downstream_ct[lane_memory['release_results'][veh]] += 1
                if len(downstream_ct) > 0:
                    for downstream_lane in downstream_ct:
                        recent_memory_txt += ", {} cars have released to {}".format(downstream_ct[downstream_lane], downstream_lane)
                recent_memory_txt += "\n"
        if recent_memory == True:
            memory_exist = True
            prompt += recent_memory_txt

    if len(similar_lane_memory) > 0:
        memory_exist = True
        prompt += "There are also other relevant past cases similar to the current situation: \n"
        for lane_memory in similar_lane_memory:
            prompt += "- Your {} lane ({:.1f}% occupancy, {} queue cars, {} moving cars before) has released {} cars after an activation".format(lane_memory['lane'], lane_memory['occupancy_before']*100, lane_memory['queue_num_before'], lane_memory['moving_num_before'], len(lane_memory['release_results']))
            downstream_ct = defaultdict(int)
            for veh in lane_memory['release_results']:
                if lane_memory['release_results'][veh]:
                        downstream_ct[lane_memory['release_results'][veh]] += 1
            if len(downstream_ct) > 0:
                for downstream_lane in downstream_ct:
                    prompt += ", {} cars have released to {}".format(downstream_ct[downstream_lane], downstream_lane)
            prompt += "\n"
    
    
    if memory_exist:
        return prompt
    else:
        return ""


            
def get_back_ground_and_note_text():
    input = "## Background Context\n"
    input += "An intersection has 12 lanes: [NL, NT, NR, SL, ST, SR, EL, ET, ER, WL, WT, WR]. Each lane is labeled by direction and movement: N for north, S for south, E for east, W for west, L for left turn, T for through, and R for right turn. For instance, ET stands for the East Through lane, where traffic moves straight ahead from east to west. WL is the West Left-turn lane, where traffic turns left from west to south. Right turns are always allowed. There are four signal options: [ETWT, NTST, ELWL, NLSL]. For example, ETWT indicates the release of both the ET and WT lanes. The signal phase duration is set to thirty seconds.\n\n"
    input += "## Note:\n"
    input += """For each Lane X, when considering activating it, keep these in mind: 
- NEVER let the occupancy of X's ANY downstream lane be close to 100% at any risk, as it will cause severe congestion.
- If the upstream or downstream information of X isn't mentioned, it means that they are in a good state with low occupancy.
- You MUST consider how much the occupancy of X's downstream lanes will increase upon releasing lane X. 
- You MUST delay the release of X if ANY of its downstream has a high occupancy rate.
- If there are many high-occupancy lanes upstream of X and X's occupancy is not low, you MUST consider releasing X so as to help upstream lanes release.
- You can't keep a lane waiting for too long. You MUST release the lane with excessive waiting time when the downstream condition allows.\n\n"""
    return input

def get_phase1_results(data):
    system_prompt = "You are a traffic signal expert responsible for managing a four-way intersection. Your primary goal is to assess the current coordination level and implement appropriate signal selection strategies. Aim to optimize traffic flow and safety, considering not only your intersection but also the impacts on adjacent upstream and downstream intersections."
    input = get_back_ground_and_note_text()
    input += "## Data\n"
    input += "### Current Observation\n"
    input += f"Timestep: {data['Timestep']}\n"
    current_observation_text = get_current_observation_text(data['Current_Observation'])
    input += current_observation_text
    input += "## Task:\n"
    input += """Analyze the data in "Current Observation" and classify the intersection into one of three coordination scenarios:
   - **No-Coordination**: No upstream or downstream sections currently have occupancy levels that require attention and coordination.
   - **Simple-Coordination**: Few upstream or downstream lanes require coordination with low traffic complexity.
   - **Complex-Coordination**: High traffic complexity with significant congestion in upstream or downstream lanes.\n\n"""
    input += "## Requirement:\n"
    input += """Please provide a JSON formatted output as follows: 
```json  
{  
  "phase1": {  
    "thought_process": "Your thought process description",  
    "answer": "NO" | "Simple" | "Complex"  
  }
}
```"""
    phase1_results = gpt4_json_reponse_generate(input, system_prompt)
    return phase1_results

def get_phase2_results_by_fast_thinking(data):
    system_prompt = "You are a traffic signal expert responsible for managing a four-way intersection. Your primary goal is to assess the current coordination level and implement appropriate signal selection strategies. Aim to optimize traffic flow and safety, considering not only your intersection but also the impacts on adjacent upstream and downstream intersections."
    input = get_back_ground_and_note_text()
    input += "## Data\n"
    input += "### Current Observation\n"
    input += f"Timestep: {data['Timestep']}\n"
    current_observation_text = get_current_observation_text(data['Current_Observation'])
    input += current_observation_text
    input += "## Task:\n"
    input += """- Focus on the "Current Observation" data.
- Keep the "Notes" in mind, compare each signal, and select the optimal one based on traffic conditions.\n\n"""
    input += "## Requirement:\n"
    input += """Please provide a JSON formatted output as follows:
```json
{
  "phase2": {  
    "thought_process": "Your thought process description",  
    "answer": "ETWT" | "NTST" | "ELWL" | "NLSL"  
  }
}
``` """
    phase2_results = gpt4_json_reponse_generate(input, system_prompt)
    return phase2_results

def get_first_stage_analysis(data):
    system_prompt = "You are a traffic signal expert responsible for managing a four-way intersection. Your primary goal is to assess the current coordination level and implement appropriate signal selection strategies. Aim to optimize traffic flow and safety, considering not only your intersection but also the impacts on adjacent upstream and downstream intersections."
    input = get_back_ground_and_note_text()
    input += "## Data\n"
    input += "### Current Observation\n"
    input += f"Timestep: {data['Timestep']}\n"
    current_observation_text = get_current_observation_text(data['Current_Observation'])
    input += current_observation_text
    input += "## Task:\n"
    input += """Conduct a comprehensive analysis by following these steps:
1. **Current Traffic Data Analysis**: Rank lanes based on queued cars and waiting times.
2. **Upstream Analysis**: Assess the conditions of nearby and long-distance upstream lanes for each lane, which lanes has a high upstream pressure.
3. **Downstream Analysis**: Assess the conditions of nearby and long-distance downstream lanes for each lane. Specifically, determine if the occupancy of any downstream lane approaches 100% or over 100%.\n\n"""
    input += "## Requirement:\n"
    input += """Please provide a JSON formatted output as follows:
```json  
{  
  "Current Traffic Data Analysis": "Your analysis text description",  
  "Upstream Analysis": "Your analysis text description",  
  "Downstream Analysis": "Your analysis text description"  
}  
``` """
    first_stage_analysis = gpt4_json_reponse_generate(input, system_prompt)
    first_stage_analysis = convert_dict_to_numbered_text(first_stage_analysis)
    return first_stage_analysis


def get_phase2_results_by_slow_thinking(data):
    first_stage_analysis = get_first_stage_analysis(data)
    signal_consequence_text = get_signal_consequence_text(data)
    final_results = get_final_results(first_stage_analysis, signal_consequence_text)
    return final_results

def get_signal_consequence_text(data):
    # timestep = data['Timestep']
    signal_consequece = data['Signal_Consequence']
    text = "4. **Signal Consequence Prediction**\n"
    for signal in signal_consequece:
        consequence = signal_consequece[signal]
        text += f"**If activate {signal}**:\n"
        text += "|Lane|Cars Input|Cars Output|Queued Cars Change|Queued Cars|Moving Cars Change|Moving Cars|Average Waiting Time Change (mins)|Average Waiting Time (mins)|Occupancy Change (%)|Occupancy (%)|\n"
        for lane in consequence:
            lane_data = consequence[lane]
            text += f"|{lane}|{lane_data['Cars Input']}|{lane_data['Cars Output']}|{lane_data['Queued Cars Change']}|{lane_data['Queued Cars']}|{lane_data['Moving Cars Change']}|{lane_data['Moving Cars']}|{lane_data['Average Waiting Time Change (mins)']}|{lane_data['Average Waiting Time (mins)']}|{lane_data['Occupancy Change (%)']}|{lane_data['Occupancy (%)']}|\n"
        text += '\n'
    return text

def get_signal_rank_text(signal_value_dict):
    text = "### Signal Priority (local)\n"
    text += "This rank only consider your own intersection, and assume the downstream allow the release.\n"
    text += "|Rank|Signal|Waiting Time Reduction|\n"
    for i, p in enumerate(sorted(signal_value_dict, key=signal_value_dict.get, reverse=True)):
        text += "|{}|{}|{:.1f} mins|\n".format(i+1, p, signal_value_dict[p]/60)
    text += '\n'
    return text

def get_final_results(first_stage_analysis, signal_consequence_text):
    system_prompt = "You are a traffic signal expert responsible for managing a four-way intersection. Your primary goal is to assess the current coordination level and implement appropriate signal selection strategies. Aim to optimize traffic flow and safety, considering not only your intersection but also the impacts on adjacent upstream and downstream intersections."
    input_text = get_back_ground_and_note_text()
    input_text += "## Your analysis\n"
    input_text += first_stage_analysis
    input_text += '\n'
    input_text += signal_consequence_text
    input_text += "## Task:\n"
    input_text += "You need to complete the rest of the analysis and select the optimal signal based on the analysis results.\n"
    input_text += """5. **Signal Comparison**: Systematically compare the effects of each signal option. 
6. **Decision Making**: Make a decision on the optimal signal to activate, remember the note, NEVER let the occupancy of X's ANY downstream lane be close to 100% at any risk, as it will cause severe congestion.\n\n"""
    input_text += "## Requirement:\n"
    input_text += """Please provide a JSON formatted output as follows:
```json
{  
  "Signal Comparison": "Your Analysis Text Description",  
  "Decision Making": "ETWT" | "NTST" | "ELWL" | "NLSL"  
}
``` """
    final_results = gpt4_json_reponse_generate(input_text, system_prompt)
    phase2thought = first_stage_analysis + '\n' + signal_consequence_text
    phase2thought += "5. **Signal Comparison**:\n"
    phase2thought += final_results['Signal Comparison']
    phase2thought += '\n\n'
    phase2thought += "6. **Decision Making**:\n"
    phase2thought += final_results['Decision Making']
    phase2answer = final_results['Decision Making']
    final_phase2_results = {
        "phase2": {
            "thought_process": phase2thought,
            "answer": phase2answer
        }
    }

    return final_phase2_results


def get_phase2_results(phase1_type, data):
    

    system_prompt = "You are a traffic signal expert responsible for managing a four-way intersection. Your primary goal is to assess the current coordination level and implement appropriate signal selection strategies. Aim to optimize traffic flow and safety, considering not only your intersection but also the impacts on adjacent upstream and downstream intersections."
    if phase1_type.lower() == 'no':
        signal_rank = data['Signal_Rank']
        best_signal = max(signal_rank, key=signal_rank.get)  
        phase2_thought = "Currently, we have no upstream or downstream coordination needs. The first signal in the 'Signal Priority (local)' is optimal, as it significantly reduces waiting times at our intersection without causing downstream congestion."
        phase2_answer = best_signal
        phase2_results = {
            "phase2": {
                "thought_process": phase2_thought,
                "answer": phase2_answer
            }
        }
        return phase2_results
    elif phase1_type.lower() == 'simple':
        phase2_results = get_phase2_results_by_fast_thinking(data)
        return phase2_results
    elif phase1_type.lower() =='complex':
        phase2_results = get_phase2_results_by_slow_thinking(data)
        return phase2_results
    else:
        raise ValueError("Invalid phase1_type. Expected 'NO', 'Simple', or 'Complex'.")

def get_input_text(data):
    input_text = get_back_ground_and_note_text()
    input_text += "## Data\n"
    historical_observation = get_historical_observation(data)
    input_text += historical_observation
    input_text += "### Current Observation\n"
    input_text += f"Timestep: {data['Timestep']}\n"
    current_observation_text = get_current_observation_text(data['Current_Observation'])
    signal_rank_text = get_signal_rank_text(data['Signal_Rank'])
    input_text += current_observation_text
    input_text += '\n'
    input_text += signal_rank_text
    input_text += "## Task:\n"
    input_text += """Your task consists of two phases:
### Phase 1: Coordination Scenario Classification
1. Analyze the data in "Current Observation" and classify the intersection into one of three coordination scenarios:
   - **No-Coordination**: No upstream or downstream coordination needs.
   - **Simple-Coordination**: Few upstream or downstream lanes require coordination with low traffic complexity.
   - **Complex-Coordination**: High traffic complexity with significant congestion in upstream or downstream lanes.

### Phase 2: Signal Selection Strategy
Based on the identified coordination scenario, select the appropriate signal using the following strategies:

#### a. No-Coordination
- Directly select the highest-ranked priority signal, as there are no external coordination considerations.

#### b. Simple-Coordination
- Focus on the "Current Observation" data.
- Keep the "Notes" in mind, compare each signal, and select the optimal one based on traffic conditions.

#### c. Complex-Coordination
Conduct a comprehensive analysis and select the optimal signal by following these steps:
1. **Current Traffic Data Analysis**: Rank lanes based on queued cars and waiting times.
2. **Upstream Analysis**: Assess the conditions of nearby and long-distance upstream lanes for each lane.
3. **Downstream Analysis**: Assess the conditions of nearby and long-distance downstream lanes for each lane.
4. **Signal Consequence Prediction**: Utilize "Historical Observation" data for inferring the consequences of each signal. 
5. **Signal Comparison**: Systematically compare the effects of each signal option.
6. **Decision Making**: Make a decision on the optimal signal to activate.\n\n"""

    input_text += "## Requirement:\n"
    input_text += """Please provide a JSON formatted output as follows: 
```json  
{  
  "phase1": {  
    "thought_process": "Your thought process text description",  
    "answer": "NO" | "Simple" | "Complex"  
  },  
  "phase2": {  
    "thought_process": "Your thought process text description",  
    "answer": "ETWT" | "NTST" | "ELWL" | "NLSL"  
  }  
}
``` """

    return input_text

def get_output_text(phase1_results, phase2_results):

    combined_results = {  
        "phase1": {  
            "thought_process": phase1_results['phase1']['thought_process'],  
            "answer": phase1_results['phase1']['answer']  
        },  
        "phase2": {  
            "thought_process": phase2_results['phase2']['thought_process'],  
            "answer": phase2_results['phase2']['answer']  
        }  
    }  
    
    # 生成带有代码块标记的JSON字符串  
    result_string = "```json\n" + json.dumps(combined_results, indent=2) + "\n```"  
    return result_string


## phase1 identification
for i, data in enumerate(tqdm(data_list)):
    new_item = {}
    phase1_results = get_phase1_results(data)
    new_item['phase1_results'] = phase1_results
    new_item['data'] = data
    new_data_phase1.append(new_item)
    print('total consume: {}'.format(LLM.total_consume))
    if i % 100 == 0:
        with open(root_path + 'new_data_phase1.json', 'w') as f:
            json.dump(new_data_phase1, f, indent=4)

## phase1 save
with open(root_path + 'new_data_phase1.json', 'w') as f:
    json.dump(new_data_phase1, f, indent=4)

## phase2data generation
for i, data in enumerate(tqdm(new_data_phase1)):
    new_item = {}
    new_item['phase1_results'] = data['phase1_results']
    phase1_type = new_item['phase1_results']['phase1']['answer']
    new_item['phase2_results'] = get_phase2_results(phase1_type, data['data'])
    new_item['data'] = data['data']
    new_item['input_text'] = get_input_text(data['data'])
    new_item['output_text'] = get_output_text(new_item['phase1_results'], new_item['phase2_results'])
    new_data.append(new_item)
    print('total consume: {}'.format(LLM.total_consume))
    if i % 100 == 0:
        with open(root_path + 'syntrain_reasoning_tuning.json', 'w') as f:
            json.dump(new_data, f, indent=4)

# new_data = exist_data + new_data
## save

with open(root_path + 'syntrain_reasoning_tuning.json', 'w') as f:
    json.dump(new_data, f, indent=4)

