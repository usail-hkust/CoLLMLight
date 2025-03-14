from .pipeline import Pipeline
from .oneline import OneLine
from . import config
import wandb
import copy
import numpy as np
import time
import os
import re
import json
import shutil
from itertools import zip_longest, permutations

location_dict_short = {"North": "N", "South": "S", "East": "E", "West": "W"}
location_direction_dict = ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]

def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)
    return dic_result

def pipeline_wrapper(dic_agent_conf, dic_traffic_env_conf, dic_path, roadnet, trafficflow):
    results_table = []
    all_rewards = []
    all_queue_len = []
    all_travel_time = []
    for i in range(1):
        dic_path["PATH_TO_MODEL"] = (dic_path["PATH_TO_MODEL"].split(".")[0] + ".json" +
                                     time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())))
        dic_path["PATH_TO_WORK_DIRECTORY"] = (dic_path["PATH_TO_WORK_DIRECTORY"].split(".")[0] + ".json" +
                                              time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())))
        ppl = Pipeline(dic_agent_conf=dic_agent_conf,
                       dic_traffic_env_conf=dic_traffic_env_conf,
                       dic_path=dic_path,
                       roadnet=roadnet,
                       trafficflow=trafficflow)
        round_results = ppl.run(round=i, multi_process=False)
        results_table.append([round_results['test_reward_over'], round_results['test_avg_queue_len_over'], round_results['test_avg_travel_time_over']])
        all_rewards.append(round_results['test_reward_over'])
        all_queue_len.append(round_results['test_avg_queue_len_over'])
        all_travel_time.append(round_results['test_avg_travel_time_over'])

        # delete junk
        cmd_delete_model = 'find <dir> -type f ! -name "round_<round>_inter_*.h5" -exec rm -rf {} \;'.replace("<dir>", dic_path["PATH_TO_MODEL"]).replace("<round>", str(int(dic_traffic_env_conf["NUM_ROUNDS"] - 1)))
        cmd_delete_work = 'find <dir> -type f ! -name "state_action.json" -exec rm -rf {} \;'.replace("<dir>", dic_path["PATH_TO_WORK_DIRECTORY"])
        os.system(cmd_delete_model)
        os.system(cmd_delete_work)

    results_table.append([np.average(all_rewards), np.average(all_queue_len), np.average(all_travel_time)])
    results_table.append([np.std(all_rewards), np.std(all_queue_len), np.std(all_travel_time)])

    table_logger = wandb.init(
        project=dic_traffic_env_conf['PROJECT_NAME'],
        group=f"{dic_traffic_env_conf['MODEL']}-{roadnet}-{trafficflow}-{len(dic_traffic_env_conf['PHASE'])}_Phases",
        name="exp_results",
        config=merge(merge(dic_agent_conf, dic_path), dic_traffic_env_conf),
    )
    columns = ["reward", "avg_queue_len", "avg_travel_time"]
    logger_table = wandb.Table(columns=columns, data=results_table)
    table_logger.log({"results": logger_table})
    wandb.finish()

    print("pipeline_wrapper end")
    return

def oneline_wrapper(dic_agent_conf, dic_traffic_env_conf, dic_path, roadnet, trafficflow):
    results_table = []
    all_rewards = []
    all_queue_len = []
    all_travel_time = []
    for i in range(1):
        dic_path["PATH_TO_MODEL"] = (dic_path["PATH_TO_MODEL"].split(".")[0] + ".json" +
                                     time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())))
        dic_path["PATH_TO_WORK_DIRECTORY"] = (dic_path["PATH_TO_WORK_DIRECTORY"].split(".")[0] + ".json" +
                                              time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())))
        oneline = OneLine(dic_agent_conf=dic_agent_conf,
                          dic_traffic_env_conf=merge(config.dic_traffic_env_conf, dic_traffic_env_conf),
                          dic_path=merge(config.DIC_PATH, dic_path),
                          roadnet=roadnet,
                          trafficflow=trafficflow
                          )
        round_results = oneline.train(round=i)
        results_table.append([round_results['test_reward_over'], round_results['test_avg_queue_len_over'],
                              round_results['test_avg_travel_time_over']])
        all_rewards.append(round_results['test_reward_over'])
        all_queue_len.append(round_results['test_avg_queue_len_over'])
        all_travel_time.append(round_results['test_avg_travel_time_over'])

        # delete junk
        cmd_delete_model = 'rm -rf <dir>'.replace("<dir>", dic_path["PATH_TO_MODEL"])
        cmd_delete_work = 'find <dir> -type f ! -name "state_action.json" -exec rm -rf {} \;'.replace("<dir>", dic_path["PATH_TO_WORK_DIRECTORY"])
        os.system(cmd_delete_model)
        os.system(cmd_delete_work)

    results_table.append([np.average(all_rewards), np.average(all_queue_len), np.average(all_travel_time)])
    results_table.append([np.std(all_rewards), np.std(all_queue_len), np.std(all_travel_time)])

    table_logger = wandb.init(
        project=dic_traffic_env_conf['PROJECT_NAME'],
        group=f"{dic_traffic_env_conf['MODEL_NAME']}-{roadnet}-{trafficflow}-{len(dic_agent_conf['FIXED_TIME'])}_Phases",
        name="exp_results",
        config=merge(merge(dic_agent_conf, dic_path), dic_traffic_env_conf),
    )
    columns = ["reward", "avg_queue_len", "avg_travel_time"]
    logger_table = wandb.Table(columns=columns, data=results_table)
    table_logger.log({"results": logger_table})
    wandb.finish()

    return

def extract_answer_by_tag(tag, text):
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None

def extract_json_from_string(text):
    """
    Extracts JSON data from a given string.
    
    Args:
        text (str): The input string containing JSON data.
    
    Returns:
        dict or None: Extracted JSON data as a dictionary, or None if extraction fails.
    """
    try:
        # Use a regular expression to find JSON-like structures in the text
        json_pattern = r'{.*}'
        match = re.search(json_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            json_string = match.group()
            try:
                # Parse the JSON string into a Python dictionary
                json_data = json.loads(json_string)
                return json_data
            except json.JSONDecodeError:
                print("Error: Failed to decode JSON. The JSON format might be incorrect.")
                return {}
        else:
            print("Error: No JSON data found in the provided string.")
            return {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}



def estimate_last_car_exit_time(queue_car, distance, speed=11, start_time=2, pass_time=1, queue_gap=1.5):
    # 第一个排队车辆通过路口的时间
    first_car_exit_time = start_time + pass_time
    
    # 其余排队车辆通过路口的时间
    remaining_cars_exit_time = (queue_car - 1) * (queue_gap + pass_time)
    
    # 所有排队车辆通过路口的总时间
    total_queue_time = first_car_exit_time + remaining_cars_exit_time
    
    # moving_car 到达路口的时间
    moving_car_arrival_time = distance / speed
    
    # 判断 moving_car 到达路口时前面是否还有排队车辆
    if moving_car_arrival_time < total_queue_time:
        # moving_car 需要等待前面的排队车辆通过后才能通过
        last_car_exit_time = total_queue_time + pass_time
    else:
        # moving_car 可以直接通过路口
        last_car_exit_time = moving_car_arrival_time + pass_time
    
    return last_car_exit_time


    

def convert_sets_to_lists(d):
    if isinstance(d, dict):
        return {k: convert_sets_to_lists(v) for k, v in d.items()}
    elif isinstance(d, set):
        return list(d)
    elif isinstance(d, list):
        return [convert_sets_to_lists(i) for i in d]
    else:
        return d

def can_form_target_list(given_list, target_list):
    # 创建给定列表元素的所有可能组合
    # 生成所有可能的排列组合
    possible_combinations = set([''.join(p) for p in permutations(given_list, len(given_list))])
    
    # 检查是否有一个组合在目标列表中
    for combination in possible_combinations:
        if combination in target_list:
            return False
    return True
    
def sum_sublists(lists):
    # 使用 zip_longest 将子列表的对应元素打包在一起，缺失值使用 0 填充，然后按位相加
    return [sum(elements) for elements in zip_longest(*lists, fillvalue=0)]

def redistribute_vehicles(vehicle_counts, full_num):
    n = len(vehicle_counts)
    for i in range(n):
        if vehicle_counts[i] > full_num:
            overflow = vehicle_counts[i] - full_num
            vehicle_counts[i] = full_num
            if i + 1 < n:
                vehicle_counts[i + 1] += overflow

    return vehicle_counts

def merge_dicts_with_max_values(release_range):
    merged_dict = {}
    
    for lane, lane_data in release_range.items():
        if isinstance(lane_data, dict):
            for key, value in lane_data.items():
                if key in merged_dict:
                    merged_dict[key] = max(merged_dict[key], value)
                else:
                    merged_dict[key] = value
    return merged_dict


def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)
    return dic_result

def check_int64_in_dict(data):
    int64_keys = []
    for key, value in data.items():
        if isinstance(value, np.int64):
            int64_keys.append(key)
        elif isinstance(value, dict):
            # Recursively check nested dictionaries
            nested_int64_keys = check_int64_in_dict(value)
            int64_keys.extend([f"{key}.{nested_key}" for nested_key in nested_int64_keys])
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                if isinstance(item, np.int64):
                    int64_keys.append(f"{key}[{idx}]")
                elif isinstance(item, dict):
                    nested_int64_keys = check_int64_in_dict(item)
                    int64_keys.extend([f"{key}[{idx}].{nested_key}" for nested_key in nested_int64_keys])
    return int64_keys

def path_check(dic_path):
    if os.path.exists(dic_path["PATH_TO_WORK_DIRECTORY"]):
        if dic_path["PATH_TO_WORK_DIRECTORY"] != "records/default":
            raise FileExistsError
        else:
            pass
    else:
        os.makedirs(dic_path["PATH_TO_WORK_DIRECTORY"])
    if os.path.exists(dic_path["PATH_TO_MODEL"]):
        if dic_path["PATH_TO_MODEL"] != "model/default":
            raise FileExistsError
        else:
            pass
    else:
        os.makedirs(dic_path["PATH_TO_MODEL"])


def copy_conf_file(dic_path, dic_agent_conf, dic_traffic_env_conf, path=None):
    if path is None:
        path = dic_path["PATH_TO_WORK_DIRECTORY"]
    json.dump(dic_agent_conf, open(os.path.join(path, "agent.conf"), "w"), indent=4)
    json.dump(dic_traffic_env_conf, open(os.path.join(path, "traffic_env.conf"), "w"), indent=4)


def copy_cityflow_file(dic_path, dic_traffic_env_conf, path=None):
    if path is None:
        path = dic_path["PATH_TO_WORK_DIRECTORY"]
    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf["TRAFFIC_FILE"]),
                os.path.join(path, dic_traffic_env_conf["TRAFFIC_FILE"]))
    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf["ROADNET_FILE"]),
                os.path.join(path, dic_traffic_env_conf["ROADNET_FILE"]))

def find_last_non_zero_index(lst):
    for i in range(len(lst) - 1, -1, -1):
        if lst[i] != 0:
            return i
    return 0


