import numpy as np
import json
import copy
import torch
from copy import deepcopy

location_dict_short = {"North": "N", "South": "S", "East": "E", "West": "W"}
location_direction_dict = ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]
location_all_direction_dict = ["NT", "NL", "NR", "ST", "SL", "SR", "ET", "EL", "ER", "WT", "WL", "WR"]
location_incoming_dict = ["N", "S", "E", "W"]
incoming2output = {"North": "S", "South": "N", "East": "W", "West": "E"}
eight_phase_list = ['ETWT', 'NTST', 'ELWL', 'NLSL', 'WTWL', 'ETEL', 'STSL', 'NTNL']
four_phase_list = {'ETWT': 0, 'NTST': 1, 'ELWL': 2, 'NLSL': 3}
phase2code = {0: 'ETWT', 1: 'NTST', 2: 'ELWL', 3: 'NLSL'}
location_dict = {"N": "North", "S": "South", "E": "East", "W": "West"}
location_dict_detail = {"N": "Northern", "S": "Southern", "E": "Eastern", "W": "Western"}
location2releaselane = {'N': ['ST', 'WL'], 'S': ['NT', 'EL'], 'E': ['WT','NL'], 'W': ['ET', 'SL']}

phase_explanation_dict_detail = {"NTST": "- NTST: Northern and southern through lanes.",
                                 "NLSL": "- NLSL: Northern and southern left-turn lanes.",
                                 "NTNL": "- NTNL: Northern through and left-turn lanes.",
                                 "STSL": "- STSL: Southern through and left-turn lanes.",
                                 "ETWT": "- ETWT: Eastern and western through lanes.",
                                 "ELWL": "- ELWL: Eastern and western left-turn lanes.",
                                 "ETEL": "- ETEL: Eastern through and left-turn lanes.",
                                 "WTWL": "- WTWL: Western through and left-turn lanes."
                                }

def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)
    return dic_result

def load_json(file):
    try:
        with open(file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise e
    return data

def dump_json(data, file, indent=None):
    try:
        with open(file, 'w') as f:
            json.dump(data, f, indent=indent)
    except Exception as e:
        raise e

def calculate_road_length(road_points):
    length = 0.0
    i = 1
    while i < len(road_points):
        length += np.sqrt((road_points[i]['x'] - road_points[i-1]['x']) ** 2 + (road_points[i]['y'] - road_points[i-1]['y']) ** 2)
        i += 1

    return length

def get_state(roads, env):
    """
    Retrieve the state of the intersection from sumo, in the form of cell occupancy
    """
    lane_queues = env.eng.get_lane_waiting_vehicle_count()
    lane_vehicles = env.eng.get_lane_vehicles()

    # init statistic info & get queue info
    statistic_state = {}
    statistic_state_incoming = {}
    for r in roads:
        # road info
        location = roads[r]["location"]
        road_length = float(roads[r]["length"])

        # get queue info
        if roads[r]["type"] == "outgoing":
            if roads[r]["go_straight"] is not None:
                queue_len = 0.0
                for lane in roads[r]["lanes"]["go_straight"]:
                    queue_len += lane_queues[f"{r}_{lane}"]
                statistic_state[f"{location_dict_short[roads[r]['location']]}T"] = {"cells": [0 for _ in range(3)],
                                                                                    "queue_len": queue_len}

            if roads[r]["turn_left"] is not None:
                queue_len = 0.0
                for lane in roads[r]["lanes"]["turn_left"]:
                    queue_len += lane_queues[f"{r}_{lane}"]
                statistic_state[f"{location_dict_short[roads[r]['location']]}L"] = {"cells": [0 for _ in range(3)],
                                                                                    "queue_len": queue_len}

            # get vehicle position info
            straight_lanes = [f"{r}_{idx}" for idx in roads[r]["lanes"]["go_straight"]]
            left_lanes = [f"{r}_{idx}" for idx in roads[r]["lanes"]["turn_left"]]
            lanes = straight_lanes + left_lanes

            for lane in lanes:
                vehicles = lane_vehicles[lane]

                # collect lane group info
                if location == "North":
                    if lane in straight_lanes:
                        lane_group = 0
                    elif lane in left_lanes:
                        lane_group = 1
                elif location == "South":
                    if lane in straight_lanes:
                        lane_group = 2
                    elif lane in left_lanes:
                        lane_group = 3
                elif location == "East":
                    if lane in straight_lanes:
                        lane_group = 4
                    elif lane in left_lanes:
                        lane_group = 5
                elif location == "West":
                    if lane in straight_lanes:
                        lane_group = 6
                    elif lane in left_lanes:
                        lane_group = 7
                else:
                    lane_group = -1

                # collect lane cell info
                for veh in vehicles:
                    veh_info = env.eng.get_vehicle_info(veh)
                    lane_pos = road_length - float(veh_info["distance"])

                    # update statistic state
                    if lane_pos <= road_length / 10:
                        gpt_lane_cell = 0
                    elif road_length / 10 < lane_pos <= road_length / 3:
                        gpt_lane_cell = 1
                    else:
                        gpt_lane_cell = 2

                    # speed > 0.1 m/s are approaching vehicles
                    if float(veh_info["speed"]) > 0.1:
                        statistic_state[location_direction_dict[lane_group]]["cells"][gpt_lane_cell] += 1

        # incoming lanes
        else:
            queue_len = 0.0
            for lane in range(2):
                queue_len += lane_queues[f"{r}_{lane}"]
            statistic_state_incoming[location_dict_short[roads[r]['location']]] = {"cells": [0 for _ in range(3)], "queue_len": queue_len}
            incoming_lanes = [f"{r}_{idx}" for idx in range(2)]

            for lane in incoming_lanes:
                vehicles = lane_vehicles[lane]

                # collect lane group info
                if location == "North":
                    lane_group = 0
                elif location == "South":
                    lane_group = 1
                elif location == "East":
                    lane_group = 2
                elif location == "West":
                    lane_group = 3
                else:
                    lane_group = -1

                # collect lane cell info
                for veh in vehicles:
                    veh_info = env.eng.get_vehicle_info(veh)
                    lane_pos = road_length - float(veh_info["distance"])

                    # update statistic state
                    if lane_pos <= road_length / 10:
                        gpt_lane_cell = 0
                    elif road_length / 10 < lane_pos <= road_length / 3:
                        gpt_lane_cell = 1
                    else:
                        gpt_lane_cell = 2

                    # speed > 0.1 m/s are approaching vehicles
                    if float(veh_info["speed"]) > 0.1:
                        statistic_state_incoming[location_incoming_dict[lane_group]]["cells"][gpt_lane_cell] += 1

    return statistic_state, statistic_state_incoming

def get_state_detail_all_lane(roads, env):
    """
    Retrieve the state of the intersection from sumo, in the form of cell occupancy
    divide the lane into many seg
    tag1
    """
    lane_queues = env.eng.get_lane_waiting_vehicle_count()
    lane_vehicles = env.eng.get_lane_vehicles()

    # init statistic info & get queue info
    statistic_state = {}
    statistic_state_incoming = {}
    outgoing_lane_speeds = []
    for r in roads:
        # road info
        location = roads[r]["location"]
        road_length = float(roads[r]["length"])

        # get queue info
        if roads[r]["type"] == "outgoing":
            if roads[r]["go_straight"] is not None:
                queue_len = 0.0
                for lane in roads[r]["lanes"]["go_straight"]:
                    queue_len += lane_queues[f"{r}_{lane}"]
                statistic_state[f"{location_dict_short[roads[r]['location']]}T"] = {"cells": [0 for _ in range(4)],
                                                                                    "queue_len": queue_len,
                                                                                    "avg_wait_time": 0.0}

            if roads[r]["turn_left"] is not None:
                queue_len = 0.0
                for lane in roads[r]["lanes"]["turn_left"]:
                    queue_len += lane_queues[f"{r}_{lane}"]
                statistic_state[f"{location_dict_short[roads[r]['location']]}L"] = {"cells": [0 for _ in range(4)],
                                                                                    "queue_len": queue_len,
                                                                                    "avg_wait_time": 0.0}
            if roads[r]["turn_right"] is not None:
                queue_len = 0.0
                # for lane in roads[r]["lanes"]["turn_right"]:
                #     queue_len += lane_queues[f"{r}_{lane}"]
                statistic_state[f"{location_dict_short[roads[r]['location']]}R"] = {"cells": [0 for _ in range(4)],
                                                                                    "queue_len": queue_len,
                                                                                    "avg_wait_time": 0.0}

            # get vehicle position info
            straight_lanes = [f"{r}_{idx}" for idx in roads[r]["lanes"]["go_straight"]]
            left_lanes = [f"{r}_{idx}" for idx in roads[r]["lanes"]["turn_left"]]
            right_lanes = [f"{r}_{idx}" for idx in roads[r]["lanes"]["turn_right"]]
            lanes = straight_lanes + left_lanes + right_lanes

            for lane in lanes:
                waiting_times = []
                vehicles = lane_vehicles[lane]

                # collect lane group info
                if location == "North":
                    if lane in straight_lanes:
                        lane_group = 0
                    elif lane in left_lanes:
                        lane_group = 1
                    elif lane in right_lanes:
                        lane_group = 2
                elif location == "South":
                    if lane in straight_lanes:
                        lane_group = 3
                    elif lane in left_lanes:
                        lane_group = 4
                    elif lane in right_lanes:
                        lane_group = 5
                elif location == "East":
                    if lane in straight_lanes:
                        lane_group = 6
                    elif lane in left_lanes:
                        lane_group = 7
                    elif lane in right_lanes:
                        lane_group = 8
                elif location == "West":
                    if lane in straight_lanes:
                        lane_group = 9
                    elif lane in left_lanes:
                        lane_group = 10
                    elif lane in right_lanes:
                        lane_group = 11
                else:
                    lane_group = -1

                # collect lane cell info
                for veh in vehicles:
                    veh_info = env.eng.get_vehicle_info(veh)
                    lane_pos = road_length - float(veh_info["distance"])

                    # update statistic state
                    if lane_pos <= road_length / 10:
                        gpt_lane_cell = 0
                    elif road_length / 10 < lane_pos <= road_length / 3:
                        gpt_lane_cell = 1
                    elif road_length / 3 < lane_pos <= (road_length / 3) * 2:
                        gpt_lane_cell = 2
                    else:
                        gpt_lane_cell = 3

                    # speed > 0.1 m/s are approaching vehicles
                    speed = float(veh_info["speed"])
                    if speed > 0.1:
                        statistic_state[location_all_direction_dict[lane_group]]["cells"][gpt_lane_cell] += 1
                        
                        outgoing_lane_speeds.append(speed)
                    else:
                        veh_waiting_time = env.waiting_vehicle_list[veh]['time'] if veh in env.waiting_vehicle_list else 0.0
                        waiting_times.append(veh_waiting_time)
                avg_wait_time = np.mean(waiting_times) if len(waiting_times) > 0 else 0.0
                statistic_state[location_all_direction_dict[lane_group]]["avg_wait_time"] = avg_wait_time


        # incoming lanes
        else:
            queue_len = 0.0
            for lane in range(2):
                queue_len += lane_queues[f"{r}_{lane}"]
            statistic_state_incoming[location_dict_short[roads[r]['location']]] = {"cells": [0 for _ in range(4)],
                                                                                   "queue_len": queue_len}
            incoming_lanes = [f"{r}_{idx}" for idx in range(2)]

            for lane in incoming_lanes:
                vehicles = lane_vehicles[lane]

                # collect lane group info
                if location == "North":
                    lane_group = 0
                elif location == "South":
                    lane_group = 1
                elif location == "East":
                    lane_group = 2
                elif location == "West":
                    lane_group = 3
                else:
                    lane_group = -1

                # collect lane cell info
                for veh in vehicles:
                    veh_info = env.eng.get_vehicle_info(veh)
                    lane_pos = road_length - float(veh_info["distance"])

                    # update statistic state
                    if lane_pos <= road_length / 10:
                        gpt_lane_cell = 0
                    elif road_length / 10 < lane_pos <= road_length / 3:
                        gpt_lane_cell = 1
                    elif road_length / 3 < lane_pos <= (road_length / 3) * 2:
                        gpt_lane_cell = 2
                    else:
                        gpt_lane_cell = 3

                    # speed > 0.1 m/s are approaching vehicles
                    if float(veh_info["speed"]) > 0.1:
                        statistic_state_incoming[location_incoming_dict[lane_group]]["cells"][gpt_lane_cell] += 1

    mean_speed = np.mean(outgoing_lane_speeds) if len(outgoing_lane_speeds) > 0 else 0.0

    return statistic_state, statistic_state_incoming, mean_speed

def fix_decreasing_list(a_dict):
    def fix_decreasing_list(lst):
        n = len(lst)
        
        for i in range(n - 2, -1, -1):
            
            if lst[i] < lst[i + 1]:
                lst[i] = lst[i + 1]
        return lst
    
    sorted_items = sorted(a_dict.items())
    val_list = [item[1] for item in sorted_items]
    key_list = [item[0] for item in sorted_items]
    
    val_list = fix_decreasing_list(val_list)
    b_dict = {}
    for i in range(len(key_list)):
        b_dict[key_list[i]] = val_list[i]
  
    return b_dict

def get_state_detail_many_seg_all_lane(roads, env, seg_num = 10):
    """
    Retrieve the state of the intersection from sumo, in the form of cell occupancy
    divide the lane into many seg
    tag1
    """
    lane_queues = env.eng.get_lane_waiting_vehicle_count()
    lane_vehicles = env.eng.get_lane_vehicles()

    # init statistic info & get queue info
    statistic_state = {}
    statistic_state_incoming = {}
    outgoing_lane_speeds = []
    for r in roads:
        # road info
        location = roads[r]["location"]
        road_length = float(roads[r]["length"])

        # get queue info
        if roads[r]["type"] == "outgoing":  ## lane that target to this inter
            if roads[r]["go_straight"] is not None:
                queue_len = 0.0
                for lane in roads[r]["lanes"]["go_straight"]:
                    queue_len += lane_queues[f"{r}_{lane}"]
                statistic_state[f"{location_dict_short[roads[r]['location']]}T"] = {"cells": [0 for _ in range(seg_num)],
                                                                                    "ql_cells": [0 for _ in range(seg_num)],
                                                                                    "queue_len": queue_len,
                                                                                    "out_of_lane": 0, 
                                                                                    "veh2cell":{},
                                                                                    "avg_wait_time": 0.0,
                                                                                    "wait_time": {},
                                                                                    "veh2pos": {},
                                                                                    "road_length": road_length}

            if roads[r]["turn_left"] is not None:
                queue_len = 0.0
                for lane in roads[r]["lanes"]["turn_left"]:
                    queue_len += lane_queues[f"{r}_{lane}"]
                statistic_state[f"{location_dict_short[roads[r]['location']]}L"] = {"cells": [0 for _ in range(seg_num)],
                                                                                    "ql_cells": [0 for _ in range(seg_num)],
                                                                                    "queue_len": queue_len,
                                                                                    "out_of_lane": 0, 
                                                                                    "veh2cell":{},
                                                                                    "avg_wait_time": 0.0,
                                                                                    "wait_time": {},
                                                                                    "veh2pos": {},
                                                                                    "road_length": road_length}
            if roads[r]["turn_right"] is not None:
                queue_len = 0.0
                for lane in roads[r]["lanes"]["turn_right"]:
                    queue_len += lane_queues[f"{r}_{lane}"]
                statistic_state[f"{location_dict_short[roads[r]['location']]}R"] = {"cells": [0 for _ in range(seg_num)],
                                                                                    "ql_cells": [0 for _ in range(seg_num)],
                                                                                    "queue_len": queue_len,
                                                                                    "out_of_lane": 0, 
                                                                                    "veh2cell":{},
                                                                                    "avg_wait_time": 0.0,
                                                                                    "wait_time": {},
                                                                                    "veh2pos": {},
                                                                                    "road_length": road_length}

            # get vehicle position info
            straight_lanes = [f"{r}_{idx}" for idx in roads[r]["lanes"]["go_straight"]]
            left_lanes = [f"{r}_{idx}" for idx in roads[r]["lanes"]["turn_left"]]
            right_lanes = [f"{r}_{idx}" for idx in roads[r]["lanes"]["turn_right"]]
            lanes = straight_lanes + left_lanes + right_lanes

            for lane in lanes:
                waiting_times = []
                vehicles = lane_vehicles[lane]

                # collect lane group info
                if location == "North":
                    if lane in straight_lanes:
                        lane_group = 0
                    elif lane in left_lanes:
                        lane_group = 1
                    elif lane in right_lanes:
                        lane_group = 2
                elif location == "South":
                    if lane in straight_lanes:
                        lane_group = 3
                    elif lane in left_lanes:
                        lane_group = 4
                    elif lane in right_lanes:
                        lane_group = 5
                elif location == "East":
                    if lane in straight_lanes:
                        lane_group = 6
                    elif lane in left_lanes:
                        lane_group = 7
                    elif lane in right_lanes:
                        lane_group = 8
                elif location == "West":
                    if lane in straight_lanes:
                        lane_group = 9
                    elif lane in left_lanes:
                        lane_group = 10
                    elif lane in right_lanes:
                        lane_group = 11
                else:
                    lane_group = -1

                # collect lane cell info
                for veh in vehicles:
                    veh_info = env.eng.get_vehicle_info(veh)
                    lane_pos = road_length - float(veh_info["distance"])
                    statistic_state[location_all_direction_dict[lane_group]]["veh2pos"][veh] = lane_pos
                    # update statistic state
                    seg_length = road_length/seg_num
                    gpt_lane_cell = int(lane_pos//seg_length)
                    statistic_state[location_all_direction_dict[lane_group]]["veh2cell"][veh] = gpt_lane_cell
                    veh_waiting_time = env.waiting_vehicle_list[veh]['time'] if veh in env.waiting_vehicle_list else 0.0
                    statistic_state[location_all_direction_dict[lane_group]]["wait_time"][veh] = veh_waiting_time
                    if veh in env.waiting_vehicle_list:
                        waiting_times.append(veh_waiting_time)
                    if gpt_lane_cell >= seg_num:
                        statistic_state[location_all_direction_dict[lane_group]]["out_of_lane"] += 1
                    else:
                        # speed > 0.1 m/s are approaching vehicles
                        speed = float(veh_info["speed"])
                        if speed > 0.1:
                            statistic_state[location_all_direction_dict[lane_group]]["cells"][gpt_lane_cell] += 1                        
                            outgoing_lane_speeds.append(speed)
                        else:
                            statistic_state[location_all_direction_dict[lane_group]]["ql_cells"][gpt_lane_cell] += 1
                            
                            
                avg_wait_time = np.mean(waiting_times) if len(waiting_times) > 0 else 0.0
                statistic_state[location_all_direction_dict[lane_group]]["avg_wait_time"] = avg_wait_time


        # incoming lanes
        else: ## lane that output from this inter
            queue_len = 0.0
            for lane in range(3):
                queue_len += lane_queues[f"{r}_{lane}"]
            # change the incoming direction to output direction

            statistic_state_incoming[incoming2output[roads[r]['location']]] = {"cells": [0 for _ in range(seg_num)],
                                                                                   "ql_cells": [0 for _ in range(seg_num)],
                                                                                   "out_of_lane": 0,
                                                                                   "veh2cell":{},
                                                                                   "queue_len": queue_len}
            incoming_lanes = [f"{r}_{idx}" for idx in range(3)]

            for lane in incoming_lanes:
                vehicles = lane_vehicles[lane]

                # collect lane group info
                if location == "North": # this location is incoming direction, we need to change it to output direction
                    lane_group = 1
                elif location == "South":
                    lane_group = 0
                elif location == "East":
                    lane_group = 3
                elif location == "West":
                    lane_group = 2
                else:
                    lane_group = -1

                # collect lane cell info
                for veh in vehicles:
                    veh_info = env.eng.get_vehicle_info(veh)
                    lane_pos = road_length - float(veh_info["distance"])

                    # update statistic state
                    seg_length = road_length/seg_num
                    gpt_lane_cell = int(lane_pos//seg_length)
                    statistic_state_incoming[location_incoming_dict[lane_group]]["veh2cell"][veh] = gpt_lane_cell
                    if gpt_lane_cell >= seg_num:
                        statistic_state_incoming[location_incoming_dict[lane_group]]["out_of_lane"] += 1
                        
                    else:
                        # speed > 0.1 m/s are approaching vehicles
                        if float(veh_info["speed"]) > 0.1:
                            statistic_state_incoming[location_incoming_dict[lane_group]]["cells"][gpt_lane_cell] += 1
                        else:
                            statistic_state_incoming[location_incoming_dict[lane_group]]["ql_cells"][gpt_lane_cell] += 1

    mean_speed = np.mean(outgoing_lane_speeds) if len(outgoing_lane_speeds) > 0 else 0.0

    return statistic_state, statistic_state_incoming, mean_speed

def get_state_detail(roads, env):
    """
    Retrieve the state of the intersection from sumo, in the form of cell occupancy
    """
    lane_queues = env.eng.get_lane_waiting_vehicle_count()
    lane_vehicles = env.eng.get_lane_vehicles()

    # init statistic info & get queue info
    statistic_state = {}
    statistic_state_incoming = {}
    outgoing_lane_speeds = []
    for r in roads:
        # road info
        location = roads[r]["location"]
        road_length = float(roads[r]["length"])

        # get queue info
        if roads[r]["type"] == "outgoing":
            if roads[r]["go_straight"] is not None:
                queue_len = 0.0
                for lane in roads[r]["lanes"]["go_straight"]:
                    queue_len += lane_queues[f"{r}_{lane}"]
                statistic_state[f"{location_dict_short[roads[r]['location']]}T"] = {"cells": [0 for _ in range(4)],
                                                                                    "queue_len": queue_len,
                                                                                    "avg_wait_time": 0.0}

            if roads[r]["turn_left"] is not None:
                queue_len = 0.0
                for lane in roads[r]["lanes"]["turn_left"]:
                    queue_len += lane_queues[f"{r}_{lane}"]
                statistic_state[f"{location_dict_short[roads[r]['location']]}L"] = {"cells": [0 for _ in range(4)],
                                                                                    "queue_len": queue_len,
                                                                                    "avg_wait_time": 0.0}
                

            # get vehicle position info
            straight_lanes = [f"{r}_{idx}" for idx in roads[r]["lanes"]["go_straight"]]
            left_lanes = [f"{r}_{idx}" for idx in roads[r]["lanes"]["turn_left"]]
            lanes = straight_lanes + left_lanes

            for lane in lanes:
                waiting_times = []
                vehicles = lane_vehicles[lane]

                # collect lane group info
                if location == "North":
                    if lane in straight_lanes:
                        lane_group = 0
                    elif lane in left_lanes:
                        lane_group = 1
                elif location == "South":
                    if lane in straight_lanes:
                        lane_group = 2
                    elif lane in left_lanes:
                        lane_group = 3
                elif location == "East":
                    if lane in straight_lanes:
                        lane_group = 4
                    elif lane in left_lanes:
                        lane_group = 5
                elif location == "West":
                    if lane in straight_lanes:
                        lane_group = 6
                    elif lane in left_lanes:
                        lane_group = 7
                else:
                    lane_group = -1

                # collect lane cell info
                for veh in vehicles:
                    veh_info = env.eng.get_vehicle_info(veh)
                    lane_pos = road_length - float(veh_info["distance"])

                    # update statistic state
                    if lane_pos <= road_length / 10:
                        gpt_lane_cell = 0
                    elif road_length / 10 < lane_pos <= road_length / 3:
                        gpt_lane_cell = 1
                    elif road_length / 3 < lane_pos <= (road_length / 3) * 2:
                        gpt_lane_cell = 2
                    else:
                        gpt_lane_cell = 3

                    # speed > 0.1 m/s are approaching vehicles
                    speed = float(veh_info["speed"])
                    if speed > 0.1:
                        statistic_state[location_direction_dict[lane_group]]["cells"][gpt_lane_cell] += 1
                        outgoing_lane_speeds.append(speed)
                    else:
                        veh_waiting_time = env.waiting_vehicle_list[veh]['time'] if veh in env.waiting_vehicle_list else 0.0
                        waiting_times.append(veh_waiting_time)
                avg_wait_time = np.mean(waiting_times) if len(waiting_times) > 0 else 0.0
                statistic_state[location_direction_dict[lane_group]]["avg_wait_time"] = avg_wait_time


        # incoming lanes
        else:
            queue_len = 0.0
            for lane in range(2):
                queue_len += lane_queues[f"{r}_{lane}"]
            statistic_state_incoming[location_dict_short[roads[r]['location']]] = {"cells": [0 for _ in range(4)],
                                                                                   "queue_len": queue_len}
            incoming_lanes = [f"{r}_{idx}" for idx in range(2)]

            for lane in incoming_lanes:
                vehicles = lane_vehicles[lane]

                # collect lane group info
                if location == "North":
                    lane_group = 0
                elif location == "South":
                    lane_group = 1
                elif location == "East":
                    lane_group = 2
                elif location == "West":
                    lane_group = 3
                else:
                    lane_group = -1

                # collect lane cell info
                for veh in vehicles:
                    veh_info = env.eng.get_vehicle_info(veh)
                    lane_pos = road_length - float(veh_info["distance"])

                    # update statistic state
                    if lane_pos <= road_length / 10:
                        gpt_lane_cell = 0
                    elif road_length / 10 < lane_pos <= road_length / 3:
                        gpt_lane_cell = 1
                    elif road_length / 3 < lane_pos <= (road_length / 3) * 2:
                        gpt_lane_cell = 2
                    else:
                        gpt_lane_cell = 3

                    # speed > 0.1 m/s are approaching vehicles
                    if float(veh_info["speed"]) > 0.1:
                        statistic_state_incoming[location_incoming_dict[lane_group]]["cells"][gpt_lane_cell] += 1

    mean_speed = np.mean(outgoing_lane_speeds) if len(outgoing_lane_speeds) > 0 else 0.0

    return statistic_state, statistic_state_incoming, mean_speed

def get_state_three_segment(roads, env):
    """
    Retrieve the state of the intersection from sumo, in the form of cell occupancy
    """
    lane_queues = env.eng.get_lane_waiting_vehicle_count()
    lane_vehicles = env.eng.get_lane_vehicles()

    # init statistic info & get queue info
    statistic_state = {}
    statistic_state_incoming = {}
    outgoing_lane_speeds = []
    for r in roads:
        # road info
        location = roads[r]["location"]
        road_length = float(roads[r]["length"])

        # get queue info
        if roads[r]["type"] == "outgoing":
            if roads[r]["go_straight"] is not None:
                queue_len = 0.0
                for lane in roads[r]["lanes"]["go_straight"]:
                    queue_len += lane_queues[f"{r}_{lane}"]
                statistic_state[f"{location_dict_short[roads[r]['location']]}T"] = {"cells": [0 for _ in range(3)],
                                                                                    "queue_len": queue_len,
                                                                                    "avg_wait_time": 0.0}

            if roads[r]["turn_left"] is not None:
                queue_len = 0.0
                for lane in roads[r]["lanes"]["turn_left"]:
                    queue_len += lane_queues[f"{r}_{lane}"]
                statistic_state[f"{location_dict_short[roads[r]['location']]}L"] = {"cells": [0 for _ in range(3)],
                                                                                    "queue_len": queue_len,
                                                                                    "avg_wait_time": 0.0}

            # get vehicle position info
            straight_lanes = [f"{r}_{idx}" for idx in roads[r]["lanes"]["go_straight"]]
            left_lanes = [f"{r}_{idx}" for idx in roads[r]["lanes"]["turn_left"]]
            lanes = straight_lanes + left_lanes

            for lane in lanes:
                waiting_times = []
                vehicles = lane_vehicles[lane]

                # collect lane group info
                if location == "North":
                    if lane in straight_lanes:
                        lane_group = 0
                    elif lane in left_lanes:
                        lane_group = 1
                elif location == "South":
                    if lane in straight_lanes:
                        lane_group = 2
                    elif lane in left_lanes:
                        lane_group = 3
                elif location == "East":
                    if lane in straight_lanes:
                        lane_group = 4
                    elif lane in left_lanes:
                        lane_group = 5
                elif location == "West":
                    if lane in straight_lanes:
                        lane_group = 6
                    elif lane in left_lanes:
                        lane_group = 7
                else:
                    lane_group = -1

                # collect lane cell info
                for veh in vehicles:
                    veh_info = env.eng.get_vehicle_info(veh)
                    lane_pos = road_length - float(veh_info["distance"])

                    # update statistic state
                    if lane_pos <= road_length / 10:
                        gpt_lane_cell = 0
                    elif road_length / 10 < lane_pos <= road_length / 3:
                        gpt_lane_cell = 1
                    else:
                        gpt_lane_cell = 2

                    # speed > 0.1 m/s are approaching vehicles
                    speed = float(veh_info["speed"])
                    if speed > 0.1:
                        statistic_state[location_direction_dict[lane_group]]["cells"][gpt_lane_cell] += 1
                        outgoing_lane_speeds.append(speed)
                    else:
                        veh_waiting_time = env.waiting_vehicle_list[veh]['time'] if veh in env.waiting_vehicle_list else 0.0
                        waiting_times.append(veh_waiting_time)
                avg_wait_time = np.mean(waiting_times) if len(waiting_times) > 0 else 0.0
                statistic_state[location_direction_dict[lane_group]]["avg_wait_time"] = avg_wait_time


        # incoming lanes
        else:
            queue_len = 0.0
            for lane in range(2):
                queue_len += lane_queues[f"{r}_{lane}"]
            statistic_state_incoming[location_dict_short[roads[r]['location']]] = {"cells": [0 for _ in range(3)],
                                                                                   "queue_len": queue_len}
            incoming_lanes = [f"{r}_{idx}" for idx in range(2)]

            for lane in incoming_lanes:
                vehicles = lane_vehicles[lane]

                # collect lane group info
                if location == "North":
                    lane_group = 0
                elif location == "South":
                    lane_group = 1
                elif location == "East":
                    lane_group = 2
                elif location == "West":
                    lane_group = 3
                else:
                    lane_group = -1

                # collect lane cell info
                for veh in vehicles:
                    veh_info = env.eng.get_vehicle_info(veh)
                    lane_pos = road_length - float(veh_info["distance"])

                    # update statistic state
                    if lane_pos <= road_length / 10:
                        gpt_lane_cell = 0
                    elif road_length / 10 < lane_pos <= road_length / 3:
                        gpt_lane_cell = 1
                    else:
                        gpt_lane_cell = 2

                    # speed > 0.1 m/s are approaching vehicles
                    if float(veh_info["speed"]) > 0.1:
                        statistic_state_incoming[location_incoming_dict[lane_group]]["cells"][gpt_lane_cell] += 1

    mean_speed = np.mean(outgoing_lane_speeds) if len(outgoing_lane_speeds) > 0 else 0.0

    return statistic_state, statistic_state_incoming, mean_speed

def trans_prompt_llama(message, chat_history, system_prompt):
    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    # The first user input is _not_ stripped
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
    message = message.strip() if do_strip else message
    texts.append(f'{message} [/INST]')
    return ''.join(texts)


def state2text(state):
    state_txt = ""
    for p in four_phase_list:
        lane_1 = p[:2]
        lane_2 = p[2:]
        queue_len_1 = int(state[lane_1]['queue_len'])
        queue_len_2 = int(state[lane_2]['queue_len'])

        seg_1_lane_1 = state[lane_1]['cells'][0]
        seg_2_lane_1 = state[lane_1]['cells'][1]
        seg_3_lane_1 = state[lane_1]['cells'][2] + state[lane_1]['cells'][3]

        seg_1_lane_2 = state[lane_2]['cells'][0]
        seg_2_lane_2 = state[lane_2]['cells'][1]
        seg_3_lane_2 = state[lane_2]['cells'][2] + state[lane_2]['cells'][3]

        state_txt += (f"Signal: {p}\n"
                      f"Relieves: {phase_explanation_dict_detail[p][8:-1]}\n"
                      f"- Early queued: {queue_len_1} ({location_dict[lane_1[0]]}), {queue_len_2} ({location_dict[lane_2[0]]}), {queue_len_1 + queue_len_2} (Total)\n"
                      f"- Segment 1: {seg_1_lane_1} ({location_dict[lane_1[0]]}), {seg_1_lane_2} ({location_dict[lane_2[0]]}), {seg_1_lane_1 + seg_1_lane_2} (Total)\n"
                      f"- Segment 2: {seg_2_lane_1} ({location_dict[lane_1[0]]}), {seg_2_lane_2} ({location_dict[lane_2[0]]}), {seg_2_lane_1 + seg_2_lane_2} (Total)\n"
                      f"- Segment 3: {seg_3_lane_1} ({location_dict[lane_1[0]]}), {seg_3_lane_2} ({location_dict[lane_2[0]]}), {seg_3_lane_1 + seg_3_lane_2} (Total)\n\n")

    return state_txt

def state2feature(state):
    # tag1
    state_txt = ""
    feature = {}
    t = 30
    for p in four_phase_list:
        feature[p] = {}
        
        lane_1 = p[:2]
        lane_2 = p[2:]
        queue_len_1 = int(state[lane_1]['queue_len'])
        avg_wait_time_1 = float(state[lane_1]['avg_wait_time'])
        queue_len_2 = int(state[lane_2]['queue_len'])
        avg_wait_time_2 = float(state[lane_2]['avg_wait_time'])

        seg_1_lane_1 = state[lane_1]['cells'][0]
        seg_2_lane_1 = state[lane_1]['cells'][1]
        seg_3_lane_1 = state[lane_1]['cells'][2] + state[lane_1]['cells'][3]

        seg_1_lane_2 = state[lane_2]['cells'][0]
        seg_2_lane_2 = state[lane_2]['cells'][1]
        seg_3_lane_2 = state[lane_2]['cells'][2] + state[lane_2]['cells'][3]

        feature[p]["Early queued"] = queue_len_1 + queue_len_2
        feature[p]["Segment 1"] = seg_1_lane_1 + seg_1_lane_2
        feature[p]["Segment 2"] = seg_2_lane_1 + seg_2_lane_2
        feature[p]["Segment 3"] = seg_3_lane_1 + seg_3_lane_2
        feature[p]["Waitime"] = queue_len_1 * (avg_wait_time_1+t) + queue_len_2 * (avg_wait_time_2+t)
        feature[p]["Waitime1"] = feature[p]["Waitime"] + feature[p]["Segment 1"] * t
        feature[p]["Waitime2"] = feature[p]["Waitime1"] + feature[p]["Segment 2"] * t
        feature[p]["Waitime3"] = feature[p]["Waitime2"] + feature[p]["Segment 3"] * t
        
        
    return feature

def getNCPrompt(i, current_states, env):
    # fill information
    # signals_text = ""
    # for i, p in enumerate(four_phase_list):
    #     signals_text += phase_explanation_dict_detail[p] + "\n"
    
    system_txt = "You are an expert in traffic management and are responsible for controlling the traffic signals at an intersection. You can apply your knowledge of traffic common sense to solve this task. "
    traffic_txt = "A traffic light regulates a four-section intersection with northern, southern, eastern, and western sections, each containing two lanes: one for through traffic and one for left-turns. "
    lane_text = "Each lane is further divided into three segments. Segment 1 is the closest to the intersection. Segment 2 is in the middle. Segment 3 is the farthest. In a lane, there may be early queued vehicles and approaching vehicles traveling in different segments. "
    vehicles_txt = "Early queued vehicles have arrived at the intersection and await passage permission. Approaching vehicles will arrive at the intersection in the future. "
    signal_txt = "The traffic light has 4 signal phases. Each signal relieves vehicles' flow in the group of two specific lanes. The state of the intersection is listed below. It describes:\n- The group of lanes relieving vehicles' flow under each signal phase.\n- The number of early queued vehicles of the allowed lanes of each signal.\n- The number of approaching vehicles in different segments of the allowed lanes of each signal.\n\n"
    state_txt = state2text(current_states[i])
    neighbor_txt_prefix = "You have the capability to take into account the states of neighboring intersections, which are listed below:\n"
    neighbor_txt = None
    answer_txt = "Please answer:\n" + \
                "Which is the most effective traffic signal that will most significantly improve the traffic " + \
                "condition during the next phase?\n\n"
    note_txt = "Note:\n" + \
                "The traffic congestion is primarily dictated by the early queued vehicles, with the MOST significant " + \
                "impact. You MUST pay the MOST attention to lanes with long queue lengths. It is NOT URGENT to " + \
                "consider vehicles in distant segments since they are unlikely to reach the intersection soon.\n\n"
    requirements_text = "Requirements:\n" + \
                    "- Let's think step by step.\n" + \
                    "- You can only choose one signal among ETWT, NTST, ELWL, NLSL:\n" + \
                    "- You must follow the following steps to provide your analysis: " + \
                        "Step 1: Predict the traffic flow of each lane in the upcoming time based on the current situation of this intersection and its adjacent intersections, and determine the most optimal traffic signal currently." + \
                    "Step 2: Answer your chosen signal.\n" + \
                    "- Your choice can only be given after finishing the analysis.\n" + \
                    "- Your choice must be identified by the tag: <signal>YOUR_CHOICE</signal>."

    # for neighbor text                
    inter_name = env.list_intersection[i].inter_name
    intersection = env.intersection_dict[inter_name]
    roads = deepcopy(intersection["roads"])
    neighbor_list = [inter for inter in env.traffic_light_node_dict[inter_name]['neighbor_ENWS'] if inter]
    road_list = list(roads.keys())
    road2inter = [r.replace("road", "intersection")[:-2] for r in road_list] 
    neighbor2loc = {inter: roads[road_list[i]]['location'] for i, inter in enumerate(road2inter) if inter in neighbor_list}

    neighbor_text = ''.join(["The states of intersection in your {}:\n{}".format(loc, state2text(current_states[env.id_to_index[inter]])) for inter, loc in neighbor2loc.items()])


    prompt = [
        {"role": "system",
         "content": system_txt},
        {"role": "user",
         "content": traffic_txt + lane_text + vehicles_txt +"\n\n"
                    + signal_txt
                    + state_txt +
                    neighbor_txt_prefix +
                    neighbor_text +
                    answer_txt +
                    note_txt +
                    requirements_text
         }
    ]

    return prompt

def getPrompt(state_txt):
    # fill information
    signals_text = ""
    for i, p in enumerate(four_phase_list):
        signals_text += phase_explanation_dict_detail[p] + "\n"
    note_txt = "Note:\n" + \
                "The traffic congestion is primarily dictated by the early queued vehicles, with the MOST significant " + \
                "impact. You MUST pay the MOST attention to lanes with long queue lengths. It is NOT URGENT to " + \
                "consider vehicles in distant segments since they are unlikely to reach the intersection soon.\n\n"
    prompt = [
        {"role": "system",
         "content": "You are an expert in traffic management. You can use your knowledge of traffic commonsense to solve this traffic signal control tasks."},
        {"role": "user",
         "content": "A traffic light regulates a four-section intersection with northern, southern, eastern, and western "
                    "sections, each containing two lanes: one for through traffic and one for left-turns. Each lane is "
                    "further divided into three segments. Segment 1 is the closest to the intersection. Segment 2 is in the "
                    "middle. Segment 3 is the farthest. In a lane, there may be early queued vehicles and approaching "
                    "vehicles traveling in different segments. Early queued vehicles have arrived at the intersection and "
                    "await passage permission. Approaching vehicles will arrive at the intersection in the future.\n\n"
                    "The traffic light has 4 signal phases. Each signal relieves vehicles' flow in the group of two "
                    "specific lanes. The state of the intersection is listed below. It describes:\n"
                    "- The group of lanes relieving vehicles' flow under each signal phase.\n"
                    "- The number of early queued vehicles of the allowed lanes of each signal.\n"
                    "- The number of approaching vehicles in different segments of the allowed lanes of each signal.\n\n"
                    + state_txt +
                    "Please answer:\n"
                    "Which is the most effective traffic signal that will most significantly improve the traffic "
                    "condition during the next phase?\n\n"
                    + note_txt +
                    "Requirements:\n"
                    "- Let's think step by step.\n"
                    "- You can only choose one of the signals listed above.\n"
                    "- You must follow the following steps to provide your analysis: Step 1: Provide your analysis "
                    "for identifying the optimal traffic signal. Step 2: Answer your chosen signal.\n"
                    "- Your choice can only be given after finishing the analysis.\n"
                    "- Your choice must be identified by the tag: <signal>YOUR_CHOICE</signal>."
         }
    ]

    return prompt

def action2code(action):
    code = four_phase_list[action]

    return code

def code2action(action):
    code = phase2code[action]

    return code

def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
