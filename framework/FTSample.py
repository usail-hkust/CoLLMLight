from utils.my_utils import *
from utils.utils import *
from models.CoLLMLightAgent import *
import os
import time
import datetime
import numpy as np
import wandb
from utils.cityflow_env import CityFlowEnv
import cityflow as engine
import utils.config as config
from tqdm import tqdm
from copy import deepcopy
import re
import matplotlib.pyplot as plt
from collections import defaultdict

class FTS:
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, roadnet, trafficflow, my_config):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.env = None
        self.roadnet = roadnet
        self.dataset_name = my_config['dataset']
        self.intersection_list = None
        self.num_col = dic_traffic_env_conf['NUM_COL']
        self.num_row = dic_traffic_env_conf['NUM_ROW']
        self.trafficflow = trafficflow
        self.my_config = my_config
        self.memo = my_config['memo']
        self.reward_type = my_config['reward_type']
        self.boundary_distance_threshold = self.my_config['boundary_distance_threshold']
        self.history_data = None
        # self.fail_log_file = f"./fails/{self.dic_agent_conf['LLM_MODEL']}-{self.dic_traffic_env_conf['TRAFFIC_FILE']}-{self.dic_traffic_env_conf['ROADNET_FILE']}.json"
        self.signal_time = 30
        self.seg_num = 10
        self.LLM_rate = 1 # how many inter use LLM to check congestion
        self.default_release_range = 9
        self.LLM_start_timepoint = 0
        self.max_boundary_release_mode = False
        self.meet_max = False
        self.params = None
        self.ignore_occupancy_threshold = my_config['ignore_threshold']
        self.communicate_threshold = 0.3
        self.car_spacing = 9 
        self.FT_data = []
        self.signal_list = list(four_phase_list.keys())
        self.long_info = my_config['long_info']
        self.feed_back = my_config['feed_back']
        self.feed_back_num = my_config['feed_back_num']
        self.debug_wollm = my_config['debug_wollm']
        self.reward_period = my_config['reward_period']
        self.eng = None
        self.history_states = []
        self.data_already_sample_path = None
        # self.data_already_sample_path = './data/FinetuneData v4/{}_FT_newyork2_raw.json'.format(self.memo)
        self.initialize()
        
    def initialize(self):
        path_check(self.dic_path)
        copy_conf_file(self.dic_path, self.dic_agent_conf, self.dic_traffic_env_conf)
        copy_cityflow_file(self.dic_path, self.dic_traffic_env_conf)
        self.env = CityFlowEnv(
            path_to_log=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=self.dic_path
        )
        self.env.reset()
        self.intersection_list = list(self.env.intersection_dict.keys())
        self.history_data_init()
        self.boundary_intersections = self.init_boundary_intersections()
        # self.llm_init()
        self.agent_init()
        self.release_data = []
        timestamp = datetime.datetime.now().strftime('%Y%m%d')
        self.out_file_name = f"{self.memo}-{self.dataset_name}-{timestamp}"
        self.sampled_inter_signal = self.load_inter_signal_from_path()

    def load_inter_signal_from_path(self):
        if self.data_already_sample_path:
            if os.path.exists(self.data_already_sample_path):
                with open(self.data_already_sample_path, 'r') as f:
                    data = json.load(f)
                    self.FT_data.extend(data)
                    inter_signal = {}
                    for d in data:
                        minimal_metric = float('inf')
                        best_signal = None
                        if d['inter_id'] not in inter_signal:
                            inter_signal[d['inter_id']] = []
                        for signal, signal_results in d['metrics'].items():
                            metric = self.calc_signal_metric(signal_results)
                            if metric < minimal_metric:
                                minimal_metric = metric
                                best_signal = signal
                        inter_signal[d['inter_id']].append(best_signal)
                return inter_signal
        else:
            return {}
    
    def calc_signal_metric(self, signal_results):
        # near_ql/near_wt/near_tt/avg_ql/avg_wt/avg_tt/far_ql/far_wt/far_tt
        if self.reward_type == 'near_ql':
            long_reward = signal_results['ql'][0]
        elif self.reward_type == 'far_ql':
            long_reward = signal_results['ql'][-1]
        elif self.reward_type == 'avg_ql':
            long_reward = sum(signal_results['ql'])
        elif self.reward_type == 'near_wt':
            long_reward = signal_results['wt'][0]
        elif self.reward_type == 'far_wt':
            long_reward = signal_results['wt'][-1]
        elif self.reward_type == 'avg_wt':
            long_reward = sum(signal_results['wt'])
        elif self.reward_type == 'near_att':
            long_reward = signal_results['att'][0]
        elif self.reward_type == 'far_att':
            long_reward = signal_results['att'][-1]
        elif self.reward_type == 'avg_att':
            long_reward = sum(signal_results['att'])
        elif self.reward_type == 'qlwt':
            long_reward = sum([signal_results['ql'][i]*signal_results['wt'][i] for i in range(len(signal_results['ql']))])
        return long_reward
    
    def agent_init(self):
        traffic_env_conf = {}
        traffic_env_conf['name'] = self.my_config['dataset']
        traffic_env_conf['size'] = (self.num_row, self.num_col)
        traffic_env_conf['signal_list'] = list(four_phase_list.keys())
        traffic_env_conf['lane_list'] = location_all_direction_dict #T and L , R
        traffic_env_conf['signal_time'] = self.signal_time
        traffic_env_conf['accumulate_patience'] = self.my_config['accumulate_patience'] ##queue 持续积累多少次就视为拥堵
        traffic_env_conf['congestion_scale_ratio'] = self.my_config['congestion_scale_ratio'] ## queue降低为原来的多少视为拥堵解除
        traffic_env_conf['boundary_distance_threshold'] = self.my_config['boundary_distance_threshold']
        traffic_env_conf['intersection_list'] = self.intersection_list
        self.agent_intersection_list = []
        for i, inter_name in enumerate(self.intersection_list):
            agent_conf = {}
            agent_conf['inter_name'] = inter_name
            agent_conf['inter_id'] = i
            agent_conf['boundary'] = True if i in self.boundary_intersections['id_list'] else False
            agent_conf['long_info'] = self.long_info
            agent_conf['feed_back'] = self.feed_back
            agent_conf['feed_back_num'] = self.feed_back_num
            agent = IntersectionAgent(agent_conf, traffic_env_conf, None)
            agent.neighbor_list = self.get_neighbor_list(i)
            self.agent_intersection_list.append(agent)
        # self.global_control_agent = GlobalAgent(traffic_env_conf, self.LLM)
    

    def llm_init(self):
        # congestion_intersection
        model = self.my_config['llm_path']
        model_type = self.my_config['llm_type']
        if model_type == 'gpt':
            self.LLM = GPT_model(model=model)
        elif model_type == 'llama':
            self.LLM = LLAMA_model(model=model)
    
    def history_data_init(self):
        self.history_data = {}
        self.history_data["release_range"] = {lane:{} for lane in location_direction_dict}
        self.history_data["release_car_num"] = [] # the total number of car release by the lane/signal in each step
        self.history_data["input_car_num"] = []
        self.history_data['effective_release_log'] = {}
        self.history_data['perf'] = {}
        self.history_data['perf']['AWT'] = []
        self.history_data['perf']['AQL'] = []
        self.history_data['perf']['ATT'] = []

        self.history_data['llm_resulst'] = {}
        for name in self.intersection_list:
            self.history_data['llm_resulst'][name] = {}
            self.history_data['llm_resulst'][name]['request'] = []
            self.history_data['llm_resulst'][name]['consider'] = []
        
        self.history_data["car_num_inside"] = {}
        self.history_data["car_num_inside"]['waiting'] = []
        self.history_data["car_num_inside"]['running'] = []
        self.history_data["car_num_inside"]['total'] = []
        self.history_data["car_num_inside"]['r_i'] = [] # count according to release and input of boundary inter
        self.history_data['car_num_outside'] = {}
        self.history_data["car_num_outside"]['waiting'] = []
        self.history_data["car_num_outside"]['running'] = []
        self.history_data["car_num_outside"]['total'] = []
        self.history_data["veh_num"] = []
        self.history_data["ql_num"] = []
        self.history_data['reject_num'] = {}
        self.history_data['sum_wait_time'] = 0
        

        self.history_data["car_num_out_of_lane"] = []
        self.history_data["avg_ql_cells"] = []


        self.history_data['boundary'] = {}
        self.history_data['boundary']['release_num'] = []
        self.history_data['boundary']['input_num'] = []
        self.history_data['boundary']['r_i_dif'] = []
        self.history_data['boundary']['max_release'] = []
        self.history_data['boundary']['max_input'] = []
        self.history_data['boundary']['max_r_i_dif'] = []
        self.history_data['boundary']['sum_release'] = []
        self.history_data['boundary']['sum_input'] = []
        self.history_data['boundary']['sum_r_i_dif'] = []
        self.history_data['boundary']['min_release'] = []
        self.history_data['boundary']['min_input'] = []
        self.history_data['boundary']['min_r_i_dif'] = []
        

        #debug
        self.history_data['veh_log'] = {}
        self.history_data['veh_log']['outside_list'] = []
        self.history_data['veh_log']['inside_list'] = []

    def init_boundary_intersections(self):
        boundary_data = {}
        boundary_name_list = []
        boundary_id_list = []
        
        for row_id in range(1, self.num_row+1):
            name_1 = 'intersection_'+'1_'+ str(row_id)
            name_2 = 'intersection_'+ str(self.num_col) +'_'+ str(row_id)
            boundary_name_list.append(name_1)
            boundary_id_list.append(self.intersection_list.index(name_1))
            boundary_name_list.append(name_2)
            boundary_id_list.append(self.intersection_list.index(name_2))
        for col_id in range(2, self.num_col):
            name_1 = 'intersection_'+ str(col_id) +'_1'
            name_2 = 'intersection_'+ str(col_id) +'_' + str(self.num_row)
            boundary_name_list.append(name_1)
            boundary_id_list.append(self.intersection_list.index(name_1))
            boundary_name_list.append(name_2)
            boundary_id_list.append(self.intersection_list.index(name_2))

        # 将 boundary_id_list 和 boundary_name_list 打包在一起，排序后再解包
        sorted_boundary = sorted(zip(boundary_id_list, boundary_name_list))

        # 解包排序后的 boundary_id_list 和 boundary_name_list
        boundary_id_list, boundary_name_list = zip(*sorted_boundary)

        # 转换成列表
        boundary_id_list = list(boundary_id_list)
        boundary_name_list = list(boundary_name_list)
        
        boundary_data['id_list'] = boundary_id_list
        boundary_data['name_list'] = boundary_name_list

        return boundary_data
        
    def test(self, logger, test_round):
        print("================ Start Test ================")
        total_run_cnt = self.dic_traffic_env_conf["RUN_COUNTS"]
        # initialize output streams
        done = False
        state = self.env.reset()
        total_reward = 0.0
        queue_length_episode = []
        waiting_time_episode = [] 
        print("end reset")
        current_time = self.env.get_current_time()  # in seconds
        
        start_time = time.time()
        # state_action_log = [[] for _ in range(len(state))]
        current_states, current_outputs = self.process_state(state)
        self.current_states = current_states
        last_veh_num = 0
        last_action_list = [None]*len(self.current_states)
        self.veh_last_release_mark = {}
        self.inter_ct = defaultdict(int) 
        
        for step_num in tqdm(range(int(total_run_cnt / self.dic_traffic_env_conf['MIN_ACTION_TIME']))):
            if done or current_time >= total_run_cnt:
                break
            
            veh_release_share = {}
            veh_input_share = {}
            self.history_states.append(deepcopy(current_states))
            # update states
            for i, state in enumerate(current_states):
                if step_num > 0:
                    self.agent_intersection_list[i].update_state(state, last_action_list[i], update_memory = True)
                else:
                    self.agent_intersection_list[i].update_state(state, None, update_memory = False)
                # up_down_stream_relation = self.get_up_down_stream_relation(i)
                # up_down_stream_view = self.get_up_down_stream_view(i, up_down_stream_relation)
                veh_release_share[i] = self.agent_intersection_list[i].veh_release_data
                veh_input_share[i] = self.agent_intersection_list[i].veh_input_data
            
            # update_traffic_memories and up down stream view
            self.update_veh_release_mark(veh_release_share)
            lane2upstream_data, lane2downstream_data = self.find_trajectory(veh_input_share)

            # adaptive communication
            for i, state in enumerate(current_states):
                self.agent_intersection_list[i].traffic_memory_update(lane2upstream_data, lane2downstream_data)
                up_down_stream_view = self.get_up_down_stream_view_from_trajectory(i)
                self.agent_intersection_list[i].update_up_down_stream_view(up_down_stream_view)
                if step_num > 0:
                    traffic_state_updown_stream = self.get_up_down_stream_traffic_state(i)
                    self.agent_intersection_list[i].update_traffic_state_updown_stream(traffic_state_updown_stream)
            
            if self.long_info:
                self.summarize_and_update_long_distance_info()

            prompt_agent_id_list = []
            for i, inter_agent in enumerate(self.agent_intersection_list):
                if inter_agent.up_down_stream_view['exist']:
                    prompt_agent_id_list.append(i)

            ## select_signal
            action_list = []
            for inter_id, inter_agent in enumerate(self.agent_intersection_list):
                effective_range_list = self.estimate_effective_range(current_states[i])
                signal_text = inter_agent.select_signal_default(effective_range_list)
                action_list.append(action2code(signal_text))
            
            self.archive = self.save_env()

            ## data sample
            for inter_id in tqdm(prompt_agent_id_list):
                best_reward = float('inf')
                inter_rewards_dict = {}
                if inter_id in self.sampled_inter_signal:
                    if self.inter_ct[inter_id] < len(self.sampled_inter_signal[inter_id]):
                        best_signal = self.sampled_inter_signal[inter_id][self.inter_ct[inter_id]]
                        action_list[inter_id] = action2code(best_signal)
                        self.inter_ct[inter_id] += 1
                        continue
                # self.env.eng.load(self.archive)
                # self.load_env(self.archive)
                effective_range_list = self.estimate_effective_range_new(current_states[inter_id])
                new_data = {}
                new_data['Intersection'] = self.intersection_list[inter_id]
                new_data['Timestep'] = step_num
                new_data['Traffic_state_history'] = self.get_traffic_state_history(inter_id)
                new_data['Current_Observation'] = self.get_current_observation(inter_id)
                up_down_stream_interlanes = new_data['Current_Observation']['up_down_stream_interlanes']
                new_data['Signal_Rank'] = self.get_signal_value(inter_id, effective_range_list)
                # new_data['Signal_Consequence'] = {}
                signal_consequence = {}
                for i in range(len(self.signal_list)):
                    new_action_list = action_list[:]
                    new_action_list[inter_id] = i
                    # self.env.eng.load(self.archive)
                    self.load_env(self.archive)
                    all_state_rewards = []
                    new_action = self.signal_list[i]
                    for step in range(self.reward_period):
                        next_state, _, done, _ = self.env.step(new_action_list)
                        next_states, next_outputs = self.process_state(next_state)
                        new_action_list = self.default_select_signal(next_states)
                        state_rewards_dict = self.calc_state_rewards(inter_id, next_states)
                        all_state_rewards.append(state_rewards_dict)
                        if step == 0:
                            signal_consequence[new_action] = self.get_signal_consequence(inter_id, next_states, up_down_stream_interlanes)

                    action_rewards_dict, long_reward = self.summarize_reward(all_state_rewards)
                    inter_rewards_dict[new_action] = action_rewards_dict
                    if long_reward < best_reward:
                        best_reward = long_reward
                        best_action_id = i
                        best_signal = self.signal_list[i]
                new_data['Signal_Consequence'] = signal_consequence
                new_data['Best_Signal'] = best_signal

                action_list[inter_id] = best_action_id
                # print(self.signal_list[0])
                self.FT_data.append(new_data)
            with open('./data/FinetuneData/SynTrain_sample.json', 'w') as f:
                json.dump(self.FT_data, f, indent=4)
                # self.sample_ft_data(inter_id, inter_rewards_dict)


            # self.env.eng.load(self.archive)
            self.load_env(self.archive)
            next_state, _, done, _ = self.env.step(action_list)
            next_states, next_outputs = self.process_state(next_state)
            last_action_list = action_list[:]

            self.update_history_data(current_states, next_states, current_outputs, next_outputs, action_list)
            global_indicator = {}
            global_indicator['release'] = self.history_data['release_car_num'][-1]
            global_indicator['input'] = self.history_data['input_car_num'][-1]
            # self.global_control_agent.update_state(global_indicator)

            print("all_lane_current_release: {}, max_release: {}".format(self.history_data['release_car_num'][-1], max(self.history_data['release_car_num'])))
            print("all_lane_current_input: {}, max_input: {}".format(self.history_data['input_car_num'][-1], max(self.history_data['input_car_num'])))
            # print("current_car_num: {}, max_car_num: {}".format(self.history_data['car_num'][-1], max(self.history_data['car_num'])))
            
            print("current_car_num_ool: {}, max_car_num_ool: {}".format(self.history_data['car_num_out_of_lane'][-1], max(self.history_data['car_num_out_of_lane'])))

            rewards = self.get_norm_reward(next_state)  # my reward

            current_time = self.env.get_current_time()  # in seconds
            # state = next_state
            current_states = next_states
            self.current_states = current_states
            current_outputs = next_outputs
            # calculate logger results
            total_reward += sum(rewards)
            queue_length_inter = []
            veh_num = []

            for inter in self.env.list_intersection:
                veh_num.append(sum(inter.dic_feature['lane_num_vehicle']))
                queue_length_inter.append(sum(inter.dic_feature['lane_num_waiting_vehicle_in']))
            queue_length_episode.append(sum(queue_length_inter)) 
            # queue_length_inter = np.array(queue_length_inter)
            # ql_num_matrix = queue_length_inter.reshape(self.num_col, self.num_row)
            # waiting time
            waiting_times = []
            for veh in self.env.waiting_vehicle_list:
                waiting_times.append(self.env.waiting_vehicle_list[veh]['time'])
            waiting_time_episode.append(np.mean(waiting_times) if len(waiting_times) > 0 else 0.0)
            self.history_data['sum_wait_time'] = int(sum(waiting_times))
            veh_num = np.sum(veh_num)
            
            ## perf at this point
            awt = round(np.mean(waiting_time_episode),3)
            aql = round(np.mean(queue_length_episode),3)
            att = self.calc_avg_travel_time()
            att = round(att,3)
            throughput = self.calc_throughput()
            print("Queuing Vehicles:", np.sum(queue_length_inter), "Veh num: ", veh_num, "new Vehs: ", veh_num - last_veh_num, "AWT: ", awt, "AQL:", aql, "ATT:", att, 'Throughput:', throughput)
            self.history_data['perf']['AWT'].append(awt)
            self.history_data['perf']['AQL'].append(aql)
            self.history_data['perf']['ATT'].append(att)
            # print("ql statistics: mean: {}, var: {}, min: {}, max: {}, 0ql_num: {}, Max_Avg_ql_cells: {}, Avg_ql_cells: {}".format(round(np.mean(queue_length_inter),4), round(np.var(queue_length_inter),4), round(np.min(queue_length_inter),4), round(np.max(queue_length_inter),4), (queue_length_inter==0).sum(), round(np.max(self.history_data["avg_ql_cells"][-1]),4), round(np.mean(self.history_data["avg_ql_cells"][-1]),4)))
            # boundary release and input statistics
            self.show_boundary_data()
            self.show_car_num()
            last_veh_num = veh_num
            # self.visualize(next_states)
            # self.save_performance_data(self.history_data['perf'])
            # if self.memo == 'FT_avgql1_s8k':
            #     self.plot_performance_data()
        # wandb logger
        total_travel_time = self.calc_avg_travel_time()
        throughput = self.calc_throughput()
        

        results = {
            "test_reward_over": total_reward,
            "test_throughput_over": throughput,
            "test_avg_queue_len_over": np.mean(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "test_queuing_vehicle_num_over": np.sum(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "test_avg_waiting_time_over": np.mean(waiting_time_episode) if len(queue_length_episode) > 0 else 0,
            "test_avg_travel_time_over": total_travel_time}
        logger.log(results)
        with open('./results/latest_results/{}.txt'.format(self.out_file_name), 'w') as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        print("Test Round:", test_round, results)
        f_history_data = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "history_data.json")
        dump_json(self.history_data, f_history_data)
        print("Testing time: ", time.time() - start_time)

        self.env.batch_log_2()

        return results
    
    def calc_throughput(self):
        current_time = self.env.get_current_time()
        vehicle_travel_times = {}
        all_veh = []
        not_leave_veh = []
        for inter in self.env.list_intersection:
            arrive_left_times = inter.dic_vehicle_arrive_leave_time
            for veh in arrive_left_times:
                all_veh.append(veh)
                if "shadow" in veh:
                    continue
                enter_time = arrive_left_times[veh]["enter_time"]
                leave_time = arrive_left_times[veh]["leave_time"]
                if np.isnan(leave_time):
                    not_leave_veh.append(veh)
                if not np.isnan(enter_time):
                    leave_time = leave_time if not np.isnan(leave_time) else current_time
                    if veh not in vehicle_travel_times:
                        vehicle_travel_times[veh] = [leave_time - enter_time]
                    else:
                        vehicle_travel_times[veh].append(leave_time - enter_time)
        leave_veh = list(set(all_veh) - set(not_leave_veh))
        throughput = len(leave_veh)

        return throughput
    
    def summarize_and_update_long_distance_info(self):
        for i, inter_agent in enumerate(self.agent_intersection_list):
            long_distance_info = {}
            for lane in inter_agent.lane_list_onlyTL:
                upstream_related_lanes = {}
                downstream_related_lanes = {}
                upstream_related_lanes = self.search_related_lanes(upstream_related_lanes, i, lane, 'upstream')
                downstream_related_lanes = self.search_related_lanes(downstream_related_lanes, i, lane, 'downstream')
                long_distance_info_this_lane = self.summarize_long_distance_info(upstream_related_lanes, downstream_related_lanes)
                long_distance_info[lane] = deepcopy(long_distance_info_this_lane)
            inter_agent.update_long_distance_info(long_distance_info)
    
    def search_related_lanes(self, related_lanes, i, lane, direc):
        if direc == 'upstream':
            candi_inter_lanes = self.agent_intersection_list[i].traffic_input_memory[lane]['upstream_lanes']
        elif direc == 'downstream':
            candi_inter_lanes = self.agent_intersection_list[i].traffic_input_memory[lane]['downstream_lanes']
        for inter_id, lane in candi_inter_lanes:
            if (inter_id, lane) in related_lanes:
                continue
            if self.agent_intersection_list[inter_id].current_state[lane]['occupancy'] > 0.5:
                related_lanes[(inter_id, lane)] = self.agent_intersection_list[inter_id].current_state[lane]
                related_lanes = self.search_related_lanes(related_lanes, inter_id, lane, direc)
        return related_lanes
    
    def summarize_long_distance_info(self, upstream_related_lanes, downstream_related_lanes):
        long_distance_info = {}
        long_distance_info['upstream'] = {}
        long_distance_info['downstream'] = {}
        long_distance_info['exist'] = []
        if len(upstream_related_lanes) <= 1:
            long_distance_info['upstream']['exist'] = False
        else:
            total_occupancy = 0
            total_queue_num = 0
            total_waiting_time = 0
            for (inter_id, lane), state in upstream_related_lanes.items():
                total_occupancy += state['occupancy']
                total_queue_num += state['queue_car_num']
                total_waiting_time += state['wait_time']
            long_distance_info['upstream']['lane_num'] = len(upstream_related_lanes)
            long_distance_info['upstream']['average_occupancy'] = total_occupancy / len(upstream_related_lanes)
            long_distance_info['upstream']['total_queue_num'] = total_queue_num
            long_distance_info['upstream']['average_waiting_time'] = total_waiting_time / total_queue_num if total_queue_num > 0 else 0.0
            long_distance_info['upstream']['exist'] = True
            long_distance_info['exist'].append('upstream')
        
        if len(downstream_related_lanes) <= 1:
            long_distance_info['downstream']['exist'] = False
        else:
            total_occupancy = 0
            total_queue_num = 0
            total_waiting_time = 0
            for (inter_id, lane), state in downstream_related_lanes.items():
                total_occupancy += state['occupancy']
                total_queue_num += state['queue_car_num']
                total_waiting_time += state['wait_time']
            long_distance_info['downstream']['lane_num'] = len(downstream_related_lanes)
            long_distance_info['downstream']['average_occupancy'] = total_occupancy / len(downstream_related_lanes)
            long_distance_info['downstream']['total_queue_num'] = total_queue_num
            long_distance_info['downstream']['average_waiting_time'] = total_waiting_time / total_queue_num if total_queue_num > 0 else 0.0
            long_distance_info['downstream']['exist'] = True
            long_distance_info['exist'].append('downstream')

        return long_distance_info

    def get_traffic_state_history(self, inter_id):
        traffic_state_log = self.agent_intersection_list[inter_id].traffic_state_log
        traffic_state_log = traffic_state_log[-5:]
        return traffic_state_log
    
    def get_signal_consequence(self, inter_id, next_states, up_down_stream_interlanes):
        non_empty_lanes = self.agent_intersection_list[inter_id].no_empty_lanes
        last_state = self.current_states[inter_id]
        local_next_state = next_states[inter_id]
        new_traffic_states = {}
        for lane in non_empty_lanes:
            new_traffic_states[lane] = {}
            cars_before = list(last_state[lane]['veh2cell'].keys()) if last_state else []
            cars_current = list(local_next_state[lane]['veh2cell'].keys())
            cars_output = [veh for veh in cars_before if veh not in cars_current]
            cars_input = [veh for veh in cars_current if veh not in cars_before]
            new_traffic_states[lane]['Cars Input'] = len(cars_input)
            new_traffic_states[lane]['Cars Output'] = len(cars_output)
            queue_before = last_state[lane]['queue_len'] if last_state else 0
            queue_current = local_next_state[lane]['queue_len']
            queue_diff = queue_current - queue_before
            new_traffic_states[lane]['Queued Cars Change'] = queue_diff
            new_traffic_states[lane]['Queued Cars'] = queue_current
            moving_before = sum(last_state[lane]['cells']) if last_state else 0
            moving_current = sum(local_next_state[lane]['cells'])
            moving_diff = moving_current - moving_before
            new_traffic_states[lane]['Moving Cars Change'] = moving_diff
            new_traffic_states[lane]['Moving Cars'] = moving_current
            avg_wait_time_before = last_state[lane]['avg_wait_time'] if last_state else 0
            avg_wait_time_before = round(avg_wait_time_before/60, 2)
            avg_wait_time_current = local_next_state[lane]['avg_wait_time']
            avg_wait_time_current = round(avg_wait_time_current/60, 2)
            avg_wait_time_diff = avg_wait_time_current - avg_wait_time_before
            new_traffic_states[lane]['Average Waiting Time Change (mins)'] = round(avg_wait_time_diff,2)
            new_traffic_states[lane]['Average Waiting Time (mins)'] = avg_wait_time_current
            occupancy_before = last_state[lane]['occupancy'] if last_state else 0
            occupancy_before = round(occupancy_before*100, 2)
            occupancy_current = local_next_state[lane]['occupancy']
            occupancy_current = round(occupancy_current*100, 2)
            occupancy_diff = occupancy_current - occupancy_before
            new_traffic_states[lane]['Occupancy Change (%)'] = round(occupancy_diff,2)
            new_traffic_states[lane]['Occupancy (%)'] = occupancy_current

        traffic_state_updown_stream = {}
        for lane in location_direction_dict:
            upstream_lanes = self.agent_intersection_list[inter_id].traffic_input_memory[lane]['upstream_lanes']
            downstream_lanes = self.agent_intersection_list[inter_id].traffic_input_memory[lane]['downstream_lanes']
            if len(upstream_lanes) > 0:
                for upstream_id, up_lane in upstream_lanes:
                    if (upstream_id, up_lane) in up_down_stream_interlanes:
                    # if (upstream_id, up_lane) in self.communicate_lanes:
                        name = f"{lane}'s upstream lane ({upstream_id}, {up_lane})"
                        last_lane_state = self.current_states[upstream_id][up_lane]
                        current_lane_state = next_states[upstream_id][up_lane]
                        traffic_state_updown_stream[name] = self.get_lane_traffic_state(last_lane_state, current_lane_state)

            if len(downstream_lanes) > 0:
                for downstream_id, down_lane in downstream_lanes:
                    if (downstream_id, down_lane) in up_down_stream_interlanes:
                    # if (downstream_id, down_lane) in self.communicate_lanes:
                        name = f"{lane}'s downstream lane ({downstream_id}, {down_lane})"
                        last_state = self.current_states[downstream_id][down_lane]
                        current_state = next_states[downstream_id][down_lane]
                        traffic_state_updown_stream[name] = self.get_lane_traffic_state(last_state, current_state)
        new_traffic_states.update(deepcopy(traffic_state_updown_stream))
        return new_traffic_states

        
    def get_lane_traffic_state(self, last_state, current_state):
        lane_traffic_state = {}
        cars_before = list(last_state['veh2cell'].keys())
        cars_current = list(current_state['veh2cell'].keys())
        cars_output = [veh for veh in cars_before if veh not in cars_current]
        cars_input = [veh for veh in cars_current if veh not in cars_before]
        lane_traffic_state['Cars Input'] = len(cars_input)
        lane_traffic_state['Cars Output'] = len(cars_output)
        queue_before = last_state['queue_len']
        queue_current = current_state['queue_len']
        queue_diff = queue_current - queue_before
        lane_traffic_state['Queued Cars Change'] = queue_diff
        lane_traffic_state['Queued Cars'] = queue_current
        moving_before = sum(last_state['cells'])
        moving_current = sum(current_state['cells'])
        moving_diff = moving_current - moving_before
        lane_traffic_state['Moving Cars Change'] = moving_diff
        lane_traffic_state['Moving Cars'] = moving_current
        avg_wait_time_before = last_state['avg_wait_time']
        avg_wait_time_before = round(avg_wait_time_before/60, 2)
        avg_wait_time_current = current_state['avg_wait_time']
        avg_wait_time_current = round(avg_wait_time_current/60, 2)
        avg_wait_time_diff = avg_wait_time_current - avg_wait_time_before
        lane_traffic_state['Average Waiting Time Change (mins)'] = round(avg_wait_time_diff,2)
        lane_traffic_state['Average Waiting Time (mins)'] = avg_wait_time_current
        occupancy_before = last_state['occupancy']
        occupancy_before = round(occupancy_before*100, 2)
        occupancy_current = current_state['occupancy']
        occupancy_current = round(occupancy_current*100, 2)
        occupancy_diff = occupancy_current - occupancy_before
        lane_traffic_state['Occupancy Change (%)'] = round(occupancy_diff,2)
        lane_traffic_state['Occupancy (%)'] = occupancy_current

        return lane_traffic_state

    def get_signal_value(self, inter_id, effective_range_list):
        lane_release_metrix = {}
        state = self.current_states[inter_id]
        for i, lane in enumerate(location_direction_dict):
            lane_range = effective_range_list[i]
            going_cars_num = np.sum(state[lane]["cells"][:lane_range+1])
            stop_cars_num = np.sum(state[lane]["ql_cells"][:lane_range+1])
            lane_release_metrix[lane] = stop_cars_num * state[lane]["avg_wait_time"] + stop_cars_num * self.signal_time + going_cars_num * self.signal_time 
        signal_value_dict = {}
        for p in self.signal_list:
            signal_value_dict[p] = lane_release_metrix[p[:2]] + lane_release_metrix[p[2:]]
        return signal_value_dict

    def estimate_effective_range_new(self, state):
        car_speed = 11
        effective_range_distance = car_speed * self.signal_time
        range_list = []
        
        for lane in location_direction_dict:
            road_length = state[lane]['road_length']
            seg_length = road_length/self.seg_num
            effective_range_cell = int(effective_range_distance//seg_length)
            if effective_range_cell >= self.seg_num:
                effective_range_cell = self.seg_num - 1
            range_list.append(effective_range_cell)
        return range_list

    def get_current_observation(self, inter_id):
        # long_distance_exist = False 
        # prompt = "####**Long-distance upstream and downstream information:** \n"
        # prompt += "|Relation|The number of lanes whose occupancy exceeds half|Queued Cars|Average Waiting Time (mins)|Average Occupancy|\n"
        # for lane in self.no_empty_lanes:
        #     if len(self.long_distance_info[lane]['exist']) > 0:
        #         for direc in self.long_distance_info[lane]['exist']:
        #             long_distance_exist = True
        #             prompt += "|{}' {}|{}|{}|{:.1f}|{:.1f}%| \n".format(lane, direc, self.long_distance_info[lane][direc]['lane_num'], self.long_distance_info[lane][direc]['total_queue_num'], self.long_distance_info[lane][direc]['average_waiting_time']/60, self.long_distance_info[lane][direc]['average_occupancy']*100)
        # if not long_distance_exist:
        #     return ""

        current_observation = {}
        current_state = self.agent_intersection_list[inter_id].current_state
        memories = self.agent_intersection_list[inter_id].memories
        non_empty_lanes = self.agent_intersection_list[inter_id].no_empty_lanes
        
        current_observation['empty_lanes'] = self.agent_intersection_list[inter_id].empty_lanes
        current_observation['non_empty_lanes'] = {}
        current_observation['up_down_stream_view'] = self.agent_intersection_list[inter_id].up_down_stream_view
        current_observation['up_down_stream_interlanes'] = [] 
        
        for lane in non_empty_lanes:
            current_observation['non_empty_lanes'][lane] = {}
            current_observation['non_empty_lanes'][lane]['queue_car_num'] = current_state[lane]['queue_car_num']
            current_observation['non_empty_lanes'][lane]['coming_car_num'] = current_state[lane]['coming_car_num']
            current_observation['non_empty_lanes'][lane]['avg_wait_time'] = current_state[lane]['avg_wait_time']/60
            current_observation['non_empty_lanes'][lane]['occupancy'] = current_state[lane]['occupancy']*100
        
        
        up_down_stream_view = self.agent_intersection_list[inter_id].up_down_stream_view
        for lane in non_empty_lanes:
            for direc in ['upstream', 'downstream']:
                if direc in up_down_stream_view[lane]:
                    stream_lanes_data = up_down_stream_view[lane][direc]
                    # ct = 1
                    for stream_lane in stream_lanes_data:
                        stream_lane_data = stream_lanes_data[stream_lane]
                        inter_lane = (stream_lane_data['inter_id'], stream_lane)
                        current_observation['up_down_stream_interlanes'].append(inter_lane)
        
        current_observation['long_distance_info'] = self.agent_intersection_list[inter_id].long_distance_info

        current_observation['memory'] = {}
        recent_memory = self.agent_intersection_list[inter_id].recent_memory
        current_observation['memory']['recent_memory'] = recent_memory
        current_observation['memory']['lane_memories'] = []
        if recent_memory:
            if len(recent_memory['pos']) > 0:
                for lane, memory_idx in recent_memory['pos']:
                    lane_memory = memories[lane][memory_idx]
                    current_observation['memory']['lane_memories'].append(lane_memory)

        similar_lane_memory = []
        for lane in non_empty_lanes:
            lane_occupancy = current_state[lane]['occupancy']
            min_similar = 0.5
            similar_idx = None
            if len(memories[lane])>1:
                for i in range(len(memories[lane])-1):
                    similar = abs(memories[lane][i]['occupancy_before'] - lane_occupancy)
                    if similar < min_similar:
                        min_similar = similar
                        similar_idx = i
                if similar_idx:
                    similar_lane_memory.append(memories[lane][similar_idx])
        current_observation['memory']['similar_lane_memory'] = similar_lane_memory

        return current_observation
        
        
        


        
        
        

    
    def get_up_down_stream_traffic_state(self, inter_id):
        traffic_state_updown_stream = {}
        for lane in location_direction_dict:
            upstream_lanes = self.agent_intersection_list[inter_id].traffic_input_memory[lane]['upstream_lanes']
            downstream_lanes = self.agent_intersection_list[inter_id].traffic_input_memory[lane]['downstream_lanes']
            if len(upstream_lanes) > 0:
                for upstream_id, up_lane in upstream_lanes:
                    if self.current_states[upstream_id][up_lane]['occupancy'] > self.ignore_occupancy_threshold:
                    # if (upstream_id, up_lane) in self.communicate_lanes:
                        name = f"{lane}'s upstream lane ({upstream_id}, {up_lane})"
                        last_state = self.history_states[-2][upstream_id][up_lane]
                        current_state = self.history_states[-1][upstream_id][up_lane]
                        traffic_state_updown_stream[name] = self.get_lane_traffic_state(last_state, current_state)

            if len(downstream_lanes) > 0:
                for downstream_id, down_lane in downstream_lanes:
                    if self.current_states[downstream_id][down_lane]['occupancy'] > self.ignore_occupancy_threshold:
                    # if (downstream_id, down_lane) in self.communicate_lanes:
                        name = f"{lane}'s downstream lane ({downstream_id}, {down_lane})"
                        last_state = self.history_states[-2][downstream_id][down_lane]
                        current_state = self.history_states[-1][downstream_id][down_lane]
                        traffic_state_updown_stream[name] = self.get_lane_traffic_state(last_state, current_state)

        return traffic_state_updown_stream

    def get_up_down_stream_view_from_trajectory(self, inter_id):
        view = {}
        view['exist'] = False
        no_empty_lanes = self.agent_intersection_list[inter_id].no_empty_lanes
        #
        for lane in no_empty_lanes:
            view[lane] = {}
            upstream_lanes = self.agent_intersection_list[inter_id].traffic_input_memory[lane]['upstream_lanes']
            downstream_lanes = self.agent_intersection_list[inter_id].traffic_input_memory[lane]['downstream_lanes']
            if len(upstream_lanes) > 0:
                view[lane]['upstream'] = {}
                for upstream_id, up_lane in upstream_lanes:
                    if self.current_states[upstream_id][up_lane]['occupancy'] > self.ignore_occupancy_threshold:
                    # if (upstream_id, up_lane) in self.communicate_lanes:
                        view[lane]['upstream'][up_lane] = self.current_states[upstream_id][up_lane]
                        view[lane]['upstream'][up_lane]['inter_id'] = upstream_id
                        view['exist'] = True 
                
            if len(downstream_lanes) > 0:
                view[lane]['downstream'] = {}
                for downstream_id, down_lane in downstream_lanes:
                    if self.current_states[downstream_id][down_lane]['occupancy'] > self.ignore_occupancy_threshold:
                    # if (downstream_id, down_lane) in self.communicate_lanes:
                        view[lane]['downstream'][down_lane] = self.current_states[downstream_id][down_lane]
                        view[lane]['downstream'][down_lane]['inter_id'] = downstream_id
                        view['exist'] = True

        return view 
    
    def calc_state_rewards(self, inter_id, next_states):
        queue_length_inter = []
        for inter in self.env.list_intersection:
            queue_length_inter.append(sum(inter.dic_feature['lane_num_waiting_vehicle_in']))
        ql = sum(queue_length_inter)
        waiting_times = []
        for veh in self.env.waiting_vehicle_list:
            waiting_times.append(self.env.waiting_vehicle_list[veh]['time'])
        wt = np.mean(waiting_times) if len(waiting_times) > 0 else 0.0
        att = self.calc_avg_travel_time()
        rewards = {}
        rewards['ql'] = ql
        rewards['wt'] = wt
        rewards['att'] = att
        neighbor_list = self.get_neighbor_list(inter_id)
        neighbor_occupancy_total = 0
        id_list = [inter_id]
        for neighbor in neighbor_list:
            id_list.append(neighbor['id'])
        for idx in id_list:
            for lane in location_direction_dict:
                neighbor_occupancy_total += next_states[idx][lane]['occupancy']
        rewards['ocp'] = neighbor_occupancy_total

        return rewards
    
    def save_env(self):
        self.eng = self.env.eng
        eng_snapshot = self.env.eng.snapshot()
        self.env.eng = None
        for inter in self.env.list_intersection:
            inter.eng = None
        env = deepcopy(self.env)
        archive = {}
        archive['eng'] = eng_snapshot
        archive['env'] = env
        return archive
    
    def load_env(self, archive):
        self.env = deepcopy(archive['env'])
        self.eng.load(archive['eng'])
        self.env.eng = self.eng
        for inter in self.env.list_intersection:
            inter.eng = self.env.eng
        
        

    
    def default_select_signal(self, next_states):
        action_list = []
        for state in next_states:
            effective_range_list = self.estimate_effective_range(state)
            lane_release_metrix = {}
            for i, lane in enumerate(location_direction_dict):
                lane_range = effective_range_list[i]
                going_cars_num = np.sum(state[lane]["cells"][:lane_range+1])
                stop_cars_num = np.sum(state[lane]["ql_cells"][:lane_range+1])
                lane_release_metrix[lane] = stop_cars_num * state[lane]["avg_wait_time"] + stop_cars_num * self.signal_time + going_cars_num * self.signal_time 
            phase_release_metrix = []
            for p in self.signal_list:
                phase_release_metrix.append(lane_release_metrix[p[:2]] + lane_release_metrix[p[2:]])
            index = phase_release_metrix.index(max(phase_release_metrix))
            signal_text = self.signal_list[index]
            action_list.append(action2code(signal_text))
        return action_list


        
    
    def summarize_reward(self, all_state_rewards):
        qls = []
        wts = []
        atts = []
        ocps = []
        for reward in all_state_rewards:
            qls.append(reward['ql'])
            wts.append(reward['wt'])
            atts.append(reward['att'])
            ocps.append(reward['ocp'])
        action_rewards_dict = {}
        action_rewards_dict['ql'] = qls
        action_rewards_dict['wt'] = wts
        action_rewards_dict['att'] = atts
        action_rewards_dict['ocp'] = ocps
        long_reward = self.calc_long_reward(qls, wts, atts, ocps)

        return action_rewards_dict, long_reward
    
    def calc_long_reward(self, qls, wts, atts, ocps):
        # near_ql/near_wt/near_tt/avg_ql/avg_wt/avg_tt/far_ql/far_wt/far_tt
        if self.reward_type == 'near_ql':
            long_reward = qls[0]
        elif self.reward_type == 'far_ql':
            long_reward = qls[-1]
        elif self.reward_type == 'avg_ql':
            long_reward = sum(qls)
        elif self.reward_type == 'near_wt':
            long_reward = wts[0]
        elif self.reward_type == 'far_wt':
            long_reward = wts[-1]
        elif self.reward_type == 'avg_wt':
            long_reward = sum(wts)
        elif self.reward_type == 'near_att':
            long_reward = atts[0]
        elif self.reward_type == 'far_att':
            long_reward = atts[-1]
        elif self.reward_type == 'avg_att':
            long_reward = sum(atts)
        elif self.reward_type == 'qlwt':
            long_reward = sum([qls[i]*wts[i] for i in range(len(qls))])
        elif self.reward_type == 'early_ocp':
            long_reward = ocps[0]
        elif self.reward_type == 'far_ocp':
            long_reward = ocps[-1]
        elif self.reward_type == 'avg_ocp':
            long_reward = sum(ocps)
        return long_reward
        
        
    
    def find_min_signal(self, inter_rewards_dict, key, i):
        min_sum = float('inf')
        best_signal = None
        for signal in inter_rewards_dict.keys():
            ql_list = inter_rewards_dict[signal][key]
            if len(ql_list) >= i:
                current_sum = sum(ql_list[:i])
                if current_sum < min_sum:
                    min_sum = current_sum
                    best_signal = signal
        return best_signal
    
    def find_signal_with_min_value_at_i(self, inter_rewards_dict, i):
        min_value = float('inf')
        best_signal = None
        for signal in inter_rewards_dict.keys():
            ql_list = inter_rewards_dict[signal]['att']
            if len(ql_list) > i:
                value_at_i = ql_list[i]
                if value_at_i < min_value:
                    min_value = value_at_i
                    best_signal = signal
        return best_signal
        
    def sample_ft_data(self, inter_id, inter_rewards_dict):
        sample = {}
        sample['inter_id'] = inter_id
        sample['intersection'] = self.intersection_list[inter_id]
        sample['state'] = {}
        sample['state']['local'] = self.agent_intersection_list[inter_id].current_state
        sample['state']['ud_stream_view'] = self.agent_intersection_list[inter_id].up_down_stream_view
        sample['metrics'] = inter_rewards_dict
        sample['best_signal'] = {}
        ql_best_signal = []
        wt_best_signal = []
        att_best_signal = []
        for i in range(self.reward_period):
            ql_best_signal.append(self.find_min_signal(inter_rewards_dict, 'ql', i+1))
            wt_best_signal.append(self.find_min_signal(inter_rewards_dict, 'wt', i+1))
            att_best_signal.append(self.find_signal_with_min_value_at_i(inter_rewards_dict, i))
        sample['best_signal']['ql'] = ql_best_signal
        
        sample['best_signal']['wt'] = wt_best_signal
        sample['best_signal']['att'] = att_best_signal
        self.FT_data.append(sample)
        with open('./data/Finetune/SynTrain_sample.json', 'w') as f:
            json.dump(self.FT_data, f, indent=4)

    def get_up_down_stream_relation(self, inter_id):
        upstream_relation = {'NT': ['North', ['NT', 'EL', 'WR']], 'NL': ['North', ['NT', 'EL', 'WR']], 'ET': ['East', ['ET', 'SL', 'NR']], 'EL': ['East', ['ET', 'SL', 'NR']], 'ST': ['South', ['ST', 'WL', 'ER']], 'SL': ['South', ['ST', 'WL', 'ER']], 'WT': ['West', ['WT', 'NL', 'SR']], 'WL': ['West', ['WT', 'NL', 'SR']]}
        downstream_relation = {'NT': ['South', ['NR','NT','NL']], 'NL': ['East', ['WR', 'WT', 'WL']], 'ET': ['West', ['ER', 'ET', 'EL']], 'EL': ['South', ['NR','NT','NL']], 'ST': ['North', ['SR', 'ST', 'SL']], 'SL': ['West', ['ER', 'ET', 'EL']], 'WT': ['East', ['WR', 'WT', 'WL']], 'WL': ['North', ['SR', 'ST', 'SL']]} 
        neighbor_list = self.get_neighbor_list(inter_id)
        loc2id = {}
        for neighbor in neighbor_list:
            loc2id[neighbor['location']] = neighbor['id']
        up_down_stream_relation = {}
        for lane in location_direction_dict:
            upstream_location = upstream_relation[lane][0]
            upstream_lanes = upstream_relation[lane][1]
            downstream_lanes = downstream_relation[lane][1]
            downstream_location = downstream_relation[lane][0]
            if upstream_location in loc2id:
                upstream_id = loc2id[upstream_location]
            else:
                upstream_id = None
            if downstream_location in loc2id:
                downstream_id = loc2id[downstream_location]
            else:
                downstream_id = None
            up_down_stream_relation[lane] = {'upstream_location': upstream_location, 'upstream_id': upstream_id, 'upstream_lanes': upstream_lanes, 'downstream_location': downstream_location,'downstream_id': downstream_id, 'downstream_lanes': downstream_lanes}
        
        return up_down_stream_relation
    
    def get_up_down_stream_view(self, inter_id, up_down_stream_relation):
        view = {}
        view['exist'] = False
        no_empty_lanes = self.agent_intersection_list[inter_id].no_empty_lanes
        if len(no_empty_lanes) <= 1:
            return view
        elif len(no_empty_lanes) == 2:
            if set(no_empty_lanes) in [set(['WT', 'ET']), set(['EL', 'WL']), set(['NT', 'ST']), set(['SL', 'NL'])]:
                return view

        for lane in no_empty_lanes:
            view[lane] = {}
            upstream_id = up_down_stream_relation[lane]['upstream_id']
            upstream_location = up_down_stream_relation[lane]['upstream_location']
            downstream_id = up_down_stream_relation[lane]['downstream_id']
            downstream_location = up_down_stream_relation[lane]['downstream_location']
            if up_down_stream_relation[lane]['upstream_id'] is not None:
                view[lane]['upstream'] = {}
                for up_lane in up_down_stream_relation[lane]['upstream_lanes']:
                    if self.current_states[upstream_id][up_lane]['occupancy'] > self.ignore_occupancy_threshold:
                        view[lane]['upstream'][up_lane] = self.current_states[upstream_id][up_lane]
                        view[lane]['upstream'][up_lane]['location'] = upstream_location
                        view['exist'] = True     
            if up_down_stream_relation[lane]['downstream_id'] is not None:
                view[lane]['downstream'] = {}
                for down_lane in up_down_stream_relation[lane]['downstream_lanes']:
                    if self.current_states[downstream_id][down_lane]['occupancy'] > self.ignore_occupancy_threshold:
                        view[lane]['downstream'][down_lane] = self.current_states[downstream_id][down_lane]
                        view[lane]['downstream'][down_lane]['location'] = downstream_location
                        view['exist'] = True
        return view 
                             
    def update_veh_release_mark(self, veh_release_share):
        new_release_data = {}
        for inter_id in veh_release_share:
            for lane in veh_release_share[inter_id]:
                release_vehs = veh_release_share[inter_id][lane]
                for veh in release_vehs:
                    new_release_data[veh] = (inter_id, lane)
        self.veh_last_release_mark.update(new_release_data)
                
    def find_trajectory(self, veh_input_share):
        lane2upstream = {}
        lane2downstream = {}
        for inter_id in veh_input_share:
            for lane in veh_input_share[inter_id]:
                downstream_lane = (inter_id, lane)
                input_vehs = veh_input_share[inter_id][lane]
                for veh in input_vehs:
                    if veh in self.veh_last_release_mark:
                        if downstream_lane not in lane2upstream:
                            lane2upstream[downstream_lane] = {}

                        upstream_lane = self.veh_last_release_mark[veh]
                        if upstream_lane not in lane2upstream[downstream_lane]:
                            lane2upstream[downstream_lane][upstream_lane] = 0
                        lane2upstream[downstream_lane][upstream_lane] += 1

                        if upstream_lane not in lane2downstream:
                            lane2downstream[upstream_lane] = {}

                        if downstream_lane not in lane2downstream[upstream_lane]:
                            lane2downstream[upstream_lane][downstream_lane] = 0
                        lane2downstream[upstream_lane][downstream_lane] += 1

        return lane2upstream, lane2downstream
    
    def message_passing(self):
        messages_for_all_inter = {}
        #messages_for_all_inter: {inter_id1: [{inter_id: inter_id2, side: 'North', type: 'downstream', congestion_degree:60%, congest_tree_size:3}, {...}]}

        for i in range(len(self.intersection_list)):
            messages_for_all_inter[i] = {}
        
        # (inter_id, lane), decide the upstream and downstream lane by traffic logs
        # congestion_dict: {(2, 'NL'): {upstream: [(3,'NT')], downstream: [(4,'ST')], congestion_degree:60%}
        congestion_dict = self.check_congestion() # find congestion lane and their upstream, down

        for key in congestion_dict:
            inter_id, lane = key
            congestion_degree = congestion_dict[key]['congestion_degree']
            upstream_lanes = congestion_dict[key]['upstream']
            downstream_lanes = congestion_dict[key]['downstream']
            upstream_congest_degree_list = [congestion_degree]
            upstream_congest_lane_list = [key] # include this inter itself
            upstream_queue_num_list = [self.current_states[inter_id][lane]['queue_len']]
            upstream_avg_wait_time_list = [self.current_states[inter_id][lane]['avg_wait_time']]
            upstream_congest_lane_list, upstream_congest_degree_list, upstream_queue_num_list, upstream_avg_wait_time_list = self.search_tree(key, congestion_dict, 'upstream', upstream_congest_lane_list, upstream_congest_degree_list, upstream_queue_num_list, upstream_avg_wait_time_list)
            
            downstream_congest_degree_list = [congestion_degree]
            downstream_congest_lane_list = [key]
            downstream_queue_num_list = [self.current_states[inter_id][lane]['queue_len']]
            downstream_avg_wait_time_list = [self.current_states[inter_id][lane]['avg_wait_time']]
            downstream_congest_lane_list, downstream_congest_degree_list, downstream_queue_num_list, downstream_avg_wait_time_list = self.search_tree(key, congestion_dict, 'downstream', downstream_congest_lane_list, downstream_congest_degree_list, downstream_queue_num_list, downstream_avg_wait_time_list)

            for upstream_key in upstream_lanes:
                target_id, target_lane = upstream_key
                message = {'inter_id': inter_id, 'congest_lane': lane, 'target_lane': target_lane, 'type':'downstream', 'congestion_degree': congestion_degree, 'congest_lane_list': downstream_congest_lane_list, 'congest_degree_list': downstream_congest_degree_list, 'queue_num_list': downstream_queue_num_list, 'avg_wait_time_list': downstream_avg_wait_time_list,}
                if target_lane not in messages_for_all_inter[target_id]:
                    messages_for_all_inter[target_id][target_lane] = []
                messages_for_all_inter[target_id][target_lane].append(message)

            for downstream_key in downstream_lanes:
                target_id, target_lane = downstream_key
                message = {'inter_id': inter_id, 'congest_lane': lane, 'target_lane': target_lane, 'type':'upstream', 'congestion_degree': congestion_degree, 'congest_lane_list': upstream_congest_lane_list, 'congest_degree_list': upstream_congest_degree_list, 'queue_num_list': upstream_queue_num_list, 'avg_wait_time_list': upstream_avg_wait_time_list}
                if target_lane not in messages_for_all_inter[target_id]:
                    messages_for_all_inter[target_id][target_lane] = []
                messages_for_all_inter[target_id][target_lane].append(message)
        ## aggregate same message to same lane
        for target_id in messages_for_all_inter:
            for lane in messages_for_all_inter[target_id]:
                messages = messages_for_all_inter[target_id][lane]
                congest_lane_list = {}
                congest_degree_list = {}
                queue_num_list = {}
                avg_wait_time_list = {}
                for message_type in ['upstream', 'downstream']:
                    congest_degree_list[message_type] = []
                    congest_lane_list[message_type] = []
                    avg_wait_time_list[message_type] = []
                    queue_num_list[message_type] = []
                for message in messages:
                    message_type = message['type']
                    congest_degree_list[message_type].extend(message['congest_degree_list'])
                    congest_lane_list[message_type].extend(message['congest_lane_list'])
                    queue_num_list[message_type].extend(message['queue_num_list'])
                    avg_wait_time_list[message_type].extend(message['avg_wait_time_list'])
                
                upstream_congest_num = len(congest_lane_list['upstream'])
                upstream_avg_congest = np.mean(congest_degree_list['upstream']) if upstream_congest_num > 0 else 0.0
                downstream_congest_num = len(congest_lane_list['downstream'])
                downstream_avg_congest = np.mean(congest_degree_list['downstream']) if downstream_congest_num > 0 else 0.0
                sum_queue_num_upstream = sum(queue_num_list['upstream'])
                sum_queue_num_downstream = sum(queue_num_list['downstream'])
                upstream_awt = sum([queue_num_list['upstream'][i]*avg_wait_time_list['upstream'][i] for i in range(len(queue_num_list['upstream']))])/sum_queue_num_upstream if sum_queue_num_upstream > 0 else 0.0
                downstream_awt = sum([queue_num_list['downstream'][i]*avg_wait_time_list['downstream'][i] for i in range(len(queue_num_list['downstream']))])/sum_queue_num_downstream if sum_queue_num_downstream > 0 else 0.0
                aggregated_message = {'upstream_congest_num': upstream_congest_num, 'upstream_avg_congest': upstream_avg_congest, 'downstream_congest_num': downstream_congest_num, 'downstream_avg_congest': downstream_avg_congest, 'upstream_awt': upstream_awt, 'downstream_awt': downstream_awt}
                messages_for_all_inter[target_id][lane] = aggregated_message

        return messages_for_all_inter

    def check_congestion(self):
        # congestion_dict: {(2, 'NL'): {upstream: [(3,'NT')], downstream: [(4,'ST')], congestion_degree:60%}
        congestion_dict = dict()
        for inter_id in range(len(self.intersection_list)):
            congest_data = self.agent_intersection_list[inter_id].congest_data
            congestion_dict.update(congest_data)
        return congestion_dict
            
    def search_tree(self, key, congestion_dict, direction, congest_lane_list, congest_degree_list, queue_num_list, avg_wait_time_list):
        subtree_interlanes = congestion_dict[key][direction]
        for interlane in subtree_interlanes:
            if interlane in congestion_dict and interlane not in congest_lane_list:
                inter_id, lane = interlane
                congest_lane_list.append(interlane)
                congest_degree_list.append(congestion_dict[interlane]['congestion_degree'])
                queue_num_list.append(self.current_states[inter_id][lane]['queue_len'])
                avg_wait_time_list.append(self.current_states[inter_id][lane]['avg_wait_time'])
                congest_lane_list, congest_degree_list, queue_num_list, avg_wait_time_list = self.search_tree(interlane, congestion_dict, direction, congest_lane_list, congest_degree_list, queue_num_list, avg_wait_time_list)


        return congest_lane_list, congest_degree_list, queue_num_list, avg_wait_time_list
    
    def save_performance_data(self, data):
        # Load existing data
        filename = f'{self.out_file_name}.json'
        filepath = './results/perf_logs/'+ filename
        perf_data = {}    
        perf_data[self.memo] = data
        # Add or update method data
        # Save updated data
        with open(filepath, 'w') as f:
            json.dump(perf_data, f, indent=4)
    
    def save_llm_response_data(self, data):
        # Load existing data
        filename = f'./results/llm_responses_log/{self.out_file_name}.json'
        llm_data = {}    
        llm_data[self.memo] = data
        with open(filename, 'w') as f:
            json.dump(llm_data, f, indent=4)

    def plot_performance_data(self):
        # Load data
        matching_files = []
        for root, dirs, files in os.walk('./results/perf_logs/'):
            for file in files:
                if file.endswith('.json'):
                    parts = file.split('-')
                    if len(parts) == 3:
                        file_dataset = parts[1]
                        if file_dataset == self.dataset_name:
                            file_path = os.path.join(root, file)
                            matching_files.append(file_path)
        perf_data = {}
        for file_path in matching_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
            except FileNotFoundError:
                data = {}
            perf_data.update(data)

        # Define colors
        colors = plt.cm.get_cmap('tab10', len(perf_data))

        # Create subplots for AWT, AQL, ATT
        fig, axes = plt.subplots(3, 1, figsize=(20, 15))
        metrics = ['AWT', 'AQL', 'ATT']
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            for j, (method_name, method_data) in enumerate(perf_data.items()):
                ax.plot(method_data[metric], label=method_name, color=colors(j))
            ax.set_title(metric)
            ax.legend()
            ax.set_xlabel('Index')
            ax.set_ylabel(metric)
        
        plt.tight_layout()
        plt.savefig('./results/state_img/{}_performance_plots.png'.format(self.memo))
        plt.show()

    def calc_avg_travel_time(self):
        current_time = self.env.get_current_time()
        vehicle_travel_times = {}
        for inter in self.env.list_intersection:
            arrive_left_times = inter.dic_vehicle_arrive_leave_time
            for veh in arrive_left_times:
                if "shadow" in veh:
                    continue
                enter_time = arrive_left_times[veh]["enter_time"]
                leave_time = arrive_left_times[veh]["leave_time"]
                if not np.isnan(enter_time):
                    leave_time = leave_time if not np.isnan(leave_time) else current_time
                    if veh not in vehicle_travel_times:
                        vehicle_travel_times[veh] = [leave_time - enter_time]
                    else:
                        vehicle_travel_times[veh].append(leave_time - enter_time)

        avg_travel_time = np.mean([sum(vehicle_travel_times[veh]) for veh in vehicle_travel_times])
        return avg_travel_time
            
    def show_boundary_data(self):
        cbr = self.history_data['boundary']['release_num'][-1] # current_boundary_release
        cbi = self.history_data['boundary']['input_num'][-1] # current_boundary_input
        cbri = self.history_data['boundary']['r_i_dif'][-1] # current_boundary_release
        boundary_name_list = self.boundary_intersections['name_list']
        self.history_data['boundary']['max_release'].append((max(cbr), boundary_name_list[cbr.index(max(cbr))]))
        self.history_data['boundary']['max_input'].append((max(cbi), boundary_name_list[cbi.index(max(cbi))]))
        self.history_data['boundary']['max_r_i_dif'].append((max(cbri), boundary_name_list[cbri.index(max(cbri))]))
        self.history_data['boundary']['sum_release'].append(round(sum(cbr),4))
        self.history_data['boundary']['sum_input'].append(round(sum(cbi),4))
        self.history_data['boundary']['sum_r_i_dif'].append(round(sum(cbri),4))
        self.history_data['boundary']['min_release'].append((min(cbr), boundary_name_list[cbr.index(min(cbr))]))
        self.history_data['boundary']['min_input'].append((min(cbi), boundary_name_list[cbi.index(min(cbi))]))
        self.history_data['boundary']['min_r_i_dif'].append((min(cbri), boundary_name_list[cbri.index(min(cbri))]))
        print("sum_r_i_dif: {}".format(round(sum(cbri),4)))

        # print("max_release: {} / {}, sum_release: {}, min_release {} / {}".format(max(cbr), boundary_name_list[cbr.index(max(cbr))], round(np.sum(cbr),4), min(cbr), boundary_name_list[cbr.index(min(cbr))]))
        # print("max_input: {} / {}, sum_input: {}, min_input {} / {}".format(max(cbi), boundary_name_list[cbi.index(max(cbi))], round(np.sum(cbi),4), min(cbi), boundary_name_list[cbi.index(min(cbi))]))
        # print("max_r_i_dif: {} / {}, sum_r_i_dif: {}, min_r_i_dif {} / {}".format(max(cbri), boundary_name_list[cbri.index(max(cbri))], round(np.sum(cbri),4), min(cbri), boundary_name_list[cbri.index(min(cbri))]))
    
    def show_car_num(self):
        print("car_num inside: {}, ql_num inside: {}".format(len(self.history_data['veh_log']['inside_list']), self.history_data['car_num_inside']['waiting'][-1]))
    
    def visualize(self, states):
        """
        更新交通情况热度图。
        
        参数：
        ql_num (ndarray): 28x7的汽车队列长度数据。
        wait_time (ndarray): 28x7的等待时间数据。
        """
        # 清除当前图形
        ql_num = []
        wait_time = []
        release_data = self.release_data
        for state in states:
            ql = 0
            wt = 0
            for lane in location_all_direction_dict:
                ql += state[lane]['queue_len']
                veh_list = list(state[lane]['veh2cell'].keys())
                # wt += state[lane]['queue_len'] * state[lane]['avg_wait_time']
                wt += sum([state[lane]['wait_time'][veh] for veh in veh_list])
            ql_num.append(ql)
            wait_time.append(wt)
        ql_num = np.array(ql_num)
        wait_time = np.array(wait_time)
        release_data = np.array(release_data)
        ql_num = np.rot90(ql_num.reshape(self.num_col, self.num_row))
        wait_time = np.rot90(wait_time.reshape(self.num_col, self.num_row))
        release_data = np.rot90(release_data.reshape(self.num_col, self.num_row))
        plt.clf()
        
        # 创建一个包含两个子图的图形
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 20))

        # 绘制汽车队列长度热度图
        im1 = ax1.imshow(ql_num, cmap='Reds', aspect='equal', interpolation='none', vmin=0, vmax=max(100, np.max(ql_num)))
        ax1.set_title('Traffic Queue Length Heatmap')
        ax1.set_xlabel('Intersection')
        ax1.set_ylabel('Street')
        ax1.set_xticks(np.arange(self.num_col))
        ax1.set_xticklabels([f'{i+1}' for i in range(self.num_col)])
        ax1.set_yticks(np.arange(self.num_row))
        ax1.set_yticklabels([f'{i}' for i in np.arange(self.num_row, 0, -1)])
        cbar1 = plt.colorbar(im1, ax=ax1)
        # cbar1.set_label('Queue Length')

        # 绘制等待时间热度图
        im2 = ax2.imshow(wait_time, cmap='Blues', aspect='equal', interpolation='none', vmin=0, vmax=max(20000, np.max(wait_time)))
        ax2.set_title('Traffic Wait Time Heatmap')
        ax2.set_xlabel('Intersection')
        ax2.set_ylabel('Street')
        ax2.set_xticks(np.arange(self.num_col))
        ax2.set_xticklabels([f'{i+1}' for i in range(self.num_col)])
        ax2.set_yticks(np.arange(self.num_row))
        ax2.set_yticklabels([f'{i}' for i in np.arange(self.num_row, 0, -1)])
        cbar2 = plt.colorbar(im2, ax=ax2)
        # cbar2.set_label('Wait Time')

        # 绘制等待时间热度图
        im3 = ax3.imshow(release_data, cmap='Greens', aspect='equal', interpolation='none', vmin=0, vmax=max(10, np.max(release_data)))
        ax3.set_title('Traffic Release Heatmap')
        ax3.set_xlabel('Intersection')
        ax3.set_ylabel('Street')
        ax3.set_xticks(np.arange(self.num_col))
        ax3.set_xticklabels([f'{i+1}' for i in range(self.num_col)])
        ax3.set_yticks(np.arange(self.num_row))
        ax3.set_yticklabels([f'{i}' for i in np.arange(self.num_row, 0, -1)])
        cbar3 = plt.colorbar(im3, ax=ax3)
        # cbar3.set_label('Release Car Num')

        # 调整布局
        plt.tight_layout()

        # 显示图形
        plt.savefig('./results/state_img/heatmaps-{}.png'.format(self.out_file_name))
        # 关闭图形窗口
        plt.close('all')

    def process_state(self, state):
        current_states = []
        current_outputs = []
        for i in range(len(state)):
            # log statistic state
            intersection = self.env.intersection_dict[self.env.list_intersection[i].inter_name]
            roads = deepcopy(intersection["roads"])
            statistic_state, statistic_incoming_state, mean_speed = self.get_state_detail_many_seg_all_lane(roads)
            arrive_left_times = self.env.list_intersection[i].dic_vehicle_arrive_leave_time
            for lane in location_all_direction_dict:
                statistic_state[lane]['stay_time'] = {}
                statistic_state[lane]['occupancy'] = len(statistic_state[lane]['veh2pos'])/(statistic_state[lane]['road_length']//self.car_spacing)
                for veh in statistic_state[lane]['veh2cell']:
                    enter_time = arrive_left_times[veh]["enter_time"]
                    current_time = self.env.current_time
                    statistic_state[lane]['stay_time'][veh] = current_time - enter_time

            current_states.append(statistic_state)
            current_outputs.append(statistic_incoming_state)
        return current_states, current_outputs

    def find_min_spacing(self, veh2pos):
        veh_pos_list = list(veh2pos.values())
        veh_pos_list.sort()
        min_spacing = float('inf')
        for i in range(len(veh_pos_list)):
            for j in range(i + 1, len(veh_pos_list)):
                spacing = abs(veh_pos_list[i]-veh_pos_list[j])
                if spacing < min_spacing:
                    min_spacing = spacing
        return min_spacing

    def get_state_detail_many_seg_all_lane(self, roads):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        divide the lane into many seg
        tag1
        """
        lane_queues = self.env.eng.get_lane_waiting_vehicle_count()
        lane_vehicles = self.env.eng.get_lane_vehicles()

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
                    statistic_state[f"{location_dict_short[roads[r]['location']]}T"] = {"cells": [0 for _ in range(self.seg_num)],
                                                                                        "ql_cells": [0 for _ in range(self.seg_num)],
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
                    statistic_state[f"{location_dict_short[roads[r]['location']]}L"] = {"cells": [0 for _ in range(self.seg_num)],
                                                                                        "ql_cells": [0 for _ in range(self.seg_num)],
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
                    statistic_state[f"{location_dict_short[roads[r]['location']]}R"] = {"cells": [0 for _ in range(self.seg_num)],
                                                                                        "ql_cells": [0 for _ in range(self.seg_num)],
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
                        veh_info = self.env.eng.get_vehicle_info(veh)
                        lane_pos = road_length - float(veh_info["distance"])
                        statistic_state[location_all_direction_dict[lane_group]]["veh2pos"][veh] = lane_pos
                        # update statistic state
                        seg_length = road_length/self.seg_num
                        gpt_lane_cell = int(lane_pos//seg_length)
                        statistic_state[location_all_direction_dict[lane_group]]["veh2cell"][veh] = gpt_lane_cell
                        veh_waiting_time = self.env.waiting_vehicle_list[veh]['time'] if veh in self.env.waiting_vehicle_list else 0.0
                        statistic_state[location_all_direction_dict[lane_group]]["wait_time"][veh] = veh_waiting_time
                        if veh in self.env.waiting_vehicle_list:
                            waiting_times.append(veh_waiting_time)
                        if gpt_lane_cell >= self.seg_num:
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

                statistic_state_incoming[incoming2output[roads[r]['location']]] = {"cells": [0 for _ in range(self.seg_num)],
                                                                                    "ql_cells": [0 for _ in range(self.seg_num)],
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
                        veh_info = self.env.eng.get_vehicle_info(veh)
                        lane_pos = road_length - float(veh_info["distance"])

                        # update statistic state
                        seg_length = road_length/self.seg_num
                        gpt_lane_cell = int(lane_pos//seg_length)
                        statistic_state_incoming[location_incoming_dict[lane_group]]["veh2cell"][veh] = gpt_lane_cell
                        if gpt_lane_cell >= self.seg_num:
                            statistic_state_incoming[location_incoming_dict[lane_group]]["out_of_lane"] += 1
                            
                        else:
                            # speed > 0.1 m/s are approaching vehicles
                            if float(veh_info["speed"]) > 0.1:
                                statistic_state_incoming[location_incoming_dict[lane_group]]["cells"][gpt_lane_cell] += 1
                            else:
                                statistic_state_incoming[location_incoming_dict[lane_group]]["ql_cells"][gpt_lane_cell] += 1

        mean_speed = np.mean(outgoing_lane_speeds) if len(outgoing_lane_speeds) > 0 else 0.0

        return statistic_state, statistic_state_incoming, mean_speed

    def update_history_data(self, states, next_states, current_outputs, next_outputs, action_list):
        self.history_data['release_car_num'].append(0)
        self.history_data['input_car_num'].append(0)
        # self.history_data['car_num'].append(0)
        self.history_data['car_num_out_of_lane'].append(0)
        self.history_data["avg_ql_cells"].append([])
        self.history_data['boundary']['release_num'].append([])
        self.history_data['boundary']['input_num'].append([])
        self.history_data['boundary']['r_i_dif'].append([])
        self.history_data["car_num_outside"]['waiting'].append(0)
        self.history_data["car_num_outside"]['running'].append(0)
        self.history_data["car_num_outside"]['total'].append(0)
        self.history_data["veh_num"].append(0)
        self.history_data["ql_num"].append(0)
        self.release_data = []
        # self.history_data['veh_log'] = {}
        # self.history_data['veh_log']['inside_veh'] = []
        # self.history_data['veh_log']['outside_veh'] = set()
        # self.history_data['veh_log']['all_veh'] = set()
        self.history_data['veh_log']['outside_list'] = []
        
        
        for i, s in enumerate(states):
            p = code2action(action_list[i])
            lane1 = p[:2]
            lane2 = p[2:]
            self.updata_history_data_release_range(lane1, s, next_states[i])
            self.updata_history_data_release_range(lane2, s, next_states[i])
            ql_cells = []
            self.release_data.append(0) 
            for lane in location_all_direction_dict:
                self.updata_history_data_per_lane(lane, s, next_states[i], ql_cells)

            self.history_data["avg_ql_cells"][-1].append(np.mean(ql_cells))
            if i in self.boundary_intersections['id_list']:
                release_num, input_num, car_num_out_side = self.count_boundary_data(i, states[i], next_states[i], current_outputs[i], next_outputs[i])
                self.history_data['boundary']['release_num'][-1].append(release_num)
                self.history_data['boundary']['input_num'][-1].append(input_num)
                self.history_data['boundary']['r_i_dif'][-1].append(release_num - input_num)
                self.history_data["car_num_outside"]['waiting'][-1] += car_num_out_side['waiting']
                self.history_data["car_num_outside"]['running'][-1] += car_num_out_side['running']
                self.history_data["car_num_outside"]['total'][-1] += car_num_out_side['total']
        # self.history_data['boundary']['r_i_dif'].append(list(np.array(self.history_data['boundary']['release_num'][-1]) - np.array(self.history_data['boundary']['input_num'][-1])))
        self.refine_history_data_release_range()
        for inter in self.env.list_intersection:
            self.history_data["veh_num"][-1] += sum(inter.dic_feature['lane_num_vehicle'])
            self.history_data["ql_num"][-1] += sum(inter.dic_feature['lane_num_waiting_vehicle_in'])
        self.history_data["car_num_inside"]['total'].append(self.history_data["veh_num"][-1] - self.history_data["car_num_outside"]['total'][-1])
        self.history_data["car_num_inside"]['waiting'].append(self.history_data["ql_num"][-1] - self.history_data["car_num_outside"]['waiting'][-1])
        self.history_data["car_num_inside"]['running'].append(self.history_data["car_num_inside"]['total'][-1] - self.history_data["car_num_inside"]['waiting'][-1])
        # self.history_data['veh_log']['all_veh/outside_veh'] = self.history_data['veh_log']['all_veh'] - self.history_data['veh_log']['outside_veh']
        self.history_data["car_num_inside"]['r_i'].append(len(self.history_data['veh_log']['inside_list']))

    def count_boundary_data(self, i, state, next_state, current_output, next_output):
        release_num = 0
        input_num = 0
        car_num_out_side = {}
        car_num_out_side['waiting'] = 0
        car_num_out_side['running'] = 0
        car_num_out_side['total'] = 0

        neighbor_list = self.get_neighbor_list(i)
        assert len(neighbor_list) < 4
        four_location = ['North', 'South', 'West', 'East']
        location_list = [neighbor['location'] for neighbor in neighbor_list]
        outside_location_list = [loc for loc in four_location if loc not in location_list]
        for loc in outside_location_list:
            current_output_veh_list = list(current_output[loc[0]]['veh2cell'].keys())
            next_output_veh_list = list(next_output[loc[0]]['veh2cell'].keys())
            release_vehs = [veh for veh in next_output_veh_list if veh not in current_output_veh_list]
            # release_num += len(release_vehs)
            for veh in release_vehs:
                if veh in self.history_data['veh_log']['inside_list']:
                    self.history_data['veh_log']['inside_list'].remove(veh)
                    release_num += 1
            # self.history_data['veh_log']['inside_veh'] = self.history_data['veh_log']['inside_veh'] - set(release_vehs)
            assert current_output[loc[0]]['queue_len'] == sum(current_output[loc[0]]['ql_cells'])

            # car_num_out_side['waiting'] += next_output[loc[0]]['queue_len'] 
            # car_num_out_side['running'] += sum(next_output[loc[0]]['cells'])
            # car_num_out_side['total'] += len(next_output_veh_list)
            
            current_input_veh_list = []
            next_input_veh_list = []
            
            for direc in ['L','T','R']:
                current_input_veh_list.extend(list(state[loc[0] + direc]['veh2cell'].keys()))
                next_input_veh_list.extend(list(next_state[loc[0] + direc]['veh2cell'].keys()))

                car_num_out_side['waiting'] += next_state[loc[0] + direc]['queue_len']
                car_num_out_side['running'] += sum(next_state[loc[0] + direc]['cells'])

            car_num_out_side['total'] += len(next_input_veh_list)

            input_vehs = [veh for veh in current_input_veh_list if veh not in next_input_veh_list]
            input_num += len(input_vehs)

            # self.history_data['veh_log']['inside_veh'].update(set(input_vehs))
            self.history_data['veh_log']['inside_list'].extend(set(input_vehs))
            # self.history_data['veh_log']['outside_veh'].update(set(next_output_veh_list + next_input_veh_list))
            self.history_data['veh_log']['outside_list'].extend(next_output_veh_list + next_input_veh_list)

        return release_num, input_num, car_num_out_side

    def updata_history_data_per_lane(self, lane, state, next_state, ql_cells):
        lane_vehs = state[lane]["veh2cell"]
        lane_vehs_next = next_state[lane]["veh2cell"]
        lane_vehs_list = list(lane_vehs.keys())
        lane_vehs_keys_next = list(lane_vehs_next.keys())
        depart_vehs = []
        stay_vehs = []
        for veh in lane_vehs_list:
            if veh in lane_vehs_keys_next:
                stay_vehs.append(veh)
            else:
                depart_vehs.append(veh)

        self.history_data['car_num_out_of_lane'][-1] += next_state[lane]["out_of_lane"]
        self.history_data['release_car_num'][-1] += len(depart_vehs)
        self.release_data[-1] += len(depart_vehs)
        self.history_data['input_car_num'][-1] += len(lane_vehs_keys_next) - len(stay_vehs)
        ql_cells.append(np.count_nonzero(next_state[lane]['ql_cells']))

    def refine_history_data_release_range(self):
        for lane in self.history_data["release_range"]:
            self.history_data["release_range"][lane] = fix_decreasing_list(self.history_data["release_range"][lane])
            
    def updata_history_data_release_range(self, lane, state, next_state):
        ql_num = state[lane]['queue_len']
        end_cell = self.identify_last_cell(state[lane]["veh2cell"], next_state[lane]["veh2cell"])
        


        if end_cell:
            if ql_num in self.history_data["release_range"][lane]:
                if end_cell > self.history_data["release_range"][lane][ql_num]:
                    self.history_data["release_range"][lane][ql_num] = end_cell
            else:
                self.history_data["release_range"][lane][ql_num] = end_cell
            if state[lane]['queue_len'] not in self.history_data['effective_release_log']:
                self.history_data['effective_release_log'][state[lane]['queue_len']] = [state[lane], end_cell]
            elif end_cell > self.history_data['effective_release_log'][state[lane]['queue_len']][1]:
                self.history_data['effective_release_log'][state[lane]['queue_len']] = [state[lane], end_cell]

    def identify_last_cell(self, lane_vehs, lane_vehs_next):
        lane_vehs_list = list(lane_vehs.keys())
        lane_vehs_keys_next = list(lane_vehs_next.keys())
        depart_vehs = []
        stay_vehs = []
        for veh in lane_vehs_list:
            if veh in lane_vehs_keys_next:
                stay_vehs.append(veh)
            else:
                try:
                    vehicle_info = self.env.eng.get_vehicle_info(veh)
                    if vehicle_info:
                        depart_vehs.append(veh)
                except Exception:
                    pass
        if stay_vehs:
            min_veh_stay = min(stay_vehs, key=lambda x: lane_vehs_next[x])
            upper_bound = lane_vehs[min_veh_stay] - lane_vehs_next[min_veh_stay]
        else:
            upper_bound = None

        if depart_vehs:
            max_veh_depart = max(depart_vehs, key=lambda x: lane_vehs[x])
            lower_bound = lane_vehs[max_veh_depart]
        else:
            lower_bound = None
        last_cell = lower_bound

        if upper_bound:
            if not last_cell:
                last_cell = upper_bound - 1
            elif last_cell > upper_bound:
                last_cell = upper_bound - 1
        
        return last_cell

    def estimate_effective_range(self, state):
        range_list = []
        for lane in location_direction_dict:
            ql_num = state[lane]['queue_len']
            exit_qls = list(self.history_data["release_range"][lane].keys())
            if ql_num in self.history_data["release_range"][lane]:
                range_list.append(self.history_data["release_range"][lane][ql_num])
            elif len(exit_qls):
                # exit_qls = list(self.history_data["release_range"][lane].keys())
                exit_qls = np.array(exit_qls)
                closest_ql = exit_qls[np.argmin(np.abs(exit_qls - ql_num))]
                if np.abs(closest_ql) < 5 or ql_num < closest_ql:
                    range_list.append(self.history_data["release_range"][lane][closest_ql])
                elif self.history_data["release_range"][lane][closest_ql] < self.default_release_range:
                    range_list.append(self.history_data["release_range"][lane][closest_ql])
                else:
                    range_list.append(self.default_release_range)
            else:
                range_list.append(self.default_release_range)
        return range_list

    def get_neighbor_list(self, inter_id):
        n_list = []
        inter_name = self.env.list_intersection[inter_id].inter_name
        inter_list = list(self.env.intersection_dict.keys())
        intersection = self.env.intersection_dict[inter_name]
        roads = deepcopy(intersection["roads"])

        neighbor_list = [inter for inter in self.env.traffic_light_node_dict[inter_name]['neighbor_ENWS'] if inter] #inter_name
        road_list = list(roads.keys())
        road2inter = [r.replace("road", "intersection")[:-2] for r in road_list] 
        neighbor2loc = {inter: roads[road_list[i]]['location'] for i, inter in enumerate(road2inter) if inter in neighbor_list}
        for neighbor_inter_name in neighbor_list:
            n_list.append({"id":inter_list.index(neighbor_inter_name), "name":neighbor_inter_name,  "location":neighbor2loc[neighbor_inter_name]})
        return n_list

    def train_test(self):
        all_config = merge(merge(self.dic_agent_conf, self.dic_path), self.dic_traffic_env_conf)
        logger = wandb.init(
            project=self.dic_traffic_env_conf['PROJECT_NAME'],
            group=f"{self.dic_traffic_env_conf['MODEL_NAME']}-{self.roadnet}-{self.trafficflow}-{len(self.dic_traffic_env_conf['PHASE'])}_Phases",
            name=f"{self.dic_traffic_env_conf['TRAFFIC_FILE'].replace('.json', '')}",
            config=all_config,
        )

        self.test(logger, 0)
        wandb.finish()

    def get_norm_reward(self, state):
        rewards = []

        for i in range(len(state)):
            vehicle_num = 0
            queue_length = 0

            intersection = self.env.intersection_dict[self.env.list_intersection[i].inter_name]
            roads = deepcopy(intersection["roads"])
            statistic_state, _, _ = get_state_detail(roads, self.env)
            for lane in statistic_state:
                queue_length += statistic_state[lane]['queue_len']

                vehicle_num += statistic_state[lane]['queue_len']
                for cell in range(len(statistic_state[lane]['cells'])):
                    vehicle_num += statistic_state[lane]['cells'][cell]

            reward = -(queue_length / vehicle_num) if vehicle_num > 0.0 else -0.0
            rewards.append(reward)

        return rewards
    