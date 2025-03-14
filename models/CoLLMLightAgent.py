from utils.my_utils import fix_decreasing_list, location_dict_detail, load_json, dump_json, get_state_detail, state2feature, getPrompt, action2code, code2action, eight_phase_list, four_phase_list, get_state_detail_all_lane, get_state_detail_many_seg_all_lane, location_direction_dict, location_all_direction_dict, location2releaselane, location_dict
from utils.utils import *
import numpy as np
from copy import deepcopy
from collections import defaultdict

class IntersectionAgent(object):
    def __init__(self, agent_conf, traffic_env_conf, LLM):
        ## indentification info
        self.agent_conf = agent_conf
        self.name = agent_conf['inter_name']
        self.id = agent_conf['inter_id']
        self.boundary = agent_conf['boundary']
        self.long_info = agent_conf['long_info']
        self.feed_back = agent_conf['feed_back']
        self.feed_back_num = agent_conf['feed_back_num']
        ## env info
        self.net_name = traffic_env_conf['name']
        self.size = traffic_env_conf['size']
        self.signal_list = traffic_env_conf['signal_list']
        self.lane_list = traffic_env_conf['lane_list'] #only T and L
        self.lane_list_onlyTL = location_direction_dict
        self.neighbor_list = None  # need be load
        self.signal_time = traffic_env_conf['signal_time']
        self.llm_engine = LLM
        self.area_inter_num = self.size[0] * self.size[1]
        self.intersection_list = traffic_env_conf['intersection_list']
        self.think_mode = agent_conf['think_mode']
        self.h_w_size = agent_conf['h_w_size']
        
        

        ## inter_history_data
        self.log_init()
        self.area_rank = {}
        self.area_rank['queue_num'] = []
        self.area_rank['wait_time'] = []

        # memories
        self.step_num = None
        self.memories = {}
        for lane in self.lane_list_onlyTL:
            self.memories[lane] = []
        self.release_veh2memorie_index = {}
        self.recent_memory = None
    
        self.careful_signals = {}
        for signal in self.signal_list:
            self.careful_signals[signal] = {}
            self.careful_signals[signal]['state'] = False
            self.careful_signals[signal]['event'] = None
        
        ## hyper param
        self.accumulate_patience = traffic_env_conf['accumulate_patience']
        self.congestion_scale_ratio = traffic_env_conf['congestion_scale_ratio']
        self.boundary_distance_threshold = traffic_env_conf['boundary_distance_threshold']

        ## other
        self.current_state = None
        self.last_queue = 0
        # self.queue_threshold
        self.accumulate_times = 0
        self.congestion_state = False
        self.congestion_queue = 0
        self.connection_threshold = 0.05
        self.traffic_input_memory = {}
        self.congest_degree = []
        self.np_empty_lanes = []
        for lane in location_direction_dict:
            self.traffic_input_memory[lane] = {}
            self.traffic_input_memory[lane]['from_count'] = {}
            self.traffic_input_memory[lane]['to_count'] = {}
            self.traffic_input_memory[lane]['upstream_lanes'] = []
            self.traffic_input_memory[lane]['downstream_lanes'] = []
        
    
    def log_init(self):
        self.log = {}
        self.log['queue_diff'] = []
        self.log['signal'] = []
        
        self.log['states'] = []
        self.log['wait_time'] = []
        self.log['queue_num'] = {}
        self.log['release_num'] = {}
        self.log['entry_num'] = {}

        for key in ['East', 'West', 'South', 'North', 'Total']: # T: total
            self.log['queue_num'][key] = []
            self.log['release_num'][key] = []
            self.log['entry_num'][key] = []
        self.traffic_state_log = []
    

        
    def update_state(self, state, last_action_id = None, update_memory = False):
        ## queue, queue_diff, state
        queue_num = 0
        wait_time = 0
        current_state = {}
        congest_degree = {}
        
        self.state = state
        self.no_empty_lanes = []
        self.empty_lanes = []
        for lane in self.lane_list:
            ql_cells = state[lane]['ql_cells']
            ql_cell_num = find_last_non_zero_index(ql_cells)
            # if lane in location_direction_dict:
            congest_degree[lane] = round(ql_cell_num / len(ql_cells), 2)
            queue_num += state[lane]['queue_len']
            queue_car_num = state[lane]['queue_len']
            coming_car_num = sum(state[lane]['cells'])
            current_state[lane] = {}
            current_state[lane]['occupancy'] = state[lane]['occupancy']
            if current_state[lane]['occupancy'] > 0 and lane in location_direction_dict:
                self.no_empty_lanes.append(lane)
            elif current_state[lane]['occupancy'] == 0 and lane in location_direction_dict:
                self.empty_lanes.append(lane)
            current_state[lane]['queue_car_num'] = queue_car_num
            current_state[lane]['coming_car_num'] = coming_car_num
            current_state[lane]['wait_time'] = queue_car_num * state[lane]['avg_wait_time']
            current_state[lane]['avg_wait_time'] = state[lane]['avg_wait_time']
            wait_time += current_state[lane]['wait_time']
            current_state[lane]['veh2cell'] = state[lane]['veh2cell']
            if lane[0] not in current_state:
                current_state[lane[0]] = {}
                self.log['queue_num'][location_dict[lane[0]]].append(queue_car_num)
                current_state[lane[0]]['queue_car_num'] = queue_car_num
                current_state[lane[0]]['coming_car_num'] = coming_car_num
                current_state[lane[0]]['wait_time'] = current_state[lane]['wait_time']
            else:
                self.log['queue_num'][location_dict[lane[0]]][-1] += queue_car_num
                current_state[lane[0]]['queue_car_num'] += queue_car_num
                current_state[lane[0]]['coming_car_num'] += coming_car_num
                current_state[lane[0]]['wait_time'] += current_state[lane]['wait_time']
            
        self.update_traffic_data(state)
         ## this states include traffic state and its changes, designed for traffic world model training
        if update_memory:
            self.update_memory(last_action_id)
            self.update_current_traffic_states(current_state, last_action_id)
            
        self.log['wait_time'].append(wait_time)
        self.log['states'].append(current_state)
        self.current_state = current_state
        self.log['queue_num']['Total'].append(queue_num)
        queue_diff = queue_num - self.last_queue
        self.log['queue_diff'].append(queue_diff)
        self.last_queue = queue_num
        self.congest_degree.append(congest_degree)



        ## check if switch congestion state
        if queue_diff > 0:
            self.accumulate_times += 1
            if self.accumulate_times >= self.accumulate_patience:
                self.congestion_state = True
                self.congestion_queue = queue_num
        else:
            self.accumulate_times = 0
            if self.congestion_state == True:
                if queue_num <= self.congestion_scale_ratio * self.congestion_queue:
                    self.congestion_state = False
    
    def update_current_traffic_states(self, state, last_action_id):
        last_state = self.log['states'][-1] if len(self.log['states']) else None
        self.current_traffic_states = {}
        self.current_traffic_states['Signal'] = self.signal_list[last_action_id]
        for lane in self.lane_list_onlyTL:
            self.current_traffic_states[lane] = {}
            cars_before = list(last_state[lane]['veh2cell'].keys()) if last_state else []
            cars_current = list(state[lane]['veh2cell'].keys())
            cars_output = [veh for veh in cars_before if veh not in cars_current]
            cars_input = [veh for veh in cars_current if veh not in cars_before]
            self.current_traffic_states[lane]['Cars Input'] = len(cars_input)
            self.current_traffic_states[lane]['Cars Output'] = len(cars_output)
            queue_before = last_state[lane]['queue_car_num'] if last_state else 0
            queue_current = state[lane]['queue_car_num']
            queue_diff = queue_current - queue_before
            self.current_traffic_states[lane]['Queued Cars Change'] = queue_diff
            self.current_traffic_states[lane]['Queued Cars'] = queue_current
            moving_before = last_state[lane]['coming_car_num'] if last_state else 0
            moving_current = state[lane]['coming_car_num']
            moving_diff = moving_current - moving_before
            self.current_traffic_states[lane]['Moving Cars Change'] = moving_diff
            self.current_traffic_states[lane]['Moving Cars'] = moving_current
            avg_wait_time_before = last_state[lane]['avg_wait_time'] if last_state else 0
            avg_wait_time_before = round(avg_wait_time_before/60, 2)
            avg_wait_time_current = state[lane]['avg_wait_time']
            avg_wait_time_current = round(avg_wait_time_current/60, 2)
            avg_wait_time_diff = avg_wait_time_current - avg_wait_time_before
            self.current_traffic_states[lane]['Average Waiting Time Change (mins)'] = round(avg_wait_time_diff,2)
            self.current_traffic_states[lane]['Average Waiting Time (mins)'] = avg_wait_time_current
            occupancy_before = last_state[lane]['occupancy'] if last_state else 0
            occupancy_before = round(occupancy_before*100, 2)
            occupancy_current = state[lane]['occupancy']
            occupancy_current = round(occupancy_current*100, 2)
            occupancy_diff = occupancy_current - occupancy_before
            self.current_traffic_states[lane]['Occupancy Change (%)'] = round(occupancy_diff,2)
            self.current_traffic_states[lane]['Occupancy (%)'] = occupancy_current

    def update_traffic_state_updown_stream(self, traffic_state_updown_stream):
        self.current_traffic_states.update(deepcopy(traffic_state_updown_stream))
        summary = {}
        summary['Total Cars Output'] = 0
        summary['Over Occupancy'] = False
        summary['Average Queued Length'] = 0
        summary['Max Average Waiting Time'] = 0
        ct = 0
        for key in list(self.current_traffic_states.keys()):
            if key == 'Signal':
                continue
            summary['Total Cars Output'] += self.current_traffic_states[key]['Cars Output']
            if self.current_traffic_states[key]['Occupancy (%)'] >= 100:
                summary['Over Occupancy'] = True 
            summary['Average Queued Length'] += self.current_traffic_states[key]['Queued Cars']
            ct += 1
            summary['Max Average Waiting Time'] = max(summary['Max Average Waiting Time'], self.current_traffic_states[key]['Average Waiting Time (mins)'])
        summary['Average Queued Length'] = round(summary['Average Queued Length']/ct, 2)
        self.current_traffic_states['Summary'] = summary

        self.traffic_state_log.append(self.current_traffic_states)  
    def update_memory(self, signal_id):
        signal_text = self.signal_list[signal_id]
        last_state = self.log['states'][-1] # this is the previous state, not current state

        recent_memory = {}
        recent_memory['signal'] = signal_text
        recent_memory['pos'] = []
        release_lanes = [signal_text[:2], signal_text[2:]]

        for lane in release_lanes:
            lane_memory = {}
            lane_memory['lane'] = lane
            lane_memory['occupancy_before'] = last_state[lane]['occupancy']
            if lane_memory['occupancy_before'] == 0:
                continue
            lane_memory['queue_num_before'] = last_state[lane]['queue_car_num']
            lane_memory['moving_num_before'] = last_state[lane]['coming_car_num']
            # lane_memory['downstream_occupancy_before'] = None
            # if lane in self.up_down_stream_view:
            #     if 'downstream' in self.up_down_stream_view[lane]:
            #         lane_memory['downstream_occupancy_before'] = {}
            #         for d_lane in self.up_down_stream_view[lane]['downstream']:
            #             inter_id = self.up_down_stream_view[lane]['downstream'][d_lane]['inter_id']
            #             lane_memory['downstream_occupancy_before'][(inter_id, d_lane)] = self.up_down_stream_view[lane]['downstream'][d_lane]['occupancy']
            lane_memory['release_results'] = {}
            lane_memory['let_downstream_over100'] = False
            for veh in self.veh_release_data[lane]: # this data update in the update_traffic_data 
                lane_memory['release_results'][veh] = None
                self.release_veh2memorie_index[veh] = (lane, len(self.memories[lane])) # latest release by this lane, it will update every time update memory
            recent_memory['pos'].append((lane, len(self.memories[lane])))
            self.memories[lane].append(lane_memory)
            
        self.recent_memory = deepcopy(recent_memory)




    def update_traffic_data(self, state):
        # signal
        # vehs
        self.veh_release_data = {}
        self.veh_input_data = {}
        for key in ['East', 'West', 'South', 'North', 'Total']:
            self.log['release_num'][key].append(0)
            self.log['entry_num'][key].append(0)

        for lane in self.lane_list:
            if len(self.log['states']):
                lane_vehs = self.log['states'][-1][lane]["veh2cell"]
            else:
                lane_vehs = dict()
            lane_vehs_next = state[lane]["veh2cell"]

            lane_vehs_list = list(lane_vehs.keys())
            lane_vehs_keys_next = list(lane_vehs_next.keys())

            depart_vehs = []
            stay_vehs = []
            
            for veh in lane_vehs_list:
                if veh in lane_vehs_keys_next:
                    stay_vehs.append(veh)
                else:
                    depart_vehs.append(veh)
            new_vehs = [veh for veh in lane_vehs_keys_next if veh not in stay_vehs]
            if 'R' not in lane:
                self.veh_release_data[lane] = depart_vehs
                self.veh_input_data[lane] = new_vehs


            self.log['release_num'][location_dict[lane[0]]][-1] += len(depart_vehs)
            self.log['entry_num'][location_dict[lane[0]]][-1] += len(lane_vehs_keys_next) - len(stay_vehs)
            self.log['release_num']['Total'][-1] += len(depart_vehs)
            self.log['entry_num']['Total'][-1] += len(lane_vehs_keys_next) - len(stay_vehs)
        
    def update_up_down_stream_view(self, up_down_stream_view):
        self.up_down_stream_view = deepcopy(up_down_stream_view)
    def update_up_down_stream_view_spatial(self, up_down_stream_view):
        self.up_down_stream_view2 = deepcopy(up_down_stream_view)

    def update_long_distance_info(self, long_distance_info):
        self.long_distance_info = deepcopy(long_distance_info)
    

    def traffic_memory_update(self, lane2upstream_data, lane2downstream_data):
        '''
            a. from_count_dict
            b. to_count_dict
            c. upstream_lanes: []
            d. downstream_lanes:
            e. congest_data {(2, 'NL'): {upstream: [(3,'NT')], downstream: [(4,'ST')], congestion_degree:60%}
            self.traffic_input_memory[lane] = {}
            self.traffic_input_memory[lane]['from_count'] = {}
            self.traffic_input_memory[lane]['to_count'] = {}
            self.traffic_input_memory[lane]['upstream_lanes'] = []
            self.traffic_input_memory[lane]['downstream_lanes'] = []
        '''
        for lane in location_direction_dict:
            key = (self.id, lane)
            if key in lane2upstream_data:
                upstream_lanes_ct = lane2upstream_data[key]
                for up_lane in upstream_lanes_ct:
                    if up_lane not in self.traffic_input_memory[lane]['from_count']:
                        self.traffic_input_memory[lane]['from_count'][up_lane] = 0
                    self.traffic_input_memory[lane]['from_count'][up_lane] += upstream_lanes_ct[up_lane]
            if key in lane2downstream_data:    
                downstream_lanes_ct = lane2downstream_data[key]
                for down_lane in downstream_lanes_ct:
                    if down_lane not in self.traffic_input_memory[lane]['to_count']:
                        self.traffic_input_memory[lane]['to_count'][down_lane] = 0
                    self.traffic_input_memory[lane]['to_count'][down_lane] += downstream_lanes_ct[down_lane]
                    

            upstream_candis = list(self.traffic_input_memory[lane]['from_count'].keys())
            upstream_prob = list(self.traffic_input_memory[lane]['from_count'].values())
            upstream_prob = np.array(upstream_prob) /sum(upstream_prob)
            positions = np.where(upstream_prob > self.connection_threshold)[0]
            positions = positions.tolist()
            self.traffic_input_memory[lane]['upstream_lanes'] = [upstream_candis[i] for i in positions]

            downstream_candis = list(self.traffic_input_memory[lane]['to_count'].keys())
            downstream_prob = list(self.traffic_input_memory[lane]['to_count'].values())
            downstream_prob = np.array(downstream_prob) /sum(downstream_prob)
            positions = np.where(downstream_prob > self.connection_threshold)[0]
            positions = positions.tolist()
            self.traffic_input_memory[lane]['downstream_lanes'] = [downstream_candis[i] for i in positions]     
            
        # congestion_dict: {(2, 'NL'): {upstream: [(3,'NT')], downstream: [(4,'ST')], congestion_degree:60%}

    def select_signal_default(self, effective_range_list):
        lane_release_metrix = {}
        for i, lane in enumerate(location_direction_dict):
            lane_range = effective_range_list[i]
            going_cars_num = np.sum(self.state[lane]["cells"][:lane_range+1])
            stop_cars_num = np.sum(self.state[lane]["ql_cells"][:lane_range+1])
            lane_release_metrix[lane] = stop_cars_num * self.state[lane]["avg_wait_time"] + stop_cars_num * self.signal_time + going_cars_num * self.signal_time 
        phase_release_metrix = []
        for p in self.signal_list:
            phase_release_metrix.append(lane_release_metrix[p[:2]] + lane_release_metrix[p[2:]])
        index = phase_release_metrix.index(max(phase_release_metrix))
        signal_text = self.signal_list[index]
        return signal_text
    
    def get_long_distance_info_text(self):
        long_distance_exist = False 
        prompt = "#### **Long-distance upstream and downstream information:** \n"
        prompt += "|Relation|The number of lanes whose occupancy exceeds half|Queued Cars|Average Waiting Time (mins)|Average Occupancy|\n"
        for lane in self.no_empty_lanes:
            if len(self.long_distance_info[lane]['exist']) > 0:
                for direc in self.long_distance_info[lane]['exist']:
                    long_distance_exist = True
                    prompt += "|{}' {}|{}|{}|{:.1f}|{:.1f}%| \n".format(lane, direc, self.long_distance_info[lane][direc]['lane_num'], self.long_distance_info[lane][direc]['total_queue_num'], self.long_distance_info[lane][direc]['average_waiting_time']/60, self.long_distance_info[lane][direc]['average_occupancy']*100)
        if not long_distance_exist:
            return ""
        
        return prompt
    
    def get_memory_text(self):
        prompt = "#### **Past Lane Activation Data (Memory):**\n"
        prompt += "There were some lane activation cases and results in the past. Based on these data, you can better estimate the influence of each signal in the current situation.\n"
        ## recent_memory
        memory_exist = False
        recent_memory = False
        if self.recent_memory:
            recent_memory_txt = "In the most recent period, you selected signal {}. Here are the results:\n".format(self.recent_memory['signal'])
            if len(self.recent_memory['pos']) > 0:
                recent_memory = True
                for lane, memory_idx in self.recent_memory['pos']:
                    lane_memory = self.memories[lane][memory_idx]
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
        similar_lane_memory = []
        for lane in self.no_empty_lanes:
            lane_occupancy = self.current_state[lane]['occupancy']
            min_similar = 0.5
            similar_idx = None
            if len(self.memories[lane])>1:
                for i in range(len(self.memories[lane])-1):
                    similar = abs(self.memories[lane][i]['occupancy_before'] - lane_occupancy)
                    if similar < min_similar:
                        min_similar = similar
                        similar_idx = i
                if similar_idx:
                    similar_lane_memory.append(self.memories[lane][similar_idx])
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


    def get_signal_rank_text(self, effective_range_list):
        ## local strategy
        lane_release_metrix = {}
        for i, lane in enumerate(location_direction_dict):
            lane_range = effective_range_list[i]
            going_cars_num = np.sum(self.state[lane]["cells"][:lane_range+1])
            stop_cars_num = np.sum(self.state[lane]["ql_cells"][:lane_range+1])
            lane_release_metrix[lane] = stop_cars_num * self.state[lane]["avg_wait_time"] + stop_cars_num * self.signal_time + going_cars_num * self.signal_time 
        signal_value_dict = {}
        for p in self.signal_list:
            signal_value_dict[p] = lane_release_metrix[p[:2]] + lane_release_metrix[p[2:]]

        signal_rank_text = "### Signal Priority (local)\n"
        signal_rank_text += "This rank only consider your own intersection, and assume the downstream allow the release.\n"
        signal_rank_text += "|Rank|Signal|Waiting Time Reduction|\n"
        for i, p in enumerate(sorted(signal_value_dict, key=signal_value_dict.get, reverse=True)):
            signal_rank_text += "|{}|{}|{:.1f} mins|\n".format(i+1, p, signal_value_dict[p]/60)
        signal_rank_text += '\n'
        return signal_rank_text

    def get_back_ground_and_note_text(self):
        input = "## Background Context\n"
        input += "An intersection has 12 lanes: [NL, NT, NR, SL, ST, SR, EL, ET, ER, WL, WT, WR]. Each lane is labeled by direction and movement: N for north, S for south, E for east, W for west, L for left turn, T for through, and R for right turn. For instance, ET stands for the East Through lane, where traffic moves straight ahead from east to west. WL is the West Left-turn lane, where traffic turns left from west to south. Right turns are always allowed. There are four signal options: [ETWT, NTST, ELWL, NLSL]. For example, ETWT indicates the release of both the ET and WT lanes. The signal phase duration is set to thirty seconds.\n\n"
        input += "## Note:\n"
        input += """For each Lane X, when considering activating it, keep these in mind: 
    - NEVER let the occupancy of X's downstream lanes be close to 100% at any risk, as it will cause severe congestion.
    - If the upstream or downstream information of X isn't mentioned, it means that they are in a good state with low occupancy.
    - You MUST consider how much the occupancy of X's downstream lanes will increase upon releasing lane X. 
    - You MUST delay the release of X if its downstream has a high occupancy rate.
    - If there are many high-occupancy lanes upstream of X and X's occupancy is not low, you MUST consider releasing X so as to help upstream lanes release.
    - You can't keep a lane waiting for too long. You MUST release the lane with excessive waiting time when the downstream condition allows.\n\n"""
        return input
    
    def get_historical_observation(self):
        timestep = len(self.log['states'])-1
        
        history_state = self.traffic_state_log[-self.h_w_size:]
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
        if self.h_w_size > 0:
            return text
        else:
            return ""

    
    def get_current_observation_text(self):
        observation_of_this_inter_text = self.get_current_lane_observation_text_tablestyle()
        
        up_down_stream_view_text = self.get_up_down_stream_view_text()
        
        # long distance
        # long_distance_info_text = self.get_long_distance_info_text()
        text = observation_of_this_inter_text
        text += '{} are empty.\n'.format(self.empty_lanes)
        text += '\n'
        text += up_down_stream_view_text
        text += '\n'
        # text += long_distance_info_text
        # text += '\n' 
        return text

    def multi_level_decision_making_prompt(self, effective_range_list):
        timestep = len(self.log['states'])-1
        input_text = self.get_back_ground_and_note_text()
        input_text += "## Data\n"
        historical_observation = self.get_historical_observation()
        input_text += historical_observation
        input_text += "### Current Observation\n"
        input_text += f"Timestep: {timestep}\n"
        current_observation_text = self.get_current_observation_text()
        signal_rank_text = self.get_signal_rank_text(effective_range_list)
        input_text += current_observation_text
        input_text += '\n'
        input_text += signal_rank_text
        if self.think_mode == 'quick':
            input_text += """## Task:\nYou MUST select the highest-ranked priority signal\n\n## Requirement:\nPlease provide a JSON formatted output as follows: \n```json  \n{  \n   "answer": "ETWT" | "NTST" | "ELWL" | "NLSL"  \n}\n``` """
        elif self.think_mode == 'mild':
            input_text += """## Task:\n- Focus on the "Current Observation" data.\n- Keep the "Notes" in mind, compare each signal, and select the optimal one based on traffic conditions.\n\n## Requirement:\nPlease provide a JSON formatted output as follows: \n```json  \n{  \n   "thought_process": "Your thought process text description",  \n   "answer": "ETWT" | "NTST" | "ELWL" | "NLSL"  \n}\n```  """
        elif self.think_mode == 'slow':
            input_text += """## Task:\nYou MUST conduct a comprehensive analysis and select the optimal signal by following these steps:\n1. **Current Traffic Data Analysis**: Rank lanes based on queued cars and waiting times.\n2. **Upstream Analysis**: Assess the conditions of nearby and long-distance upstream lanes for each lane.\n3. **Downstream Analysis**: Assess the conditions of nearby and long-distance downstream lanes for each lane.\n4. **Signal Consequence Prediction**: Utilize "Historical Observation" data for inferring the consequences of each signal. \n5. **Signal Comparison**: Systematically compare the effects of each signal option.\n6. **Decision Making**: Make a decision on the optimal signal to activate.\n\n## Requirement:\nPlease provide a JSON formatted output as follows: \n```json  \n{  \n    "thought_process": "1. **Current Traffic Data Analysis**: xxxxx",  \n    "answer": "ETWT" | "NTST" | "ELWL" | "NLSL"  \n}\n``` """
        elif self.think_mode == 'qm':
            input_text += """## Task:\nYour task consists of two phases:\n### Phase 1: Coordination Scenario Classification\n1. Analyze the data in "Current Observation" and classify the intersection into one of two coordination scenarios:\n   - **No-Coordination**: No upstream or downstream coordination needs.\n   - **Simple-Coordination**: Few upstream or downstream lanes require coordination with low traffic complexity.\n\n### Phase 2: Signal Selection Strategy\nBased on the identified coordination scenario, select the appropriate signal using the following strategies:\n\n#### a. No-Coordination\n- Directly select the highest-ranked priority signal, as there are no external coordination considerations.\n\n#### b. Simple-Coordination\n- Focus on the "Current Observation" data.\n- Keep the "Notes" in mind, compare each signal, and select the optimal one based on traffic conditions.\n\n## Requirement:\nPlease provide a JSON formatted output as follows: \n```json  \n{  \n  "phase1": {  \n    "thought_process": "Your thought process text description",  \n    "answer": "NO" | "Simple"  \n  },  \n  "phase2": {  \n    "thought_process": "Your thought process text description",  \n    "answer": "ETWT" | "NTST" | "ELWL" | "NLSL"  \n  }  \n}\n``` """
        elif self.think_mode == 'qs':
            input_text += """## Task:\nYour task consists of two phases:\n### Phase 1: Coordination Scenario Classification\n1. Analyze the data in "Current Observation" and classify the intersection into one of two coordination scenarios:\n   - **No-Coordination**: No upstream or downstream coordination needs.\n   - **Complex-Coordination**: High traffic complexity with significant congestion in upstream or downstream lanes.\n\n### Phase 2: Signal Selection Strategy\nBased on the identified coordination scenario, select the appropriate signal using the following strategies:\n\n#### a. No-Coordination\n- Directly select the highest-ranked priority signal, as there are no external coordination considerations.\n\n#### b. Complex-Coordination\nConduct a comprehensive analysis and select the optimal signal by following these steps:\n1. **Current Traffic Data Analysis**: Rank lanes based on queued cars and waiting times.\n2. **Upstream Analysis**: Assess the conditions of nearby and long-distance upstream lanes for each lane.\n3. **Downstream Analysis**: Assess the conditions of nearby and long-distance downstream lanes for each lane.\n4. **Signal Consequence Prediction**: Utilize "Historical Observation" data for inferring the consequences of each signal. \n5. **Signal Comparison**: Systematically compare the effects of each signal option.\n6. **Decision Making**: Make a decision on the optimal signal to activate.\n\n## Requirement:\nPlease provide a JSON formatted output as follows: \n```json  \n{  \n  "phase1": {  \n    "thought_process": "Your thought process text description",  \n    "answer": "NO" | "Complex"  \n  },  \n  "phase2": {  \n    "thought_process": "Your thought process text description",  \n    "answer": "ETWT" | "NTST" | "ELWL" | "NLSL"  \n  }  \n}\n``` """
        elif self.think_mode == 'ms':
            input_text += """## Task:\nYour task consists of two phases:\n### Phase 1: Coordination Scenario Classification\n1. Analyze the data in \"Current Observation\" and classify the intersection into one of two coordination scenarios:\n   - **Simple-Coordination**: Few upstream or downstream lanes require coordination with low traffic complexity.\n   - **Complex-Coordination**: High traffic complexity with significant congestion in upstream or downstream lanes.\n\n### Phase 2: Signal Selection Strategy\nBased on the identified coordination scenario, select the appropriate signal using the following strategies:\n\n#### a. Simple-Coordination\n- Focus on the \"Current Observation\" data.\n- Keep the \"Notes\" in mind, compare each signal, and select the optimal one based on traffic conditions.\n\n#### b. Complex-Coordination\nConduct a comprehensive analysis and select the optimal signal by following these steps:\n1. **Current Traffic Data Analysis**: Rank lanes based on queued cars and waiting times.\n2. **Upstream Analysis**: Assess the conditions of nearby and long-distance upstream lanes for each lane.\n3. **Downstream Analysis**: Assess the conditions of nearby and long-distance downstream lanes for each lane.\n4. **Signal Consequence Prediction**: Utilize \"Historical Observation\" data for inferring the consequences of each signal. \n5. **Signal Comparison**: Systematically compare the effects of each signal option.\n6. **Decision Making**: Make a decision on the optimal signal to activate.\n\n## Requirement:\nPlease provide a JSON formatted output as follows: \n```json  \n{  \n  \"phase1\": {  \n    \"thought_process\": \"Your thought process text description\",  \n    \"answer\": \"NO\" | \"Simple\" | \"Complex\"  \n  },  \n  \"phase2\": {  \n    \"thought_process\": \"Your thought process text description\",  \n    \"answer\": \"ETWT\" | \"NTST\" | \"ELWL\" | \"NLSL\"  \n  }  \n}\n```"""
        
        else:
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
    
    def select_signal_based_on_up_down_stream_view_prompt(self):
        prompt = "**Current Observations:**\n"
        # prompt += "If a lane among 12 lanes is not mentioned, it means there are no vehicles on it at present. \n"
        observation_of_this_inter_text = self.get_current_lane_observation_text_tablestyle()
        
        up_down_stream_view_text = self.get_up_down_stream_view_text()
        prompt += "\n"
        prompt += observation_of_this_inter_text
        prompt += '{} are empty.\n'.format(self.empty_lanes)
        prompt += "\n"
        prompt += up_down_stream_view_text
        prompt += "\n"
        # prompt += "If the upstream or downstream of a lane is not mentioned, it means its occupancy is low.\n"
        if self.long_info:
            long_distance_info_text = self.get_long_distance_info_text()
            prompt += long_distance_info_text
            prompt += "\n"
        if self.feed_back:
            memories_text = self.get_memory_text()
            prompt += memories_text
            prompt += "\n"

        prompt += "**Task:**\n"
        prompt += "You must select a signal from the options [ETWT, NTST, ELWL, NLSL] to activate in the next period. For example, ETWT indicates the release of both the ET and WT lanes.\n\n"
        
        prompt += "**NOTE:**\n"
        # prompt += "- If there are many cars upstream of a certain lane, you need to estimate whether there is enough space in this lane currently and consider whether to release this lane.\n"
        # prompt += "- If you want to release a lane, you need to consider whether its downstream has enough space.\n"

        # prompt += "- Occupancy above 100% will result in severe congestion. Make sure that the occupancy of your lane and your upstream and downstream lanes never exceeds 100%.\n"
        # prompt += "- If your lane is over 100%, only release it when your downstream has enough space.\n"
        prompt += "For each Lane X, when considering activating it, keep these in mind: \n"
        prompt += "- NEVER let the occupancy of X's downstream lanes be close to 100% at any risk, as it will cause severe congestion.\n"
        prompt += "- If the upstream or downstream information of X isn't mentioned, it means that they are in a good state with low occupancy.\n"
        prompt += "- You MUST consider how much the occupancy of X's downstream lanes will increase upon releasing lane X. \n"
        prompt += "- You MUST delay the release of X if its downstream has a high occupancy rate.\n"
        prompt += "- If there are many high-occupancy lanes upstream of X and X's occupancy is not low, you MUST consider releasing X so as to help upstream lanes release.\n"
        prompt += "- You can't keep a lane waiting for too long. You MUST release the lane with excessive waiting time when the downstream condition allows.\n\n"
        # prompt += "- If nearby downstream lanes of a lane are highly occupied, releasing it will worsen the situation.\n"
        

        prompt += "**Requirement:**\n"
        prompt += "- Let's think step by step. \n"

        prompt += "- You must follow the following steps: Step 1: Current Traffic Data Analysis - Identify the (queue cars, waiting time) ranking of lanes. Step 2: Upstream Analysis - For each lane, analyze its nearby and long-distance upstream situations. Step 3: Downstream Analysis - For each lane, analyze its downstream situation and estimate how much the occupancy of downstream lanes will increase if this lane is activated based on the information in memories. Step 4: Compare each signal. Step 5: Make a decision.\n"
        # prompt += "- Consider these questions carefully before choosing a signal: 1. Is there a lane where vehicles wait too long or have a significant queue? 2. Does your current situation meet the needs of other intersections? 3. How will selecting a certain signal impact the average waiting time and queue length at all intersections?"
        # prompt += "- You must follow the following steps: Step 1: Provide your thoughts. Step 2: Give the signal answer.\n"

        prompt += "- Your answer can only be given after finishing the thought.\n"
        # prompt += "- Explain briefly before answering.\n"
        prompt += "- Answer format: You MUST provide a final signal within tags <signal>...</signal>.\n"
        prompt += "- Answer options: [<signal>ETWT</signal>, <signal>NTST</signal>, <signal>ELWL</signal>, <signal>NLSL</signal>]\n"
        return prompt

    def get_current_lane_observation_text_tablestyle(self):
        text = "#### **Intersection Lanes - Controlled by You:** \n"
        text += "|Lane|Queued Cars|Moving Cars|Average Waiting Time (mins)|Occupancy|\n"
        for lane in location_direction_dict:
            not_empty = self.current_state[lane]['queue_car_num'] + self.current_state[lane]['coming_car_num']
            if not_empty:
                text += "|{}|{}|{}|{:.1f}|{:.1f}%|\n".format(lane, self.current_state[lane]['queue_car_num'], self.current_state[lane]['coming_car_num'], self.current_state[lane]['avg_wait_time']/60, self.current_state[lane]['occupancy']*100)
        return text
    
    def get_up_down_stream_view_text(self):
        information_text = "#### **Nearby Upstream and Downstream Lanes - Controlled by Other Intersections near You:** \n"
        data_title_text = "We use (inter_id, lane) to represent these lanes. For instance, (1, 'NL') represents the NL lane at Intersection 1.\n"
        # data_title_text += "If the upstream or downstream information of a lane isn't mentioned, it means that they are in a good state with low occupancy.\n"
        # text += "|Relation|Lane|Queued Cars|Moving Cars|Average Waiting Time (mins)|Occupancy|\n"
        data_title_text += "|Relation|Queued Cars|Moving Cars|Average Waiting Time (mins)|Occupancy|\n"
        upstream_down_stream_text = ''
        for lane in self.no_empty_lanes:
            for direc in ['upstream', 'downstream']:
                if direc in self.up_down_stream_view[lane]:
                    stream_lanes_data = self.up_down_stream_view[lane][direc]
                    # ct = 1
                    for stream_lane in stream_lanes_data:
                        stream_lane_data = stream_lanes_data[stream_lane]
                        downstream_inter_lane = (stream_lane_data['inter_id'], stream_lane)
                        not_empty = len(stream_lane_data['veh2pos']) 
                        if not_empty:
                            # text += "|Your {}'s {}|{} neighbor's {}|{}|{}|{:.1f}|{:.1f}%|\n".format(lane, direc, stream_lane_data['location'], stream_lane, stream_lane_data['queue_len'], sum(stream_lane_data['cells']), stream_lane_data['avg_wait_time']/60, stream_lane_data['occupancy']*100)
                            upstream_down_stream_text += "|Your {}'s {} lane {}|{}|{}|{:.1f}|{:.1f}%|\n".format(lane, direc, downstream_inter_lane, stream_lane_data['queue_len'], sum(stream_lane_data['cells']), stream_lane_data['avg_wait_time']/60, stream_lane_data['occupancy']*100)
                            # ct += 1
        if len(upstream_down_stream_text) == 0:
            upstream_down_stream_text = "All nearby upstream and downstream lanes are in good state with low occupancy.\n"
            total_text = information_text + upstream_down_stream_text
        else:
            total_text = information_text + data_title_text + upstream_down_stream_text
        return total_text

    
    def get_observation_text(self):
        current_state = self.log['states'][-1]
        text = "**Current Observation of each road:**\n"
        for loc in ['North', 'South','East','West']:
            text += "- {}: {} queued cars, {} moving cars, {:.1f} mins wait time \n".format(loc, current_state[loc[0]]['queue_car_num'], current_state[loc[0]]['coming_car_num'], current_state[loc[0]]['wait_time']/60)
        history_len = 11 if len(self.log['states']) >=10 else len(self.log['states'])
        if history_len > 1:
            text += "**Traffic Data of this intersection: (Last {:.1f} mins, 30-sec intervals):**\n".format((history_len-1)/2)
        text += "NOTE: In the following data, the list is arranged chronologically with the earliest time points on the left and the latest time points on the right.\n"
        text += "Car Release Volume: \n"
        for key in ['North', 'South','East','West', 'Total']:
            text += "- {}: {}\n".format(key, self.log['release_num'][key][-history_len:])
        text += "Car Entry Volume: \n"
        for key in ['North', 'South','East','West', 'Total']:
            text += "- {}: {}\n".format(key, self.log['entry_num'][key][-history_len:])
        return text

    def get_lane_observation_text(self, state_idx = -1, target_lane = None):
        if target_lane:
            target_lane_state = self.log['states'][state_idx][target_lane]
            text = "Current observation of {} lane: {} queued cars, {} moving cars, {:.1f} mins average waiting time, {}% occupancy. \n".format(target_lane, target_lane_state['queue_car_num'], target_lane_state['coming_car_num'], target_lane_state['avg_wait_time']/60, self.congest_degree[state_idx][target_lane]*100)
        else:
            current_state = self.log['states'][state_idx]
            text = "**Current Observation of each lane:**\n"
            for lane in location_direction_dict:
                text += "- {}: {} queued cars, {} moving cars, {:.1f} mins average waiting time, {}% occupancy. \n".format(lane, current_state[lane]['queue_car_num'], current_state[lane]['coming_car_num'], current_state[lane]['avg_wait_time']/60, self.congest_degree[state_idx][lane]*100)
            
        return text
    def name2loc(self, intersection_name):
        intersection_name = intersection_name.split('_')
        return (int(intersection_name[1]), int(intersection_name[2]))
    
    def get_road_observation_text(self, state_idx, target_road):
        road = target_road[0].upper()
        current_road_state = self.log['states'][state_idx][road]
        inter_loc = self.name2loc(self.name)
        road_avg_waiting_time = current_road_state['wait_time']/current_road_state['queue_car_num'] if current_road_state['queue_car_num'] !=0 else 0.0
        road_lane_congest_degree_list = [self.congest_degree[state_idx][lane] for lane in self.lane_list if road in lane]
        road_avg_congest_degree = sum(road_lane_congest_degree_list)/len(road_lane_congest_degree_list)
        text = "Current observation of {} road of intersection {}: {} queued cars, {} moving cars, {:.1f} mins average waiting time, {}% occupancy. \n".format(target_road, inter_loc, current_road_state['queue_car_num'], current_road_state['coming_car_num'], road_avg_waiting_time/60, road_avg_congest_degree*100)

        return text
    
    def get_lane_state(self, state_idx, lane):
        return self.log['states'][state_idx][lane]
    
    def get_road_state(self, state_idx, road):
        road = road[0].upper()
        road_state = self.log['states'][state_idx][road]
        road_state['avg_wait_time'] = road_state['wait_time']/road_state['queue_car_num'] if road_state['queue_car_num'] !=0 else 0.0
        return road_state  
    
    def get_system_prompt(self):
     # text += "Your main goal is to efficiently manage traffic flow and coordinate with other intersections to ensure minimal waiting times and vehicle queues across all intersections   . This involves selecting the optimal traffic signal phases at each interval based on current traffic conditions and the needs of other intersections."
        text = "You are a traffic signal controller at a four-way intersection with 12 lanes: [NL, NT, NR, SL, ST, SR, EL, ET, ER, WL, WT, WR]. Each lane is labeled by direction and movement: N for north, S for south, E for east, W for west, L for left turn, T for through, and R for right turn. For instance, ET stands for the East Through lane, where traffic moves straight ahead from east to west. WL is the West Left-turn lane, where traffic turns left from west to south. Right turns are always allowed. There are four signal options: [ETWT, NTST, ELWL, NLSL]. For example, ETWT indicates the release of both the ET and WT lanes. Your goal is to optimize traffic flow and coordinate with nearby intersections to minimize wait times and queues by selecting the best signal phases based on current conditions.\n"
        # text = "You are a traffic light controller at a four-way intersection, managing traffic from the east, west, north, and south. Each direction has three lanes designated for left turns, straight movements, and right turns. Thus, we have 12 lanes: [NL, NT, NR, SL, ST, SR, EL, ET, ER, WL, WT, WR]. We represent each lane using the initials of its direction combined with the roadway direction. For instance, ET stands for the East Through lane, where traffic moves straight ahead from east to west. WL is the West Left-turn lane, where traffic turns left from west to south. Right turns are permitted at any time. \nThe average speed of cars is 11 meters per second.  \nYour main goal is to efficiently manage traffic flow and coordinate with other intersections to ensure minimal waiting times and vehicle queues across all intersections. This involves selecting the optimal traffic signal phases at each interval based on current traffic conditions and the needs of other intersections.\n"
        return text
    
