from utils.my_utils import location_direction_dict, location_dict
from utils.utils import *
import numpy as np
from copy import deepcopy
from collections import defaultdict
from utils.prompts import get_advance_traffic_reasoning_prompt, get_reactive_action_prompt

class IntersectionAgent(object):
    def __init__(self, agent_conf, traffic_env_conf, LLM):
        ## indentification info
        self.agent_conf = agent_conf
        self.name = agent_conf['inter_name']
        self.id = agent_conf['inter_id']
        # self.boundary = agent_conf['boundary']
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
        self.h_w_size = agent_conf['h_w_size'] if 'h_w_size' in agent_conf else None
        self.syn_psr = False # default fase
        
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

        ## other
        self.current_state = None
        self.last_queue = 0
        # self.queue_threshold
        self.accumulate_times = 0
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
    

        
    def update_state(self, state, last_action_id = None):
        ## queue, queue_diff, state
        queue_num = 0
        wait_time = 0
        if last_action_id!=None:
            update_memory = True
        else:
            update_memory = False
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

    def get_historical_observation_v2(self, timestep = None):
        # get the history before timestep t-1
        if timestep == None:
            timestep = len(self.log['states'])-1
        if self.syn_psr:
            history_state = self.traffic_state_log[-self.h_w_size:]
        else:
            history_state = self.traffic_state_log[-self.h_w_size-1:-1]
            

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
        if self.h_w_size > 0:
            return text
        else:
            return ""
    
    def get_historical_observation(self, timestep = None):
        if timestep == None:
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
    
    def get_current_observation_text_v2(self, step_num):
        signal = self.traffic_state_log[-1]['Signal']
        text = "## Current Observation\n"
        text += f"Timestep {step_num} signal: {signal}\n"
        return text
    
    def get_current_observation_text_v3(self, step_num):
        signal = self.traffic_state_log[-1]['Signal']
        text = "# Current Observation\n"
        text += "- Lanes include both signal-controlled lanes at the current intersection (e.g., NL, NT, SL, ST, EL, ET, WL, WT) and upstream/downstream lanes from neighboring intersections (e.g., SL's upstream lane (4, ST)). **Only lanes with vehicles are shown.**\n"
        text += "- Values are shown as: value or value(+change from previous timestep).\n\n"
        text += f"Timestep {step_num-1} signal: {signal}\n"
        current_state = self.traffic_state_log[-1]
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

    def estimate_effective_range(self):
        state = self.state
        car_speed = 11.11
        if len(self.intersection_list) == 33:
            car_speed = 16.67
        seg_num = 10
        effective_range_distance = car_speed * self.signal_time
        range_list = []
        for lane in location_direction_dict:
            road_length = state[lane]['road_length']
            seg_length = road_length/seg_num
            effective_range_cell = int(effective_range_distance//seg_length)
            if effective_range_cell >= seg_num:
                effective_range_cell = seg_num - 1
            range_list.append(effective_range_cell)
        return range_list
    
    def get_local_signal_priority_text(self):
        text = "# Local Signal Priority\n"
        text += "This rank only considers your own intersection and assumes the downstream allows the release.\n"
        lane_release_metrix = {}
        effective_range_list = self.estimate_effective_range()
        for i, lane in enumerate(location_direction_dict):
            lane_range = effective_range_list[i]
            going_cars_num = np.sum(self.state[lane]["cells"][:lane_range+1])
            stop_cars_num = np.sum(self.state[lane]["ql_cells"][:lane_range+1])
            lane_release_metrix[lane] = stop_cars_num * self.state[lane]["avg_wait_time"] + stop_cars_num * self.signal_time + going_cars_num * self.signal_time 
        signal_value_dict = {}
        for p in self.signal_list:
            signal_value_dict[p] = lane_release_metrix[p[:2]] + lane_release_metrix[p[2:]]

        text += "|Rank|Signal|Waiting Time Reduction|\n"
        for i, p in enumerate(sorted(signal_value_dict, key=signal_value_dict.get, reverse=True)):
            text += "|{}|{}|{:.1f} mins|\n".format(i+1, p, signal_value_dict[p]/60)
        text += '\n'
        return text
    
    def advance_traffic_reasoning_prompt(self, step_num):
        # step_num is timestep t, advance traffic reasoning in t-1
        if self.syn_psr:
            step_num = step_num+1
        historical_observation_text = self.get_historical_observation_v2(step_num-1)
        current_observation_text = self.get_current_observation_text_v2(step_num-1)
        prompt = get_advance_traffic_reasoning_prompt(t0=step_num-1, t1=step_num, historical_observation=historical_observation_text, current_observation=current_observation_text)
        return prompt
    
    def reactive_action_prompt(self, advance_traffic_reasoning_results, step_num):
        current_observation_text = self.get_current_observation_text_v3(step_num)
        local_signal_priority = self.get_local_signal_priority_text()
        prompt = get_reactive_action_prompt(t0=step_num-1, t1=step_num, advance_traffic_reasoning_results=advance_traffic_reasoning_results, current_observation=current_observation_text, local_signal_priority=local_signal_priority)
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
    
