"""
Fixed-Time agent.
Use pre-assigned time duration for each phase.
"""

from .agent import Agent


class FixedtimeAgent(Agent):
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round, intersection_id):
        # Initialize base class
        super(FixedtimeAgent, self).__init__(dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id)

        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.phase_length = len(self.dic_traffic_env_conf["PHASE"])
        self.action = 0  # Initial signal phase

        # Phase number mapping (for compatibility with phase numbering in certain environments)
        self.DIC_PHASE_MAP = {i: i for i in range(self.phase_length)}

    def choose_action(self, count, state):
        """Fixed-time rotation strategy: switch to the next phase when the current phase time reaches the preset duration"""
        cur_phase_raw = state["cur_phase"][0]
        if cur_phase_raw == -1:
            return self.action  # Initial state: maintain current phase

        cur_phase = self.DIC_PHASE_MAP[cur_phase_raw]
        time_in_phase = state["time_this_phase"][0]
        fixed_time = self.dic_agent_conf["FIXED_TIME"][cur_phase]

        if time_in_phase >= fixed_time:
            # Switch to next phase (looping)
            self.action = (cur_phase + 1) % self.phase_length
        # Otherwise maintain current phase

        return self.action
