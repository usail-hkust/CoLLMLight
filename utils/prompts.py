SYSTEM_PROMPT = f"""You are a traffic signal control expert responsible for managing a four-way intersection within a larger road network. Your objective is to optimize both local and network-wide traffic flow and safety by selecting appropriate signal phases at each decision point. You must consider not only the traffic condition at your intersection, but also the influence on upstream and downstream intersections.

Your decision-making follows a two-stage process:

1. **Advance Traffic Reasoning (at timestep t-1):**  
   Based on current and historical traffic observations, you should conduct sufficient but necessary analysis to understand the key lanes, congestion risks, and propagation effects. Your goal is to provide a well-grounded recommendation and justification that can support the next-stage decision.

2. **Reactive Action (at timestep t):**  
   Given the real-time observation at timestep t and the prior reasoning, you must promptly determine the optimal signal phase. If significant changes are detected compared to previous reasoning, you should adapt your decision accordingly.

Follow all safety principles and ensure that each decision balances local urgency and global impact.

## Background Context
An intersection has 12 lanes: [NL, NT, NR, SL, ST, SR, EL, ET, ER, WL, WT, WR]. Each lane is labeled by direction and movement: N for north, S for south, E for east, W for west, L for left turn, T for through, and R for right turn. For instance, ET stands for the East Through lane, where traffic moves straight ahead from east to west. WL is the West Left-turn lane, where traffic turns left from west to south. Right turns are always allowed. There are four signal options: [ETWT, NTST, ELWL, NLSL]. For example, ETWT indicates the release of both the ET and WT lanes. The signal phase duration is set to thirty seconds.

## Note:
For each Lane X, when considering activating it, keep these in mind: 
    - NEVER NEVER let the occupancy of X's downstream lanes be close to 100% at any risk, as it will cause severe congestion, this is FIRST PRIORITY.
    - If the upstream or downstream information of X isn't mentioned, it means that they are in a good state with low occupancy.
    - You MUST consider how much the occupancy of X's downstream lanes will increase upon releasing lane X. 
    - You MUST delay the release of X if its downstream has a high occupancy rate.
    - If there are many high-occupancy lanes upstream of X and X's occupancy is not low, you MUST consider releasing X so as to help upstream lanes release.
    - You can't keep a lane waiting for too long. You MUST release the lane with excessive waiting time when the downstream condition allows.

"""


Advance_Traffic_Reasoning_Template = """You are now in the **Advance Traffic Reasoning** phase (at time step {t0}).

Your goal is to perform deep but adaptive reasoning based on historical and current traffic conditions. This reasoning will guide the signal decision for the next time step {t1}.

---

{historical_observation}

---

{current_observation}


---

## Instruction
Please proceed through the following steps:
### Step 1: Identify Critical Lanes
Examine the current intersection and its upstream/downstream neighbors to identify critical local and neighboring lanes that are critical or congested and require attention; output as:
Local Critical Lanes: ...
Neighboring Critical Lanes: ...

### Step 2: Perform Adaptive Reasoning
Select and apply only the **necessary reasoning steps** to support an effective and fast decision at the next time step {t1}.

Suggested reasoning topics include (but are not limited to):  
- Local queue and waiting time analysis
- Upstream/downstream traffic influence
- Reflection on previous signal effectiveness
- Prediction of the traffic state at timestep {t1}
- Anticipated effects of activating specific signal phases (e.g., NTST, ETWT, NLSL, ELWL) at {t1}

### Step 3: Conditional Signal Suggestions
Provide one or more signal strategies that may be appropriate at time step {t1}.  
For each **suggested signal**, specify the situations under which it would be effective, including reasoning and expected effects.

Structure each strategy as:

- **Suggested Signal**: The signal phase that may be activated at time step {t1}  
    - **Applicable Conditions**: What kind of traffic pattern or development this signal is suitable for
    - **Rationale**: Why this signal is an appropriate choice under those conditions  
    - **Expected Effect**: The anticipated outcome if this signal is activated

### Step 4: Summary

## Notes
- Your reasoning should be adaptive: longer and deeper if traffic conditions are complex, shorter if simple.
- You may skip Steps 2 and 3 if the scenario is simple—for example, when there are no critical neighboring lanes.

## Output Format  
Your output should follow the structure below:

## Identify Critical Lanes

## Reasoning Topic 1

## Reasoning Topic 2

...

## Conditional Signal Suggestions

## Summary
"""

Reactive_Action_Template = """You are now in the **Reactive Action** phase at time step {t1}.

Your goal is to quickly and accurately select the signal phase to activate at this moment, based on:
- The prior **Advance Traffic Reasoning (ATR)** conducted at time step {t0}
- The **Current Observation** at time step {t1}
- The **Local Signal Priority** ranking based on estimated waiting time reduction

---

# Advance Traffic Reasoning (ATR) at time step {t0}

{advance_traffic_reasoning_results}

---

{current_observation}

---

{local_signal_priority}

---

# Instruction

Please:

1. Compare the ATR suggestions with the current traffic state.  
2. If the current situation aligns with ATR predictions, select the recommended signal accordingly.  
3. If the current situation **deviates** from ATR, make necessary adjustments to respond to new patterns.  
4. If **upstream/downstream influence is minimal or negligible**, directly select the top-ranked signal from the Local Signal Priority table.  
5. Do not congest downstream lanes. When the occupancy of Lane X, plus that of any other X's downstream lanes, is close to 100%, you must delay the release of X.
6. Decide whether the ATR output was **actually useful** in your decision using the `used_atr` flag:  
   - `used_atr = 1`: The ATR output was referenced and influenced the decision.  
   - `used_atr = 0`: The decision was made independently of the ATR output.

---

# Output Format

Return your result in the following **structured JSON format**:

```json
{{
  "signal": "<Selected signal phase, e.g., ETWT, NTST, NLSL, ELWL>",
  "reason": "<Brief justification: why this signal is chosen based on traffic observation, local signal ranking, and (if applicable) the ATR suggestion.>",
  "used_atr": <0 or 1>
}}
```"""

def get_advance_traffic_reasoning_prompt(t0, t1, historical_observation, current_observation) -> str:
    prompt = Advance_Traffic_Reasoning_Template.format(
        t0=t0,
        t1=t1,
        historical_observation=historical_observation,
        current_observation=current_observation,
    )
    return prompt


def get_reactive_action_prompt(t0, t1, advance_traffic_reasoning_results, current_observation, local_signal_priority) -> str:
    prompt = Reactive_Action_Template.format(
        t0=t0,
        t1=t1,
        advance_traffic_reasoning_results=advance_traffic_reasoning_results,
        current_observation=current_observation,
        local_signal_priority=local_signal_priority,
    )
    return prompt

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
    #data.keys(): dict_keys(['Intersection', 'Timestep', 'Traffic_state_history', 'Current_Observation', 'Signal_Rank', 'Signal_Consequence', 'Best_Signal'])
    t0 = data['Timestep']-1
    t1 = data['Timestep']
    historical_observation = get_historical_observation_atr(t0, data['Traffic_state_history'])
    current_observation = get_current_observation_atr(t1, data['Traffic_state_history'])
    prompt = get_advance_traffic_reasoning_prompt(t0, t1, historical_observation, current_observation)
    return prompt

def get_advance_traffic_reasoning_prompt_adaptive(t0, t1, historical_observation, current_observation, complex_level):
    prompt = get_advance_traffic_reasoning_prompt(t0, t1, historical_observation, current_observation)
    if complex_level == 'easy':
        prompt += (
            "## Requirement (Strict): You must only use the following reasoning steps:\n"
            "- Identify critical lanes\n"
            "- Suggest a local traffic signal strategy\n"
        )
    else:
        prompt += (
            "## Requirement: Select only the necessary reasoning topics relevant to the current situation. Avoid redundancy.\n"
        )
    return prompt

def get_atr_user_prompt_wsys(data):
    #data.keys(): dict_keys(['Intersection', 'Timestep', 'Traffic_state_history', 'Current_Observation', 'Signal_Rank', 'Signal_Consequence', 'Best_Signal'])
    t0 = data['Timestep']-1
    t1 = data['Timestep']
    historical_observation = get_historical_observation_atr(t0, data['Traffic_state_history'])
    current_observation = get_current_observation_atr(t1, data['Traffic_state_history'])
    prompt = get_advance_traffic_reasoning_prompt(t0, t1, historical_observation, current_observation)
    return SYSTEM_PROMPT + "\n" + prompt

def get_ra_user_prompt(data, atr_results):
    t0 = data['Timestep']-1
    t1 = data['Timestep']
    current_observation_text = get_current_observation_ra(t1, data['Traffic_state_history'])
    local_signal_priority = get_local_signal_priority_text(data['Signal_Rank'])
    prompt = get_reactive_action_prompt(t0, t1, advance_traffic_reasoning_results=atr_results, current_observation=current_observation_text, local_signal_priority=local_signal_priority)
    return prompt

def get_ra_user_prompt_wsys(data, atr_results):
    t0 = data['Timestep']-1
    t1 = data['Timestep']
    current_observation_text = get_current_observation_ra(t1, data['Traffic_state_history'])
    local_signal_priority = get_local_signal_priority_text(data['Signal_Rank'])
    prompt = get_reactive_action_prompt(t0, t1, advance_traffic_reasoning_results=atr_results, current_observation=current_observation_text, local_signal_priority=local_signal_priority)
    return SYSTEM_PROMPT + "\n" + prompt
