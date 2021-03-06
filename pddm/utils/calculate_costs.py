# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from scipy.special import expit

def cost_per_step(pt, prev_pt, costs, actions, dones, reward_func, catastrophe_pred, risk_aversion_type, beta):
    step_rews, step_dones = reward_func(pt[..., :-1], actions)
    if catastrophe_pred and risk_aversion_type == "state":
        collision = expit(pt[..., -1]) #sigmoids it
        costs[collision > beta] += 500
    

    dones = np.logical_or(dones, step_dones)
    costs[dones > 0] += 500
    costs[dones == 0] -= step_rews[dones == 0]

    return costs, dones


def calculate_costs(resulting_states_list, actions, reward_func,
                    evaluating, take_exploratory_actions, traj_sampling_ratio, catastrophe_pred, risk_aversion_type, beta):
    """Rank various predicted trajectories (by cost)

    Args:
        resulting_states_list :
            predicted trajectories
            [ensemble_size, horizon+1, N, statesize]
        actions :
            the actions that were "executed" in order to achieve the predicted trajectories
            [N, h, acsize]
        reward_func :
            calculates the rewards associated with each state transition in the predicted trajectories
        evaluating :
            determines whether or not to use model-disagreement when selecting which action to execute
            bool
        take_exploratory_actions :
            determines whether or not to use model-disagreement when selecting which action to execute
            bool

    Returns:
        cost_for_ranking : cost associated with each candidate action sequence [N,]
    """

    ensemble_size = len(resulting_states_list)
    tiled_actions = np.tile(actions, (ensemble_size, 1, 1))

    ###########################################################
    ## some reshaping of the predicted trajectories to rate
    ###########################################################

    N = len(resulting_states_list[0][0])

    #resulting_states_list is [ensSize, H+1, N, statesize]
    resulting_states = []
    for timestep in range(len(resulting_states_list[0])): # loops over H+1
        all_per_timestep = []
        for entry in resulting_states_list: # loops over ensSize
            all_per_timestep.append(entry[timestep])
        all_per_timestep = np.concatenate(all_per_timestep)  #[ensSize*N, statesize]
        resulting_states.append(all_per_timestep)
    #resulting_states is now [H+1, ensSize*N, statesize]

    ###########################################################
    ## calculate costs associated with each predicted trajectory
    ######## treat each traj from each ensemble as just separate trajs
    ###########################################################

    #init vars for calculating costs
    costs = np.zeros((N * len(resulting_states_list),))
    prev_pt = resulting_states[0]
    dones = np.zeros((N * len(resulting_states_list),))

    #accumulate cost over each timestep
    for pt_number in range(len(resulting_states_list[0]) - 1):

        #array of "current datapoint" [(ensemble_size*N) x state]
        pt = resulting_states[pt_number + 1]
        #update cost at the next timestep of the H-step rollout
        actions_per_step = tiled_actions[:, pt_number]
        costs, dones = cost_per_step(pt, prev_pt, costs, actions_per_step, dones, reward_func, catastrophe_pred, risk_aversion_type, beta)
        #update
        prev_pt = np.copy(pt)

    ###########################################################
    ## assigns costs associated with each predicted trajectory
    ####### need to consider each ensemble separately again
    ####### perform ranking based on either
    #"mean costs" over ensemble predictions (for a given action sequence A)
    # or
    #"model disagreement" over ensemble predictions (for a given action sequence A)
    ###########################################################

    #consolidate costs (ensemble_size*N) --> (N)
    new_costs = []
    for i in range(N):
        # 1-a0 1-a1 1-a2 ... 2-a0 2-a1 2-a2 ... 3-a0 3-a1 3-a2...
        new_costs.append(costs[i::N])  #start, stop, step

    new_costs = np.array(new_costs)
    ######### Trajectory sampling aggregation
    new_costs = np.reshape(new_costs, (int(N//traj_sampling_ratio), -1))
    #####################################################################################################    
    if catastrophe_pred and risk_aversion_type == "reward":
        new_costs = np.array(new_costs)
        # Discounted reward sum calculation for CARL (Reward). At percentile == 100, this is normal PDDM
        if beta <= 1:
            k_percentile = np.expand_dims(-np.percentile(-new_costs, q=beta * 100, axis=1, interpolation="nearest"), 1)
            cost_mask = new_costs < k_percentile
        new_costs[cost_mask] = 0
        discounted_sum = np.sum(new_costs, axis=1)
        new_costs[cost_mask] = float('nan')
        lengths = np.sum(~np.isnan(new_costs), axis=1)
        mean_cost = discounted_sum / lengths
        # if invalid trajectory, then make the cost the mean so they cancel out in SD calculation
        mean_cost_repeated = np.repeat(np.expand_dims(mean_cost, 1), new_costs.shape[1], -1)
        new_costs = np.where(cost_mask, mean_cost_repeated, new_costs) 
        std_cost = np.sqrt(np.sum((new_costs - mean_cost_repeated)**2, axis=1) / lengths)
    else:
        mean_cost = np.mean(new_costs, 1)
        std_cost = np.std(new_costs, 1)
    #####################################################################################################    

    #rank by rewards
    if evaluating:
        cost_for_ranking = mean_cost
    #sometimes rank by model disagreement, and sometimes rank by rewards
    else:
        if take_exploratory_actions:
            cost_for_ranking = mean_cost - 4 * std_cost
            print("   ****** taking exploratory actions for this rollout")
        else:
            cost_for_ranking = mean_cost

    return cost_for_ranking, mean_cost, std_cost
