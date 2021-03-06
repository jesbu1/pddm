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
import copy
import matplotlib.pyplot as plt
from scipy.special import expit

# my imports
from pddm.samplers import trajectory_sampler
from pddm.utils.helper_funcs import do_groundtruth_rollout
from pddm.utils.helper_funcs import turn_acs_into_acsK
from pddm.utils.calculate_costs import calculate_costs

class MPPI(object):

    def __init__(self, env, dyn_models, reward_func, rand_policy, use_ground_truth_dynamics,
                 execute_sideRollouts, plot_sideRollouts, params):

        ###########
        ## params
        ###########
        self.K = params.K
        self.horizon = params.horizon
        self.N = params.num_control_samples
        self.traj_sampling_ratio = params.traj_sampling_ratio
        self.catastrophe_pred = params.catastrophe_pred
        self.finetuning = params.finetuning
        self.caution_beta = params.beta
        self.risk_aversion_type = params.risk_aversion_type
        self.rand_policy = rand_policy
        self.use_ground_truth_dynamics = use_ground_truth_dynamics
        self.dyn_models = dyn_models
        self.execute_sideRollouts = execute_sideRollouts
        self.plot_sideRollouts = plot_sideRollouts
        self.reward_func = reward_func
        self.env = copy.deepcopy(env)

        #############
        ## init mppi vars
        #############
        self.sample_velocity = params.rand_policy_sample_velocities
        self.ac_dim = self.env.env.action_space.shape[0]
        self.mppi_kappa = params.mppi_kappa
        self.sigma = params.mppi_mag_noise * np.ones(self.env.action_dim)
        self.beta = params.mppi_beta
        self.mppi_mean = np.zeros((self.horizon, self.ac_dim))  #start mean at 0


        #############
        ## Debugging
        #############
        self.observation_history = []
        self.catastrophe_labels = []
        self.catastrophe_scores = []
        self.prediction_history = []
        self.action_history = []
        #self.positive_prediction_correct = [[]]
        #self.negative_prediction_correct = [[] for _ in range(self.horizon)]
        #self.total_prediction_correct = [[] for _ in range(self.horizon)]

    ###################################################################
    ###################################################################
    #### update action mean using weighted average of the actions (by their resulting scores)
    ###################################################################
    ###################################################################

    def mppi_update(self, scores, mean_scores, std_scores, all_samples):

        #########################
        ## how each sim's score compares to the best score
        ##########################
        S = np.exp(self.mppi_kappa * (scores - np.max(scores)))  # [N,]
        denom = np.sum(S) + 1e-10

        ##########################
        ## weight all actions of the sequence by that sequence's resulting reward
        ##########################
        S_shaped = np.expand_dims(np.expand_dims(S, 1), 2)  #[N,1,1]
        weighted_actions = (all_samples * S_shaped)  #[N x H x acDim]
        self.mppi_mean = np.sum(weighted_actions, 0) / denom

        ##########################
        ## return 1st element of the mean, which corresps to curr timestep
        ##########################
        return self.mppi_mean[0]

    def clear_debugging_vars(self):
        self.observation_history = []
        self.catastrophe_labels = []
        self.catastrophe_scores = []
        self.prediction_history = []
        self.action_history = []
        #self.prediction_correct = [[] for _ in range(self.horizon)]
        #self.negative_prediction_correct = [[] for _ in range(self.horizon)]
        #self.total_prediction_correct = [[] for _ in range(self.horizon)]

    def get_action(self, step_number, curr_state_K, actions_taken_so_far,
                   starting_fullenvstate, evaluating, take_exploratory_actions):

        # init vars
        curr_state_K = np.array(curr_state_K)  #[K, sa_dim]

        ## Debugging
        self.observation_history.append(curr_state_K[0])
        if len(self.observation_history) > self.horizon + 1:
            self.observation_history.pop(0)

        # remove the 1st entry of mean (mean from past timestep, that was just executed)
        # and copy last entry (starting point, for the next timestep)
        past_action = self.mppi_mean[0].copy()
        self.mppi_mean[:-1] = self.mppi_mean[1:]

        ##############################################
        ## sample candidate action sequences
        ## by creating smooth filtered trajecs (noised around a mean)
        ##############################################

        np.random.seed()  # to get different action samples for each rollout

        #sample noise from normal dist, scaled by sigma
        if(self.sample_velocity):
            eps_higherRange = np.random.normal(
                loc=0, scale=1.0, size=(self.N, self.horizon,
                                        self.ac_dim)) * self.sigma
            lowerRange = 0.3*self.sigma
            num_lowerRange = int(0.1*self.N)
            eps_lowerRange = np.random.normal(
                loc=0, scale=1.0, size=(num_lowerRange, self.horizon,
                                        self.ac_dim)) * lowerRange
            eps_higherRange[-num_lowerRange:] = eps_lowerRange
            eps=eps_higherRange.copy()
        else:
            eps = np.random.normal(
                loc=0, scale=1.0, size=(self.N, self.horizon,
                                        self.ac_dim)) * self.sigma

        # actions = mean + noise... then smooth the actions temporally
        all_samples = eps.copy()
        for i in range(self.horizon):

            if(i==0):
                all_samples[:, i, :] = self.beta*(self.mppi_mean[i, :] + eps[:, i, :]) + (1-self.beta)*past_action
            else:
                all_samples[:, i, :] = self.beta*(self.mppi_mean[i, :] + eps[:, i, :]) + (1-self.beta)*all_samples[:, i-1, :]

        # The resulting candidate action sequences:
        # all_samples : [N, horizon, ac_dim]
        all_samples = np.clip(all_samples, -1, 1)

        ########################################################################
        ### make each action element be (past K actions) instead of just (curr action)
        ########################################################################

        #all_samples : [N, horizon, ac_dim]
        #all_acs : [N, horizon, K, ac_dim]
        all_acs = turn_acs_into_acsK(actions_taken_so_far, all_samples, self.K,
                                     self.N, self.horizon)

        #################################################
        ### Get result of executing those candidate action sequences
        #################################################

        if self.use_ground_truth_dynamics:
            paths = trajectory_sampler.sample_paths_parallel(
                self.N,
                all_samples,
                actions_taken_so_far,
                starting_fullenvstate,
                self.env,
                suppress_print=True,
            )  #list of dicts, each w observations/actions/etc.

            #the taken number of paths is num_cpu*(floor(self.N/num_cpu))
            #rather than self.N, so update parameter accordingly
            self.N = len(paths)
            all_samples = all_samples[:self.N]

            resulting_states = [entry['observations'] for entry in paths]
            resulting_states = np.swapaxes(resulting_states, 0, 1)
            resulting_states_list = [resulting_states]
        else:
            ##################################################
            ### Trajectory sampling edit, every self.traj_sampling_ratio is a new set of actions
            ##################################################
            all_acs = np.repeat(all_acs, self.traj_sampling_ratio, axis=0)

            resulting_states_list = self.dyn_models.do_forward_sim(
                [curr_state_K, 0], np.copy(all_acs))
            resulting_states_list = np.swapaxes(resulting_states_list, 0,1)  #[ensSize, horizon+1, N, statesize]

        ############################
        ### evaluate the predicted trajectories
        ############################

        # calculate costs [N,]
        use_catastrophe_in_prediction = self.catastrophe_pred and self.finetuning
        costs, mean_costs, std_costs = calculate_costs(resulting_states_list, all_samples,
                                self.reward_func, evaluating, take_exploratory_actions, self.traj_sampling_ratio, use_catastrophe_in_prediction, self.risk_aversion_type, self.caution_beta)

        # uses all paths to update action mean (for horizon steps)
        # Note: mppi_update needs rewards, so pass in -costs
        selected_action = self.mppi_update(-costs, -mean_costs, std_costs, all_samples)

        ## Debugging
        prediction_horizon = 7
        #self.prediction_history.append(np.array(resulting_states_list)[:, -(8 - prediction_horizon), :, -1])
        #if len(self.prediction_history) > self.horizon + 1:
        #    self.prediction_history.pop(0)
        self.action_history.append(np.copy(self.mppi_mean))
        if len(self.action_history) > self.horizon + 1:
            self.action_history.pop(0)

        # debugging catastrophe prediction
        if len(self.observation_history) == self.horizon + 1:
            horizon_steps_ago_state = np.expand_dims(np.array(self.observation_history[-(prediction_horizon + 1)]), 0)
            selected_acs = np.expand_dims(np.expand_dims(self.action_history[-(prediction_horizon + 1)], 0), 2)
            resulting_final_actions_states_list = self.dyn_models.do_forward_sim(
                [horizon_steps_ago_state, 0], np.copy(selected_acs))
            resulting_final_actions_states_list = np.array(resulting_final_actions_states_list)[prediction_horizon, :, :, -1]
            cat_pred = expit(resulting_final_actions_states_list)
            #cat_pred = np.where(cat_pred > 0.5, np.ones(cat_pred.shape), np.zeros(cat_pred.shape))
            self.catastrophe_labels.append(self.observation_history[-1][-1])
            #self.catastrophe_scores.append(self.prediction_history[self.horizon - prediction_horizon])
            self.catastrophe_scores.append(cat_pred)
        #for i in range(len(self.observation_history)):
        #    correct = self.observation_history[i][-1] == cat_pred
        #    self.total_prediction_correct[i].append(correct.mean())
        #    if self.observation_history[i][-1] == 1:
        #        positive_correct = np.mean(cat_pred[i] == 1)
        #        self.positive_prediction_correct[0].append(positive_correct)
        #    else:
        #        negative_correct = np.mean(cat_pred[i] == 0)
        #        self.negative_prediction_correct[i].append(negative_correct)


        #########################################
        ### execute the candidate action sequences on the real dynamics
        ### instead of just on the model
        ### useful for debugging/analysis...
        #########################################
        if self.execute_sideRollouts:
            if (step_number % self.horizon)==0:
                cmap = plt.get_cmap('jet_r')
                num_sims = 5
                indices_to_vis = [0, 1, 2, 3, 4, 6, -3, -2]
                curr_plot = 1
                num_plots = len(indices_to_vis)
                for index_state_to_vis in indices_to_vis:
                    plt.subplot(num_plots, 1, curr_plot)
                    plt.ylabel(index_state_to_vis)
                    for sim_num in range(num_sims):
                        true_states = do_groundtruth_rollout(
                            all_samples[sim_num], self.env,
                            starting_fullenvstate, actions_taken_so_far)
                        color = cmap(float(sim_num) / num_sims)

                        ###if(step_number%10==0):
                        plt.plot(
                            resulting_states_list[-1]
                            [:, sim_num, index_state_to_vis],
                            '--',
                            c=color,
                            label=sim_num)
                        plt.plot(
                            np.array(true_states)[:, index_state_to_vis],
                            '-',
                            c=color)
                    curr_plot += 1

                if self.plot_sideRollouts:
                    plt.legend()
                    plt.show()

        return selected_action, resulting_states_list
