# # Unity ML-Agents Toolkit
# ## ML-Agent Learning (SAC)
# Contains an implementation of SAC as described in https://arxiv.org/abs/1801.01290
# and implemented in https://github.com/hill-a/stable-baselines

import logging
from collections import deque, defaultdict
from typing import List, Any, Dict
import os

import numpy as np
import tensorflow as tf

from mlagents.envs import AllBrainInfo, BrainInfo
from mlagents.envs.action_info import ActionInfoOutputs
from mlagents.envs.timers import timed, hierarchical_timer
from mlagents.trainers.buffer import Buffer, PriorityBuffer
from mlagents.trainers.sac.policy import SACPolicy
from mlagents.trainers.trainer import UnityTrainerException
from mlagents.trainers.rl_trainer import RLTrainer, AllRewardsOutput
from mlagents.trainers.components.reward_signals import RewardSignalResult


LOGGER = logging.getLogger("mlagents.trainers")
BUFFER_TRUNCATE_PERCENT = 0.8


class AgentLastInfo:
    def __init__(self):
        self.last_brain_info = None
        self.last_take_action_outputs = None


class AgentLastInfos:
    def __init__(self):
        self.agent_last_infos = {}

    def get(self, agentid):
        if self.agent_last_infos.get(agentid):
            return self.agent_last_infos[agentid]
        else:
            self.agent_last_infos[agentid] = AgentLastInfo()
            return self.agent_last_infos[agentid]


class SACTrainer(RLTrainer):
    """
    The SACTrainer is an implementation of the SAC algorithm, with support
    for discrete actions and recurrent networks.
    """

    def __init__(
        self, brain, reward_buff_cap, trainer_parameters, training, load, seed, run_id
    ):
        """
        Responsible for collecting experiences and training SAC model.
        :param trainer_parameters: The parameters for the trainer (dictionary).
        :param training: Whether the trainer is set for training.
        :param load: Whether the model should be loaded.
        :param seed: The seed the model will be initialized with
        :param run_id: The The identifier of the current run
        """
        super().__init__(brain, trainer_parameters, training, run_id, reward_buff_cap)
        self.param_keys = [
            "batch_size",
            "buffer_size",
            "buffer_init_steps",
            "hidden_units",
            "learning_rate",
            "init_entcoef",
            "max_steps",
            "normalize",
            "num_update",
            "num_layers",
            "time_horizon",
            "sequence_length",
            "summary_freq",
            "tau",
            "use_recurrent",
            "summary_path",
            "memory_size",
            "model_path",
            "reward_signals",
            "vis_encode_type",
        ]

        self.check_param_keys()

        self.step = 0
        self.train_interval = (
            trainer_parameters["train_interval"]
            if "train_interval" in trainer_parameters
            else 1
        )
        self.reward_signal_updates_per_train = (
            trainer_parameters["reward_signals"]["reward_signal_num_update"]
            if "reward_signal_num_update" in trainer_parameters["reward_signals"]
            else trainer_parameters["num_update"]
        )

        self.checkpoint_replay_buffer = (
            trainer_parameters["save_replay_buffer"]
            if "save_replay_buffer" in trainer_parameters
            else False
        )
        self.policy = SACPolicy(seed, brain, trainer_parameters, self.is_training, load)
        self.training_buffer = PriorityBuffer(
            max_size=self.trainer_parameters["buffer_size"]
        )
        self.agent_last_infos = AgentLastInfos()

        # Load the replay buffer if load
        # if load and self.checkpoint_replay_buffer:
        #     try:
        #         self.load_replay_buffer()
        #     except (AttributeError, FileNotFoundError):
        #         LOGGER.warning(
        #             "Replay buffer was unable to load, starting from scratch."
        #         )
        #     LOGGER.debug(
        #         "Loaded update buffer with {} sequences".format(
        #             len(self.training_buffer.update_buffer["actions"])
        #         )
        #     )

        for _reward_signal in self.policy.reward_signals.keys():
            self.collected_rewards[_reward_signal] = {}

        self.episode_steps = {}

    def save_model(self) -> None:
        """
        Saves the model. Overrides the default save_model since we want to save
        the replay buffer as well.
        """
        self.policy.save_model(self.get_step)
        # if self.checkpoint_replay_buffer:
        #     self.save_replay_buffer()

    # def save_replay_buffer(self) -> None:
    #     """
    #     Save the training buffer's update buffer to a pickle file.
    #     """
    #     filename = os.path.join(self.policy.model_path, "last_replay_buffer.hdf5")
    #     LOGGER.info("Saving Experience Replay Buffer to {}".format(filename))
    #     with open(filename, "wb") as file_object:
    #         self.training_buffer.update_buffer.save_to_file(file_object)
    #
    # def load_replay_buffer(self) -> Buffer:
    #     """
    #     Loads the last saved replay buffer from a file.
    #     """
    #     filename = os.path.join(self.policy.model_path, "last_replay_buffer.hdf5")
    #     LOGGER.info("Loading Experience Replay Buffer from {}".format(filename))
    #     with open(filename, "rb+") as file_object:
    #         self.training_buffer.update_buffer.load_from_file(file_object)
    #     LOGGER.info(
    #         "Experience replay buffer has {} experiences.".format(
    #             len(self.training_buffer.update_buffer["actions"])
    #         )
    #     )

    def add_rewards_outputs(
        self,
        rewards_out: AllRewardsOutput,
        values: Dict[str, np.ndarray],
        agent_id: str,
        agent_idx: int,
        agent_next_idx: int,
    ) -> float:
        """
        Takes the value output of the last action and store it into the training buffer.
        """
        return rewards_out.environment[agent_next_idx]

    def construct_curr_info(self, next_info: BrainInfo) -> BrainInfo:
        """
        Constructs a BrainInfo which contains the most recent previous experiences for all agents
        which correspond to the agents in a provided next_info.
        :BrainInfo next_info: A t+1 BrainInfo.
        :return: curr_info: Reconstructed BrainInfo to match agents of next_info.
        """
        visual_observations: List[List[Any]] = [
            []
        ]  # TODO add types to brain.py methods
        vector_observations = []
        text_observations = []
        memories = []
        rewards = []
        local_dones = []
        max_reacheds = []
        agents = []
        prev_vector_actions = []
        prev_text_actions = []
        action_masks = []
        for agent_id in next_info.agents:
            agent_brain_info = self.agent_last_infos.get(agent_id).last_brain_info
            if agent_brain_info is None:
                agent_brain_info = next_info
            agent_index = agent_brain_info.agents.index(agent_id)
            for i in range(len(next_info.visual_observations)):
                visual_observations[i].append(
                    agent_brain_info.visual_observations[i][agent_index]
                )
            vector_observations.append(
                agent_brain_info.vector_observations[agent_index]
            )
            text_observations.append(agent_brain_info.text_observations[agent_index])
            if self.policy.use_recurrent:
                if len(agent_brain_info.memories) > 0:
                    memories.append(agent_brain_info.memories[agent_index])
                else:
                    memories.append(self.policy.make_empty_memory(1))
            rewards.append(agent_brain_info.rewards[agent_index])
            local_dones.append(agent_brain_info.local_done[agent_index])
            max_reacheds.append(agent_brain_info.max_reached[agent_index])
            agents.append(agent_brain_info.agents[agent_index])
            prev_vector_actions.append(
                agent_brain_info.previous_vector_actions[agent_index]
            )
            prev_text_actions.append(
                agent_brain_info.previous_text_actions[agent_index]
            )
            action_masks.append(agent_brain_info.action_masks[agent_index])
        if self.policy.use_recurrent:
            memories = np.vstack(memories)
        curr_info = BrainInfo(
            visual_observations,
            vector_observations,
            text_observations,
            memories,
            rewards,
            agents,
            local_dones,
            prev_vector_actions,
            prev_text_actions,
            max_reacheds,
            action_masks,
        )
        return curr_info

    def add_experiences(
        self,
        curr_all_info: AllBrainInfo,
        next_all_info: AllBrainInfo,
        take_action_outputs: ActionInfoOutputs,
    ) -> None:
        """
        Adds experiences to each agent's experience history.
        :param curr_all_info: Dictionary of all current brains and corresponding BrainInfo.
        :param next_all_info: Dictionary of all current brains and corresponding BrainInfo.
        :param take_action_outputs: The outputs of the Policy's get_action method.
        """
        self.trainer_metrics.start_experience_collection_timer()
        if take_action_outputs:
            self.stats["Policy/Entropy"].append(take_action_outputs["entropy"].mean())
            self.stats["Policy/Learning Rate"].append(
                take_action_outputs["learning_rate"]
            )
            for name, signal in self.policy.reward_signals.items():
                self.stats[signal.value_name].append(
                    np.mean(take_action_outputs["value_heads"][name])
                )

        curr_info = curr_all_info[self.brain_name]
        next_info = next_all_info[self.brain_name]

        for agent_id in curr_info.agents:
            self.agent_last_infos.get(agent_id).last_brain_info = curr_info
            self.agent_last_infos.get(
                agent_id
            ).last_take_action_outputs = take_action_outputs

        if curr_info.agents != next_info.agents:
            curr_to_use = self.construct_curr_info(next_info)
        else:
            curr_to_use = curr_info

        # Evaluate and store the reward signals
        tmp_reward_signal_outs = {}
        for name, signal in self.policy.reward_signals.items():
            tmp_reward_signal_outs[name] = signal.evaluate(curr_to_use, next_info)
        # Store the environment reward
        tmp_environment = np.array(next_info.rewards)

        rewards_out = AllRewardsOutput(
            reward_signals=tmp_reward_signal_outs, environment=tmp_environment
        )

        for agent_id in next_info.agents:
            stored_info = self.agent_last_infos.get(agent_id).last_brain_info
            stored_take_action_outputs = self.agent_last_infos.get(
                agent_id
            ).last_take_action_outputs
            if stored_info is not None:
                idx = stored_info.agents.index(agent_id)
                next_idx = next_info.agents.index(agent_id)
                if not stored_info.local_done[idx]:
                    new_exp = {}
                    for i, _ in enumerate(stored_info.visual_observations):
                        new_exp["visual_obs%d" % i] = stored_info.visual_observations[
                            i
                        ][idx]
                        new_exp[
                            "next_visual_obs%d" % i
                        ] = next_info.visual_observations[i][next_idx]
                    if self.policy.use_vec_obs:
                        new_exp["vector_obs"] = stored_info.vector_observations[idx]
                        new_exp["next_vector_in"] = next_info.vector_observations[
                            next_idx
                        ]
                    if self.policy.use_recurrent:
                        if stored_info.memories.shape[1] == 0:
                            stored_info.memories = np.zeros(
                                (len(stored_info.agents), self.policy.m_size)
                            )
                        new_exp["memory"] = stored_info.memories[idx]

                    new_exp["masks"] = 1.0
                    new_exp["done"] = next_info.local_done[idx]
                    # Add the outputs of the last eval
                    actions = stored_take_action_outputs["action"]
                    new_exp["actions"] = actions[idx]
                    # Store action masks if neccessary
                    if not self.policy.use_continuous_act:
                        new_exp["action_mask"] = stored_info.action_masks[idx]
                    new_exp["prev_action"] = stored_info.previous_vector_actions[idx]

                    values = stored_take_action_outputs["value_heads"]

                    # Add the value outputs if needed
                    new_exp["environment_rewards"] = self.add_rewards_outputs(
                        rewards_out, values, agent_id, idx, next_idx
                    )

                    for name, rewards in self.collected_rewards.items():
                        if agent_id not in rewards:
                            rewards[agent_id] = 0
                        if name == "environment":
                            # Report the reward from the environment
                            rewards[agent_id] += rewards_out.environment[next_idx]
                            new_exp[
                                "{}_rewards".format(name)
                            ] = rewards_out.environment[next_idx]
                        else:
                            new_exp[
                                "{}_rewards".format(name)
                            ] = rewards_out.reward_signals[name].scaled_reward[next_idx]
                            # Report the reward signals
                            rewards[agent_id] += rewards_out.reward_signals[
                                name
                            ].scaled_reward[next_idx]
                    q1_losses = self.policy.calculate_loss([new_exp])
                    self.training_buffer.add([new_exp], q1_losses)
                if not next_info.local_done[next_idx]:
                    if agent_id not in self.episode_steps:
                        self.episode_steps[agent_id] = 0
                    self.episode_steps[agent_id] += 1
        self.trainer_metrics.end_experience_collection_timer()

    def process_experiences(
        self, current_info: AllBrainInfo, new_info: AllBrainInfo
    ) -> None:
        """
        Checks agent histories for processing condition, and processes them as necessary.
        Processing involves calculating value and advantage targets for model updating step.
        :param current_info: Dictionary of all current brains and corresponding BrainInfo.
        :param new_info: Dictionary of all next brains and corresponding BrainInfo.
        """
        info = new_info[self.brain_name]
        for l in range(len(info.agents)):
            # agent_actions = self.training_buffer[info.agents[l]]["actions"]
            if (
                info.local_done[l]
                # or len(agent_actions) >= self.trainer_parameters["time_horizon"]
            ):  # and len(agent_actions) > 0:
                agent_id = info.agents[l]

                # self.training_buffer.append_update_buffer(
                #     agent_id,
                #     batch_size=None,
                #     training_length=self.policy.sequence_length,
                # )
                #
                # self.training_buffer[agent_id].reset_agent()
                if info.local_done[l]:
                    self.stats["Environment/Episode Length"].append(
                        self.episode_steps.get(agent_id, 0)
                    )
                    self.episode_steps[agent_id] = 0
                    for name, rewards in self.collected_rewards.items():
                        if name == "environment":
                            self.cumulative_returns_since_policy_update.append(
                                rewards.get(agent_id, 0)
                            )
                            self.stats["Environment/Cumulative Reward"].append(
                                rewards.get(agent_id, 0)
                            )
                            self.reward_buffer.appendleft(rewards.get(agent_id, 0))
                            rewards[agent_id] = 0
                        else:
                            self.stats[
                                self.policy.reward_signals[name].stat_name
                            ].append(rewards.get(agent_id, 0))
                            rewards[agent_id] = 0

    def is_ready_update(self) -> bool:
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to whether or not update_model() can be run
        """
        return (
            len(self.training_buffer) >= self.trainer_parameters["batch_size"]
            and self.step >= self.trainer_parameters["buffer_init_steps"]
        )

    @timed
    def update_policy(self) -> None:
        """
        If train_interval is met, update the SAC policy given the current reward signals.
        If reward_signal_train_interval is met, update the reward signals from the buffer.
        """
        if self.step % self.train_interval == 0:
            self.trainer_metrics.start_policy_update_timer(
                number_experiences=len(self.training_buffer),
                mean_return=float(np.mean(self.cumulative_returns_since_policy_update)),
            )
            self.update_sac_policy()
            self.update_reward_signals()
            self.trainer_metrics.end_policy_update()

    def update_sac_policy(self) -> None:
        """
        Uses demonstration_buffer to update the policy.
        The reward signal generators are updated using different mini batches.
        If we want to imitate http://arxiv.org/abs/1809.02925 and similar papers, where the policy is updated
        N times, then the reward signals are updated N times, then reward_signal_updates_per_train
        is greater than 1 and the reward signals are not updated in parallel.
        """

        self.cumulative_returns_since_policy_update: List[float] = []
        n_sequences = max(
            int(self.trainer_parameters["batch_size"] / self.policy.sequence_length), 1
        )

        num_updates = self.trainer_parameters["num_update"]
        batch_update_stats: Dict[str, list] = defaultdict(list)
        for _ in range(num_updates):
            LOGGER.debug("Updating SAC policy at step {}".format(self.step))
            buffer = self.training_buffer
            if len(buffer) >= self.trainer_parameters["batch_size"]:
                sampled_minibatch, batch_priorities, batch_is_weights = buffer.get_batch(
                    self.trainer_parameters["batch_size"],
                    # sequence_length=self.policy.sequence_length,
                )
                buffer.update_last_batch(batch_priorities * 0.95)
                sampled_minibatch["is_weights"] = batch_is_weights
                # Get rewards for each reward
                for name, signal in self.policy.reward_signals.items():
                    sampled_minibatch[
                        "{}_rewards".format(name)
                    ] = signal.evaluate_batch(sampled_minibatch).scaled_reward

                update_stats = self.policy.update(
                    sampled_minibatch, n_sequences, update_target=True
                )
                for stat_name, value in update_stats.items():
                    batch_update_stats[stat_name].append(value)

        # Truncate update buffer if neccessary. Truncate more than we need to to avoid truncating
        # a large buffer at each update.
        # if (
        #     len(self.training_buffer.update_buffer["actions"])
        #     > self.trainer_parameters["buffer_size"]
        # ):
        #     self.training_buffer.truncate_update_buffer(
        #         int(self.trainer_parameters["buffer_size"] * BUFFER_TRUNCATE_PERCENT)
        #     )

        for stat, stat_list in batch_update_stats.items():
            self.stats[stat].append(np.mean(stat_list))

        if self.policy.bc_module:
            update_stats = self.policy.bc_module.update()
            for stat, val in update_stats.items():
                self.stats[stat].append(val)

    def update_reward_signals(self) -> None:
        """
        Iterate through the reward signals and update them. Unlike in PPO,
        do it separate from the policy so that it can be done at a different
        interval.
        This function should only be used to simulate
        http://arxiv.org/abs/1809.02925 and similar papers, where the policy is updated
        N times, then the reward signals are updated N times. Normally, the reward signal
        and policy are updated in parallel.
        """
        buffer = self.training_buffer
        num_updates = self.reward_signal_updates_per_train
        n_sequences = max(
            int(self.trainer_parameters["batch_size"] / self.policy.sequence_length), 1
        )
        batch_update_stats: Dict[str, list] = defaultdict(list)
        for _ in range(num_updates):
            # Get minibatches for reward signal update if needed
            reward_signal_minibatches = {}
            for name, signal in self.policy.reward_signals.items():
                LOGGER.debug("Updating {} at step {}".format(name, self.step))
                # Some signals don't need a minibatch to be sampled - so we don't!
                if signal.update_dict:
                    reward_signal_minibatches[
                        name
                    ], priorities, is_weights = buffer.get_batch(
                        self.trainer_parameters["batch_size"],
                        # sequence_length=self.policy.sequence_length,
                    )
                    update_stats = self.policy.update_reward_signals(
                        reward_signal_minibatches, n_sequences
                    )
                    for stat_name, value in update_stats.items():
                        batch_update_stats[stat_name].append(value)
        for stat, stat_list in batch_update_stats.items():
            self.stats[stat].append(np.mean(stat_list))
