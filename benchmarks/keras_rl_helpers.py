"""Various support classes for using jitterbug-dmc with keras-rl"""

import pickle
import collections
import numpy as np

from rl.processors import WhiteningNormalizerProcessor
from rl.callbacks import Callback


class JitterbugProcessor(WhiteningNormalizerProcessor):
    """A processor to convert Jitterbug `things` to keras-rl format"""

    def process_observation(self, observation):
        """Convert the Dict space observation to a flat vector"""
        if isinstance(observation, collections.OrderedDict):
            # Observation dict has already been flattened by dm_control
            return observation['observations']
        else:
            # Manually flatten observation dict
            return np.concatenate([v for v in observation.values()], axis=0)

    def process_reward(self, reward):
        """DeepMind Control returns numpy arrays for reward, we want float"""
        return reward[0]

    def process_action(self, action):
        """Clip actions"""
        return np.clip(action, -1.0, 1.0)


class AgentCheckpointCallback(Callback):
    """A keras-rl callback class to save the agent weights during training"""

    def __init__(self, agent, model_weights_path):
        """C-tor"""
        self.agent = agent
        self.model_weights_path = model_weights_path
        self.training_progress_path = "{}.trainingepisoderewards.pkl".format(
            model_weights_path
        )
        self.episode_rewards = []
        super().__init__()

    def save(self):
        """Save the agent weights and training progress"""
        self.agent.save_weights(self.model_weights_path, overwrite=True)
        with open(self.training_progress_path, "wb") as file:
            pickle.dump(self.episode_rewards, file)

    def on_episode_end(self, episode, logs):
        """Save the agent at the end of every episode"""
        self.episode_rewards.append(logs['episode_reward'])
        self.save()

    def on_train_end(self, logs):
        """Save the agent at the end of training"""
        self.save()
