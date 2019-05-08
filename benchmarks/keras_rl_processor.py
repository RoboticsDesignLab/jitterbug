"""A Keras-RL processor class for the Jitterbug domain"""

import collections
import numpy as np

from rl.processors import WhiteningNormalizerProcessor


class JitterbugProcessor(WhiteningNormalizerProcessor):
    """A processor to convert Jitterbug observations and actions"""

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
