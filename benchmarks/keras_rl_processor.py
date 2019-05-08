"""A Keras-RL processor class for the Jitterbug domain"""

import numpy as np

from rl.processors import WhiteningNormalizerProcessor


class JitterbugProcessor(WhiteningNormalizerProcessor):
    """A processor to convert Jitterbug observations and actions"""

    # Observation wrapper is not needed if flat_observation=True is passed to
    # the environment constructor
    # def process_observation(self, observation):
    #     """Convert the Dict space observation to a flat vector"""
    #     return np.concatenate([v for v in observation.values()], axis=0)

    def process_action(self, action):
        """Clip actions"""
        return np.clip(action, -1.0, 1.0)
