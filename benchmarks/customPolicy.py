from stable_baselines.common.policies import FeedForwardPolicy
import tensorflow as tf


class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(
            *args,
            **kwargs,
            net_arch=[256, 256, dict(vf=[256], pi=[256])],
            feature_extraction="mlp",
            act_fun=tf.nn.relu
        )
