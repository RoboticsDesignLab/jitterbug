from stable_baselines.ddpg.policies import FeedForwardPolicy
import tensorflow as tf

class CustomPolicy(FeedForwardPolicy):
	def __init__(self, *args, **kwargs):
         super(CustomPolicy, self).__init__(*args, **kwargs,
                                            layers=[300,300,300],
                                            feature_extraction="mlp",
 										    act_fun=tf.nn.relu)

