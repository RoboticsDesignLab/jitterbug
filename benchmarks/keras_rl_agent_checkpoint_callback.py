"""A keras-rl callback that saves agent weights during training"""

from rl.callbacks import Callback


class AgentCheckpointCallback(Callback):
    """A keras-rl callback class to save the agent weights during training"""

    def __init__(self, agent, model_weights_path):
        """C-tor"""
        self.agent = agent
        self.model_weights_path = model_weights_path
        super().__init__()

    def save(self):
        """Save the agent"""
        self.agent.save_weights(self.model_weights_path, overwrite=True)

    def on_episode_end(self, episode, logs):
        """Save the agent at the end of every episode"""
        self.save()

    def on_train_end(self, logs):
        """Save the agent at the end of training"""
        self.save()
