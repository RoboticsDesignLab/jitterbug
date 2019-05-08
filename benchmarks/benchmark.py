"""Evaluates various RL algorithms on the Jitterbug task suite"""

import os
import numpy as np

# Use keras for models
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

# Use keras-rl for RL algorithms
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.callbacks import Callback

from dm_control import suite

# Add root folder to path
import sys
sys.path.append("E:\\Development\\jitterbug-dmc")

import jitterbug_dmc
from benchmarks.keras_rl_processor import JitterbugProcessor


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


def ddpg_agent_to_policy(agent):
    """Takes a DDPG agent and returns a policy for use with dmc environments"""

    if not agent.compiled:
        raise RuntimeError(
            "Please compile the agent before extracting a policy"
        )

    agent.training = False
    agent.reset_states()

    def policy(ts):
        """A policy function closure for the given agent"""

        observation = ts.observation
        if agent.processor is not None:
            observation = agent.processor.process_observation(observation)

        action = agent.forward(observation)
        if agent.processor is not None:
            action = agent.processor.process_action(action)

        return action

    return policy


def train_ddpg_agent(*, task="move_from_origin", random_seed=None):
    """Main

    Args:
        random_seed (int): Random seed
    """

    np.random.seed(random_seed)

    # Load the environment
    env_dmc = suite.load(
        domain_name="jitterbug",
        task_name=task,
        visualize_reward=True,
        task_kwargs={
            "random": random_seed
        },
        environment_kwargs={
            "flat_observation": True
        }
    )
    env_gym = jitterbug_dmc.JitterbugGymEnv(env_dmc)

    num_actions = env_gym.action_space.shape[0]
    num_observations = env_gym.observation_space.spaces['observations'].shape[0]
    task_str = "jitterbug_{}".format(task)
    model_weights_path = 'ddpg.{}.weights.h5f'.format(task_str)

    # Build a DDPG agent

    # Build an actor network
    actor = Sequential()
    actor.add(Flatten(input_shape=(1, num_observations)))
    actor.add(Dense(300))
    actor.add(Activation('relu'))
    actor.add(Dense(200))
    actor.add(Activation('relu'))
    actor.add(Dense(num_actions))
    actor.add(Activation('tanh'))

    # Build a critic network
    action_input = Input(shape=(num_actions,), name='action_input')
    observation_input = Input(
        shape=(1, num_observations),
        name='observation_input'
    )
    flattened_observation = Flatten()(observation_input)
    x = Dense(400)(flattened_observation)
    x = Activation('relu')(x)
    x = Concatenate()([x, action_input])
    x = Dense(300)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)

    # Configure and compile the agent
    memory = SequentialMemory(limit=int(1e6), window_length=1)
    random_process = OrnsteinUhlenbeckProcess(
        size=num_actions,
        theta=0.15,
        mu=0.3,
        sigma=0.3
    )
    agent = DDPGAgent(
        nb_actions=num_actions,
        actor=actor,
        critic=critic,
        critic_action_input=action_input,
        memory=memory,
        batch_size=64,
        delta_clip=1,
        nb_steps_warmup_critic=1000,
        nb_steps_warmup_actor=1000,
        random_process=random_process,
        gamma=0.99,
        target_model_update=1e-3,
        processor=JitterbugProcessor()
    )
    lr = 1e-4
    agent.compile(
        [
            Adam(lr=lr),
            Adam(lr=lr)
        ],
        metrics=['mae']
    )

    if os.path.exists(model_weights_path):
        # Continue training
        print(f"Loading weights from {model_weights_path}")
        agent.load_weights(model_weights_path)

        print("Continuing training...")

    # # Train the agent
    # agent.fit(
    #     env_gym,
    #     #nb_steps=int(1e8),
    #     nb_steps=int(1e4),
    #     visualize=False,
    #     verbose=1,
    #     callbacks=[
    #         AgentCheckpointCallback(agent, model_weights_path)
    #     ]
    # )

    # Finally, test our agent
    agent.load_weights(model_weights_path)

    # Get a policy function for the trained agent
    policy = ddpg_agent_to_policy(agent)

    from dm_control import viewer
    import matplotlib.pyplot as plt
    from benchmarks import evaluate_policy, plot_policy_returns

    # Preview the trained policy
    viewer.launch(env_dmc, policy=policy)

    # Evaluate the trained policy
    rewards = evaluate_policy(
        "face_direction",
        policy,
        environment_kwargs={
            "flat_observation": True
        }
    )

    # Plot the results
    plt.figure(figsize=(9, 6))
    plot_policy_returns(rewards, label="DDPG")
    x = range(1, 1000 + 1)
    plt.plot(x, x, 'r--')
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative Reward")
    plt.title("DDPG Policy for {}".format(task))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def eval_ddpg_agent():
    pass


if __name__ == '__main__':

    # Train an agent
    train_ddpg_agent(
        task="face_direction",
        random_seed=123
    )
