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

from dm_control import suite
import jitterbug_dmc
from benchmarks.keras_rl_processor import JitterbugProcessor


def train_ddpg_agent(*, task="move_from_origin", random_seed=None):
    """Main

    Args:
        random_seed (int): Random seed
    """

    np.random.seed(random_seed)

    # Load the environment
    env = jitterbug_dmc.JitterbugGymEnv(
        suite.load(
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
    )

    num_actions = env.action_space.shape[0]
    num_observations = env.observation_space.spaces['observations'].shape[0]
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
    print("Actor Network:")
    print(actor.summary())
    print()

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
    print("Critic Network:")
    print(critic.summary())
    print()

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

    # if os.path.exists(model_weights_path):
    #     # Continue training
    #     print(f"Loading weights from {model_weights_path}")
    #     agent.load_weights(model_weights_path)
    #
    #     print("Continuing training...")

    # Train the agent
    agent.fit(
        env,
        nb_steps=int(1e8),
        visualize=False,
        verbose=1
    )

    # Save the final weights
    agent.save_weights(model_weights_path, overwrite=True)

    # Finally, evaluate our agent
    agent.load_weights(model_weights_path)
    agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=1000)


if __name__ == '__main__':

    # Train an agent
    train_ddpg_agent(
        task="face_direction",
        random_seed=123
    )
