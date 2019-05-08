"""Evaluates various RL algorithms on the Jitterbug task suite"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Uncomment to disable GPU training in tensorflow (must be before keras imports)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from dm_control import suite
from dm_control import viewer

# Add root folder to path so we can access benchmarks module
import sys
sys.path.append("E:\\Development\\jitterbug-dmc")

import jitterbug_dmc
import benchmarks


class JitterbugDDPGAgent(DDPGAgent):
    """A DDPG agent for the Jitterbug task"""

    def __init__(self, num_observations):
        """Constructor

        Builds the actor and critic networks, sets hyper-parameters and compiles
        the agent
        """

        num_actions = 1

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

        # Call super-ctor
        super().__init__(
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
            processor=benchmarks.JitterbugProcessor()
        )

        # Compile immediately
        self._lr = 1e-4
        self.compile(
            [
                Adam(lr=self._lr),
                Adam(lr=self._lr)
            ],
            metrics=['mae']
        )

    def get_policy(self):
        """Get a policy function for use with dm_control environments"""

        if not self.compiled:
            raise RuntimeError(
                "Please compile the agent before extracting a policy"
            )

        self.training = False
        self.reset_states()

        def policy(ts):
            """A policy function closure for the given agent"""

            observation = ts.observation
            if self.processor is not None:
                observation = self.processor.process_observation(observation)

            action = self.forward(observation)
            if self.processor is not None:
                action = self.processor.process_action(action)

            return action

        return policy


def train_ddpg_agent(env, agent, weights_path, *, num_steps=int(1e8)):
    """Train a DDPG agent on the given dm_control environment

    Args:
        env (dm_control Environment): Environment to train on
        agent (keras-rl DDPG agent): DDPG agent to train
    """

    # Convert to gym interface
    env_gym = jitterbug_dmc.JitterbugGymEnv(env)

    # Train the agent
    agent.fit(
        env_gym,
        nb_steps=num_steps,
        visualize=False,
        verbose=1,
        callbacks=[
            benchmarks.AgentCheckpointCallback(agent, weights_path)
        ]
    )


def eval_ddpg_agent(env, agent, *, visualise=True):
    """Evaluate a trained agent"""

    # Get a policy function for the trained agent
    policy = agent.get_policy()

    # Preview the trained policy
    if visualise:
        viewer.launch(env, policy=policy)

    # Evaluate the trained policy
    rewards = benchmarks.evaluate_policy(
        "face_direction",
        policy,
        environment_kwargs={
            "flat_observation": True
        }
    )

    # Plot the results
    plt.figure(figsize=(9, 6))
    benchmarks.plot_policy_returns(rewards, label="DDPG")
    x = range(1, 1000 + 1)
    plt.plot(x, x, 'r--')
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative Reward")
    plt.title("DDPG Policy for {}".format(env.task.task))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def demo(*, random_seed=123, task="face_direction"):
    """Train and evaluate a DDPG agent"""

    np.random.seed(random_seed)

    # Load the environment
    env = suite.load(
        domain_name="jitterbug",
        task_name=task,
        visualize_reward=True,
        task_kwargs={
            "random": random_seed
        },

        # Important: the keras-rl DDPG agent needs flat observations
        environment_kwargs={
            "flat_observation": True
        }
    )

    # Construct a DDPG agent
    agent_weights_path = f"ddpg.{task}.weights.h5f"
    agent = JitterbugDDPGAgent(
        num_observations=env.observation_spec()['observations'].shape[0]
    )

    # Train the agent
    train_ddpg_agent(env, agent, agent_weights_path)

    # Load pre-trained agent weights
    agent.load_weights(agent_weights_path)

    # Evaluate the trained agent
    eval_ddpg_agent(env, agent, visualise=False)

    print("Done")

if __name__ == '__main__':
    demo()
