"""Evaluates various RL algorithms on the Jitterbug task suite"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Uncomment to disable GPU training in tensorflow (must be before keras imports)
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
sys.path.append("/home/jeremy/jitterbug-dmc")

import jitterbug_dmc
import benchmarks

# Baselines module
from baselines.run import build_env
from baselines.a2c.a2c import Model as A2CModel
from baselines.common.policies import build_policy
from baselines.common.cmd_util import make_mujoco_env
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

from baselines.common.cmd_util import (
    common_arg_parser,
    parse_unknown_args,
    make_vec_env,
    make_env
)

from gym.envs.registration import register, make

from stable_baselines.a2c.a2c import A2C
from stable_baselines.common.policies import (
    FeedForwardPolicy,
    ActorCriticPolicy
)
from stable_baselines.results_plotter import load_results, ts2xy


class JitterbugDDPGAgent(DDPGAgent):
    """A DDPG agent for the Jitterbug task"""

    def __init__(self, num_observations):
        """Constructor

        Builds the actor and critic networks, sets hyper-parameters and compiles
        the agent
        """

        num_actions = 1
        print(target_model_update)
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


class JitterbugA2CAgent(A2C):
    """An A2C agent for the Jitterbug task"""

    def __init__(self, policy, env, policy_kwargs=None):

        # Make the environment compatible with stable_baselines package
        env_gym = jitterbug_dmc.JitterbugGymEnv(env)
        env_gym.num_envs = 1
        env_gym.observation_space = env_gym.observation_space["observations"]
        env_vec = DummyVecEnv([lambda: env_gym])

        super().__init__(
            policy=policy,
            env=env_vec,
            policy_kwargs=policy_kwargs
        )


    def train(self, nb_steps, callback=None):
        """Train the A2C agent.

        Args:
            nb_steps (int): total number of steps used for training
        """

        self.learn(total_timesteps=nb_steps,
            #callback=callback
        )


def train_ddpg_agent(
        env,
        agent,
        weights_path,
        training_progress_path,
        *,
        num_steps=int(2e6)
):
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
            benchmarks.AgentCheckpointCallback(
                agent,
                weights_path,
                training_progress_path
            )
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
        env,
        policy
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


def demo(task, *, random_seed=123):
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
    agent_training_progress_path = f"ddpg.{task}.training_progress.pkl"
    agent = JitterbugDDPGAgent(
        num_observations=env.observation_spec()['observations'].shape[0]
    )

    # # Load pre-trained agent weights
    # if os.path.exists(agent_weights_path):
    #     agent.load_weights(agent_weights_path)

    # Train the agent
    train_ddpg_agent(
        env,
        agent,
        agent_weights_path,
        agent_training_progress_path
    )

    # Load pre-trained agent weights
    agent.load_weights(agent_weights_path)

    # Evaluate the trained agent
    eval_ddpg_agent(env, agent, visualise=True)

    print("Done")


# Create log dir
log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)
best_mean_reward, n_steps= -np.inf, 0


def callback(_locals, _globals):
  """
  Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
  :param _locals: (dict)
  :param _globals: (dict)
  """
  
  global n_steps, best_mean_reward
  #print(n_steps)
  #print(n_steps)
  # Print stats every 1000 calls
  if (n_steps + 1) % 1000 == 0:
      # Evaluate policy training performance
      x, y = ts2xy(load_results(log_dir), 'timesteps')
      print(x)
      if len(x) > 0:
          mean_reward = np.mean(y[-100:])
          print(x[-1], 'timesteps')
          print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

          # New best model, you could save the agent here
          if mean_reward > best_mean_reward:
              best_mean_reward = mean_reward
              # Example for saving best model
              print("Saving new best model")
              _locals['self'].save(log_dir + 'best_model.pkl')
  n_steps += 1
  return True


def demoA2C(task, *, random_seed=123):
    """Train and evaluate A2C agent"""

    random_seed=123

    # Load the environment
    np.random.seed(random_seed)

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

    # Define the architecture of the NN
    policy_kwargs = dict(act_fun=tf.nn.tanh, 
        net_arch=[32, 32]
        )
   
   # Construct the A2C agent
    agent = JitterbugA2CAgent(policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs
        )

    path_trained_agent = f"a2c.{task}.model_parameters.pkl"
    agent.train(
        1000,
        #callback=callback
    )

    agent.save(path_trained_agent)


if __name__ == '__main__':
    demoA2C("move_in_direction")
