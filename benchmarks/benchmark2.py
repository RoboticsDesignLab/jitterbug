"""Evaluates various RL algorithms on the Jitterbug task suite"""

import os

# Uncomment to disable GPU training in tensorflow (must be before keras imports)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from dm_control import suite

# Add root folder to path so we can access benchmarks module
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    ".."
))

import jitterbug_dmc

# stable-baselines modules
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.a2c.a2c import A2C
from stable_baselines.ppo2.ppo2 import PPO2
from stable_baselines.ddpg.ddpg import DDPG
from stable_baselines.common.policies import register_policy
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise

import gym
import numpy as np


def use_trained_agent(load_path,
                      env,
                      nb_steps=1000,
                      policy=None
                      ):
    """Use an Agent which is already trained to perform a task.

    Args:
        load_path (str): path where the parameters of the model are saved
        env (suite environment): environment
        nb_steps (int): number of steps used to render
        policy (None): policy. If the agent uses a custom policy, it has to
                       be passed explicitly to be able to load the model
        """
    env_vec = make_compatible_environment(env, "/tmp/gym/ddpg/")
    agent = DDPG.load(load_path=load_path, policy=policy, env=env_vec)
    obs = env_vec.reset()
    for i in range(nb_steps):
        action, _states = agent.predict(obs)
        obs, rewards, dones, info = env_vec.step(action)
        env_vec.render()


def make_compatible_environment(env, path):
    """Make a suite environment compatible with stable-baselines.

    Args:
        env (suite environment): environment
        path (str): path used to monitor the learning
        """
    env_gym = jitterbug_dmc.JitterbugGymEnv(env)
    env_gym.num_envs = 1
    log_dir = path
    os.makedirs(log_dir, exist_ok=True)
    env_mon = Monitor(env_gym, log_dir, allow_early_resets=True)
    env_flat = gym.wrappers.FlattenDictWrapper(env_mon, dict_keys=["observations"])
    env_vec = DummyVecEnv([lambda: env_flat])
    return env_vec


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """

    global n_steps_monitor, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps_monitor + 1) % 1000 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        # print(x)
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print(
                "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model.pkl')
    n_steps_monitor += 1
    return True


class JitterbugDDPGAgent(DDPG):
    """A DDPG agent for the Jitterbug task"""

    def __init__(self,
                 policy,
                 env,
                 verbose=1,
                 batch_size=64,
                 actor_lr=1e-4,
                 critic_lr=1e-4,
                 index=0):
        # Make the environment compatible with stable_baselines package
        log_dir = "/tmp/gym/ddpg/" + str(index) + "/"
        env_vec = make_compatible_environment(env, log_dir)

        # Noise
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.array([0.3]), sigma=0.3, theta=0.15)

        super().__init__(
            policy=policy,
            env=env_vec,
            verbose=verbose,
            batch_size=batch_size,
            action_noise=action_noise,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            memory_limit=int(1e6),
            normalize_observations=True
        )

    def train(self, nb_steps, callback=None):
        """Train the A2C agent.

        Args:
            nb_steps (int): total number of steps used for training
            callback (callable): callback function to monitor the learning process
        """
        self.learn(total_timesteps=nb_steps,
                   callback=callback
                   )


class JitterbugA2CAgent(A2C):
    """An A2C agent for the Jitterbug task"""

    def __init__(self,
                 policy,
                 env,
                 verbose=1,
                 max_grad_norm=0.5,
                 learning_rate=1e-4,
                 n_steps=16,
                 index=0):
        # Make the environment compatible with stable_baselines package
        log_dir = "/tmp/gym/a2c/" + str(index) + "/"
        env_vec = make_compatible_environment(env, log_dir)

        super().__init__(
            policy=policy,
            env=env_vec,
            n_steps=n_steps,
            verbose=verbose,
            learning_rate=learning_rate,
            lr_schedule="linear",
            ent_coef=0.001,
            max_grad_norm=max_grad_norm,
        )

    def train(self, nb_steps, callback=None):
        """Train the A2C agent.

        Args:
            nb_steps (int): total number of steps used for training
            callback (callable): callback function to monitor the learning process
        """
        self.learn(total_timesteps=nb_steps,
                   callback=callback
                   )


class JitterbugPPO2Agent(PPO2):
    """A PPO2 agent for the Jitterbug task"""

    def __init__(self,
                 policy,
                 env,
                 verbose=1,
                 n_steps=2048,
                 nminibatches=32,
                 noptepochs=10,
                 cliprange=0.1,
                 index=0):
        # Make the environment compatible with stable_baselines package
        log_dir = "/tmp/gym/ppo/" + str(index) + "/"
        env_vec = make_compatible_environment(env, log_dir)

        super().__init__(
            policy=policy,
            env=env_vec,
            verbose=verbose,
            n_steps=n_steps,
            nminibatches=nminibatches,
            noptepochs=noptepochs,
            ent_coef=0.001,
            cliprange=cliprange,
            # cliprange_vf=-1
        )

    def train(self, nb_steps, callback=None):
        """Train the A2C agent.

        Args:
            nb_steps (int): total number of steps used for training
        """

        self.learn(total_timesteps=nb_steps,
                   callback=callback
                   )

def demoDDPG(task,
             *,
             random_seed=123,
             batch_size=64,
             actor_lr=1e-4,
             critic_lr=1e-4,
             index=0
             ):
    """Train and evaluate DDPG agent"""
    from customPolicy_ddpg import CustomPolicy

    # Register the policy, it will check that the name is not already taken
    register_policy('CustomPolicy', CustomPolicy)

    random_seed = 123

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

    # Construct the DDPG agent
    agent = JitterbugDDPGAgent(policy='CustomPolicy',
                               env=env,
                               verbose=1,
                               batch_size=batch_size,
                               actor_lr=actor_lr,
                               critic_lr=critic_lr,
                               index=index
                               )

    #Train the DDPG agent
    agent.train(1e4,
    			callback=callback
    			)

    # Save the DDPG agent
    path_trained_agent = f"./trained_ddpg_{task}"
    # agent.save(path_trained_agent)

    # Use the DDPG agent
    #use_trained_agent("./ddpg-results/1/best_model",
    #                  env,
    #                  nb_steps=1000,
    #                  policy=CustomPolicy
    #                  )


def demoA2C(task,
            *,
            random_seed=123,
            max_grad_norm=0.5,
            learning_rate=1e-4,
            n_steps=16,
            index=0,
            n_envs=1
            ):
    """Train and evaluate A2C agent"""
    from customPolicy import CustomPolicy
    # Register the policy, it will check that the name is not already taken
    register_policy('CustomPolicy', CustomPolicy)

    random_seed = random_seed

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

    # Construct the A2C agent
    agent = JitterbugA2CAgent(policy=CustomPolicy,
                              env=env,
                              verbose=1,
                              max_grad_norm=max_grad_norm,
                              learning_rate=learning_rate,
                              n_steps=n_steps,
                              index=index
                              )

    agent.n_envs = n_envs
    agent.n_batch = agent.n_envs * agent.n_steps
    print(f"n_envs = {agent.n_envs}")
    print(f"n_batch = {agent.n_batch}")
    path_trained_agent = f"a2c.{task}.model_parameters.pkl"

    # Train the A2C agent
    agent.train(int(1e4),
                callback=callback
                )

    # Save the A2C agent
    path_trained_agent = f"./trained_a2c_{task}"
    # agent.save(path_trained_agent)

    # Use the A2C agent
    #use_trained_agent("./a2c-results/",
    #                  env,
    #                  nb_steps=1000,
    #                  policy=CustomPolicy
    #                 )


def demoPPO2(task,
             *,
             random_seed=123,
             n_steps=2048,
             index=0,
             nminibatches=32,
             noptepochs=10,
             cliprange=0.1,
             n_envs=1
            ):
    """Train and evaluate PPO2 agent"""
    from customPolicy import CustomPolicy
    # Register the policy, it will check that the name is not already taken
    register_policy('CustomPolicy', CustomPolicy)

    random_seed = random_seed

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

    # Construct the PPO2 agent
    agent = JitterbugPPO2Agent(policy='CustomPolicy',
                               env=env,
                               verbose=1,
                               n_steps=n_steps,
                               index=index,
                               nminibatches=nminibatches,
                               noptepochs=noptepochs,
                               cliprange=cliprange,
                               )

    agent.n_envs = n_envs
    agent.n_batch = agent.n_envs * agent.n_steps
    print(f"n_envs = {agent.n_envs}")
    print(f"n_batch = {agent.n_batch}")

    #Train the PPO2 agent
    agent.train(int(1e4),
    			callback=callback
    			)

    # Save the PPO2 agent
    path_trained_agent = f"./trained_ppo_{task}"
    # agent.save(path_trained_agent)

    # Use the PPO2 agent
    # use_trained_agent("./ppo-results/",
    #                  env,
    #                  nb_steps=1000,
    #                  policy=CustomPolicy
    #                 )


if __name__ == '__main__':
    i = 200
    n_steps_i = 32
    log_dir = "/tmp/gym/ddpg/" + str(i) + "/"
    os.makedirs(log_dir, exist_ok=True)
    best_mean_reward, n_steps_monitor = -np.inf, 0
    demoPPO2("face_direction",index=i)
