"""Evaluates various RL algorithms on the Jitterbug task suite"""



import os
import sys
import gym
import random
import numpy as np
from pprint import pprint

# Important: the below 3 imports must be in this order, or the program
# crashes under Ubuntu due to a protocol buffer version mismatch error
import tensorflow as tf
import stable_baselines
from dm_control import suite

from stable_baselines.a2c.a2c import A2C
from stable_baselines.ppo2.ppo2 import PPO2

from stable_baselines.ddpg.ddpg import DDPG
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise

from stable_baselines.trpo_mpi.trpo_mpi import TRPO

from stable_baselines.bench import Monitor
from stable_baselines.common.policies import register_policy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.results_plotter import load_results, ts2xy

# Add root folder to path so we can access benchmarks module
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    ".."
))
import jitterbug_dmc


# Globals
n_steps_monitor = 0
best_mean_reward = -np.inf


class CustomPolicyDDPG(stable_baselines.ddpg.policies.FeedForwardPolicy):
    """A DDPG specific FeedForward policy"""

    def __init__(self, *args, **kwargs):
        super(CustomPolicyDDPG, self).__init__(
            *args,
            **kwargs,
            layers=[300,300,300],
            feature_extraction="mlp",
            act_fun=tf.nn.relu
        )


class CustomPolicyGeneral(stable_baselines.common.policies.FeedForwardPolicy):
    """A general FeedForward policy"""

    def __init__(self, *args, **kwargs):
        super(CustomPolicyGeneral, self).__init__(
            *args,
            **kwargs,
            net_arch=[256, 256, dict(vf=[256], pi=[256])],
            feature_extraction="mlp",
            act_fun=tf.nn.relu
        )


def use_trained_agent(
    load_path,
    env,
    nb_steps=1000,
    nb_epochs=5000,
    monitor_path="/tmp/gym/",
    policy=None,
    render=False
):
    """Use an Agent which is already trained to perform a task.

    Args:
        load_path (str): path where the parameters of the model are saved
        env (suite environment): environment
        nb_steps (int): number of steps used to render
        nb_epochs (int): number of epochs
        monitor_path (str): path where the monitor saves the data gathered while performing the task
        policy (None): policy. If the agent uses a custom policy, it has to
                       be passed explicitly to be able to load the model
        render (bool): whether to show or not the agent training
        """
    env_vec = make_compatible_environment(env, monitor_path)
    agent = DDPG.load(load_path=load_path, policy=policy, env=env_vec)
    obs = env_vec.reset()
    for i in range(nb_epochs):
        for t in range(nb_steps):
            action, _states = agent.predict(obs)
            obs, rewards, dones, info = env_vec.step(action)
            if render:
                env_vec.render()
        print(str(nb_steps*(i+1))+" steps completed")


def make_compatible_environment(env, log_dir):
    """Make a suite environment compatible with stable-baselines.

    Args:
        env (suite environment): environment
        log_dir (str): path used to monitor the learning
        """
    env_gym = jitterbug_dmc.JitterbugGymEnv(env)
    env_gym.num_envs = 1
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

    def __init__(
        self,
        policy,
        env,
        verbose=1,
        batch_size=64,
        actor_lr=1e-4,
        critic_lr=1e-4,
        log_dir="."
    ):

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

    def __init__(
        self,
        policy,
        env,
        verbose=1,
        max_grad_norm=0.5,
        learning_rate=1e-4,
        n_steps=16,
        log_dir="."
    ):
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

    def __init__(
        self,
        policy,
        env,
        verbose=1,
        n_steps=2048,
        nminibatches=32,
        noptepochs=10,
        cliprange=0.1,
        log_dir="."
    ):

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


class JitterbugTRPOAgent(TRPO):
    """An A2C agent for the Jitterbug task"""

    def __init__(self,
                 policy,
                 env,
                 verbose=1,
                 log_dir=".",
                 cg_damping=1e-2,
                 cg_iters=10,
                 vf_stepsize=3e-4,
                 vf_iters=3,
                 lam=0.98,
                 entcoeff=0.0,
                 timesteps_per_batch=1024):

        env_vec = make_compatible_environment(env, log_dir)

        super().__init__(
            policy=policy,
            env=env_vec,
            verbose=verbose,
            cg_damping=cg_damping,
            cg_iters=cg_iters,
            vf_stepsize=vf_stepsize,
            vf_iters=vf_iters,
            lam=lam,
            entcoeff=entcoeff,
            timesteps_per_batch=timesteps_per_batch
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


def demoDDPG(
    task,
    log_dir,
    *,
    random_seed=None,
    batch_size=64,
    actor_lr=1e-4,
    critic_lr=1e-4,
    path_autoencoder=None
):
    """Train and evaluate DDPG agent"""

    # Cast args to types
    if random_seed is not None:
        random_seed = int(random_seed)
    batch_size = int(batch_size)
    actor_lr = float(actor_lr)
    critic_lr = float(critic_lr)

    # Fix random seed
    random.seed(random_seed)
    np.random.seed(random_seed)

    env = suite.load(
        domain_name="jitterbug",
        task_name=task,
        visualize_reward=True,
        #task_kwargs={
        #    "random": random_seed
        #},

        # Important: the keras-rl DDPG agent needs flat observations
        environment_kwargs={
            "flat_observation": True,
        }
    )

    # Construct the DDPG agent
    agent = JitterbugDDPGAgent(
        policy=CustomPolicyDDPG,
        env=env,
        verbose=1,
        batch_size=batch_size,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        log_dir=log_dir
    )

    # Train the DDPG agent
    agent.train(
        20e6,
        callback=callback
    )

    # Save the DDPG agent
    if path_autoencoder != None:
        env.task.jitterbug_autoencoder.save_autoencoder(path_autoencoder)

    path_trained_agent = f"./trained_ddpg_{task}"
    # agent.save(path_trained_agent)

    # Use the DDPG agent
    #use_trained_agent(load_path="./ddpg-results/6/best_model",
     #                 env=env,
     #                 nb_steps=1000,
     #                 policy=CustomPolicy,
     #                 monitor_path="/tmp/ddpg/13/",
     #                 render=False,
     #                 nb_epochs=5000
     #                 )


def demoA2C(
    task,
    log_dir,
    *,
    random_seed=None,
    max_grad_norm=0.5,
    learning_rate=1e-4,
    n_steps=16,
    n_envs=1
):
    """Train and evaluate A2C agent"""

    # Cast args to types
    if random_seed is not None:
        random_seed = int(random_seed)
    max_grad_norm = float(max_grad_norm)
    learning_rate = float(learning_rate)
    n_steps = int(n_steps)
    n_envs = int(n_envs)

    random.seed(random_seed)
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
    agent = JitterbugA2CAgent(policy=CustomPolicyGeneral,
                              env=env,
                              verbose=1,
                              max_grad_norm=max_grad_norm,
                              learning_rate=learning_rate,
                              n_steps=n_steps,
                              log_dir=log_dir
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


def demoPPO2(
    task,
    log_dir,
    *,
    random_seed=None,
    n_steps=2048,
    nminibatches=32,
    noptepochs=10,
    cliprange=0.1,
    n_envs=1
):
    """Train and evaluate PPO2 agent"""

    # Cast args to types
    if random_seed is not None:
        random_seed = int(random_seed)
    n_steps = int(n_steps)
    nminibatches = int(nminibatches)
    noptepochs = int(noptepochs)
    cliprange = float(cliprange)
    n_envs = int(n_envs)

    random.seed(random_seed)
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
    agent = JitterbugPPO2Agent(policy=CustomPolicyGeneral,
                               env=env,
                               verbose=1,
                               n_steps=n_steps,
                               log_dir=log_dir,
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


def demoTRPO(
    task,
    *,
    random_seed=None,
    cg_damping=1e-2,
    cg_iters=10,
    vf_stepsize=3e-4,
    vf_iters=3,
    lam=0.98,
    entcoeff=0.0,
    timesteps_per_batch=1024
):
    """Train and evaluate A2C agent"""

    # Cast args to types
    if random_seed is not None:
        random_seed = int(random_seed)
    cg_damping = float(cg_damping)
    cg_iters = int(cg_iters)
    vf_iters = int(vf_iters)
    lam = float(lam)
    entcoeff = float(entcoeff)
    timesteps_per_batch = int(timesteps_per_batch)

    random.seed(random_seed)
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

    # Construct the TRPO agent
    agent = JitterbugTRPOAgent(
        policy=CustomPolicyGeneral,
        env=env,
        verbose=1,
        log_dir=log_dir,
        cg_damping=cg_damping,
        cg_iters=cg_iters,
        vf_stepsize=vf_stepsize,
        vf_iters=vf_iters,
        lam=lam,
        entcoeff=entcoeff,
        timesteps_per_batch=timesteps_per_batch
    )

    path_trained_agent = f"a2c.{task}.model_parameters.pkl"

    # Train the TRPO agent
    agent.train(int(5e6),
                callback=callback
                )

    # Save the TRPO agent
    path_trained_agent = f"./trained_dqn_{task}"
    # agent.save(path_trained_agent)

    # Use the TRPO agent
    #use_trained_agent("./dqn-results/",
    #                  env,
    #                  nb_steps=1000,
    #                  policy=CustomPolicy
    #                 )


if __name__ == '__main__':

    # First arg is function to call
    func = globals()[sys.argv[1]]

    # Second arg is task
    task = sys.argv[2]

    # Third arg is logging directory
    log_dir = sys.argv[3]
    os.makedirs(log_dir, exist_ok=True)

    print("Training {} on {}, logging to {}".format(
        func,
        task,
        log_dir
    ))

    # Remainder of args are keyword parameters
    kwargs = {
        k: v
        for k, v in zip(sys.argv[4::2], sys.argv[5::2])
    }
    print("Arguments are:")
    pprint(kwargs)

    func(
        task=task,
        log_dir=log_dir,
        **kwargs
    )
