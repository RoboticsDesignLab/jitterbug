"""Evaluate RL algorithms on the Jitterbug task suite"""

import os
import sys
import gym
import time
import pprint
import random
import warnings
import multiprocessing

import numpy as np

# Suppress tensorflow deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=FutureWarning, module="tensorboard")

# Important: the below 3 imports must be in this order, or the program
# crashes under Ubuntu due to a protocol buffer version mismatch error
import tensorflow as tf
import stable_baselines
from dm_control import suite

# Import agents from stable_baselines
from stable_baselines.a2c.a2c import A2C
from stable_baselines.ppo2.ppo2 import PPO2
from stable_baselines.ddpg.ddpg import DDPG
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise

# Get some extra utilities
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.results_plotter import load_results, ts2xy

# Add root folder to path so we can access benchmarks module
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    ".."
))
import jitterbug_dmc


class CustomPolicyDDPG(stable_baselines.ddpg.policies.FeedForwardPolicy):
    """A DDPG specific FeedForward policy"""

    def __init__(self, *args, **kwargs):
        super(CustomPolicyDDPG, self).__init__(
            *args,
            **kwargs,
            layers=[350, 250],
            feature_extraction="mlp",
            act_fun=tf.nn.relu
        )


class CustomPolicyGeneral(stable_baselines.common.policies.FeedForwardPolicy):
    """A general FeedForward policy"""

    def __init__(self, *args, **kwargs):
        super(CustomPolicyGeneral, self).__init__(
            *args,
            **kwargs,
            net_arch=[350, 250],
            feature_extraction="mlp",
            act_fun=tf.nn.relu
        )


def train(
    task,
    alg,
    logdir,
    *,
    random_seed=None,
    num_steps=int(100e6),
    log_every=int(100e3),
    num_parallel=32,
    **kwargs
):
    """Train and evaluate an agent

    Args:
        task (str): Jitterbug task to train on
        alg (str): Algorithm to train, one of;
            - 'ddpg':
            - 'a2c':
            - 'ppo2':
        logdir (str): Logging directory

        random_seed (int): Random seed to use, or None
        num_steps (int): Number of training steps to train for
        log_every (int): Save and log progress every this many timesteps
        num_parallel (int): Number of parallel environments to run. Only used
            for A2C and PPO2.
    """

    assert alg in ('ddpg', 'a2c', 'ppo2'), "Invalid alg: {}".format(alg)

    # Cast args to types
    if random_seed is not None:
        random_seed = int(random_seed)
    else:
        random_seed = int(time.time())

    # Fix random seed
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Prepare the logging directory
    os.makedirs(logdir, exist_ok=True)

    print("Training {} on {} with seed {} for {} steps "
          "(log every {}), saving to {}".format(
        alg,
        task,
        random_seed,
        num_steps,
        log_every,
        logdir
    ))

    # Construct DMC env
    env_dmc = suite.load(
        domain_name="jitterbug",
        task_name=task,
        task_kwargs=dict(random=random_seed),
        environment_kwargs=dict(flat_observation=True)
    )

    # Convert DMC env to Gym env with logging
    env_gym = gym.wrappers.FlattenDictWrapper(
        Monitor(
            jitterbug_dmc.JitterbugGymEnv(env_dmc),
            logdir,
            allow_early_resets=True
        ),
        dict_keys=["observations"]
    )

    # Wrap gym env in a dummy parallel vector
    if False and (alg in ('a2c', 'ppo2') and num_parallel > multiprocessing.cpu_count()):
        warnings.warn("Number of parallel workers "
                      "({}) > CPU count ({}), setting to # CPUs".format(
            num_parallel,
            multiprocessing.cpu_count()
        ))
        print("Using {} parallel environments".format(num_parallel))
        num_parallel = multiprocessing.cpu_count()
        env_vec = SubprocVecEnv([lambda: env_gym for _ in range(num_parallel)])
    else:
        num_parallel = 1
        env_vec = DummyVecEnv([lambda: env_gym])

    def _cb(_locals, _globals):
        """Callback for during training"""

        if 'last_log' not in _cb.__dict__:
            _cb.last_log = -np.inf

        if isinstance(_locals['self'], DDPG):
            ep_r_hist = list(_locals['episode_rewards_history'])
        elif isinstance(_locals['self'], PPO2):
            ep_r_hist = [d['r'] for d in _locals['ep_info_buf']]
        else:
            raise ValueError("Invalid algorithm: {}".format(
                _locals['self']
            ))
        steps = 1000 * len(ep_r_hist)

        steps_since_last_log = steps - _cb.last_log
        if steps_since_last_log >= log_every:
            _cb.last_log = steps
            print("t={}, ep. r = {:.2f} last 5 ep. mean r = {:.2f}".format(
                steps,
                ep_r_hist[-1] if len(ep_r_hist) >= 1 else np.nan,
                np.mean(ep_r_hist[-5:])
            ))
            path = os.path.join(logdir, "model.{}.pkl".format(steps))
            print("Saving to {}".format(path))
            _locals['self'].save(path)

        return True

    if alg == 'ddpg':

        # Wrap to get a Gym env with some logging capabilities

        # Default parameters for DDPG
        kwargs.setdefault("normalize_returns", True)
        kwargs.setdefault("return_range", (0., 1.))
        kwargs.setdefault("normalize_observations", True)
        kwargs.setdefault("observation_range", (-5., 5.))

        kwargs.setdefault("batch_size", 256)

        kwargs.setdefault("actor_lr", 1e-4)
        kwargs.setdefault("critic_lr", 1e-4)

        kwargs.setdefault("buffer_size", 1000000)

        kwargs.setdefault("action_noise", OrnsteinUhlenbeckActionNoise(
            mean=np.array([0.3]),
            sigma=0.3,
            theta=0.15
        ))

        print("Constructing DDPG agent with settings:")
        pprint.pprint(kwargs)

        # Construct the agent
        agent = DDPG(
            policy=CustomPolicyDDPG,
            env=env_vec,
            verbose=1,
            **kwargs
        )

        # Train for a while (logging and saving checkpoints as we go)
        agent.learn(
            total_timesteps=num_steps,
            callback=_cb,
            log_interval=10
        )

    elif alg == 'a2c':

        kwargs.setdefault("learning_rate", 1e-4)
        kwargs.setdefault("n_steps", 256 // num_parallel)
        kwargs.setdefault("ent_coef", 0.01)                        ### ?????
        kwargs.setdefault("lr_schedule", 'linear')

        print("Constructing A2C agent with settings:")
        pprint.pprint(kwargs)

        agent = A2C(
            policy=CustomPolicyGeneral,
            env=env_vec,
            verbose=1,
            **kwargs
        )

        # Train for a while (logging and saving checkpoints as we go)
        agent.learn(
            total_timesteps=num_steps,
            callback=_cb,
            log_interval=10
        )

    elif alg == 'ppo2':

        kwargs.setdefault("learning_rate", 1e-4)
        kwargs.setdefault("n_steps", 256 // num_parallel)
        kwargs.setdefault("ent_coef", 0.01)                        ### ?????
        kwargs.setdefault("cliprange", 0.1)                        ### ?????

        print("Constructing PPO2 agent with settings:")
        pprint.pprint(kwargs)

        agent = PPO2(
            policy=CustomPolicyGeneral,
            env=env_vec,
            verbose=1,
            **kwargs
        )

        # Train for a while (logging and saving checkpoints as we go)
        agent.learn(
            total_timesteps=num_steps,
            callback=_cb,
            log_interval=100
        )

    else:
        raise ValueError("Invalid alg: {}".format(alg))

    # Save final model
    agent.save(os.path.join(logdir, 'model.final.pkl'))

    print("Done")


if __name__ == '__main__':

    import os
    import json
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--alg",
        type=str,
        choices=('ddpg', 'ppo2'),
        required=True,
        help="Algorithm to train"
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task to run"
    )

    parser.add_argument(
        "--logdir",
        type=str,
        required=False,
        default=".",
        help="Logging directory prefix"
    )

    parser.add_argument(
        "--kwargs",
        type=json.loads,
        required=False,
        default={},
        help="Agent keyword arguments"
    )

    args = parser.parse_args()

    train(alg=args.alg, task=args.task, logdir=args.logdir, **args.kwargs)

