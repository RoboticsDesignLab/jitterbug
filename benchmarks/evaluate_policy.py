"""Evaluate a policy on a Jitterbug task"""

import numpy as np

from dm_control import suite
import jitterbug_dmc


def evaluate_policy(task, policy, *, num_repeats=20, **kwargs):
    """Evaluate a policy many times on a given task

    Args:
        task (str): Jitterbug task string
        policy (function): Policy function

        num_repeats (int): Number of policy evaluations to return
        **kwargs: Optional extra arguments to pass to suite.load

    Returns:
        (numpy array): num_repeats x env._step_limit-1 numpy array of policy
            rewards attained
    """

    # Construct environment so we can query step limit
    env = suite.load(
        domain_name="jitterbug",
        task_name=task,
        **kwargs
    )

    print("Evaluating {} on {}".format(policy, task))

    results = np.empty((num_repeats, int(env._step_limit - 1)))
    for repeat in range(num_repeats):
        print("Run {} / {}".format(repeat+1, num_repeats))

        # Re-construct environment ensure a random new seed
        env = suite.load(
            domain_name="jitterbug",
            task_name=task,
            **kwargs
        )
        ts = env.reset()

        for i in range(int(env._step_limit - 1)):
            action = policy(ts)
            ts = env.step(action)
            results[repeat, i] = ts.reward

    return results

def demo():
    """Demo"""

    # Evaluate policy
    task = "face_direction"
    rewards = evaluate_policy(
        task,
        eval(f"jitterbug_dmc.heuristic_policies.{task}")
    )

    # Convert rewards to returns
    returns = np.cumsum(rewards, axis=1)

    # Find the 5th, Median and 95th percentiles
    lower, median, upper = np.percentile(returns, (5, 50, 95), axis=0)

    # Plot the results
    import matplotlib.pyplot as plt
    x = range(1, len(median) + 1)
    plt.figure(figsize=(9, 6))
    p1 = plt.plot(x, median, '-', label='Median')
    plt.fill_between(
        x,
        lower,
        upper,
        color=p1[0].get_color(),
        alpha=0.1,
        label='5th-95th Percentiles'
    )
    plt.plot(x, x, 'r--', label='Maximum')
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative Reward")
    plt.title("Heuristic Policy for {}".format(task))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{task}_heuristic_perf.png", dpi=600)
    plt.show()


if __name__ == '__main__':
    demo()



