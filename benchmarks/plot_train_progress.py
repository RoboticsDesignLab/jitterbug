
import os
import sys
import csv
import glob
import warnings
import numpy as np
from pprint import pprint
from matplotlib import pyplot as plt


def ma(data, window, *, mode='valid'):
    """Moving average filter

    Args:
        data (numpy array): The data to be smoothed
        window (int): The window used to smooth the data
        mode (str): The mode used to smooth the data
    Returns:
        data_smoothed (numpy array): The data smoothed by the moving average
    """
    N = len(data)
    data_smoothed = np.zeros(N)
    mask = np.ones(window)/window
    
    data_smoothed = np.convolve(data, mask, mode)

    return data_smoothed


def get_reward_from_csv(file):
    """Extract reward vector from a csv file"""
    reward = []

    print("Reading reward from {}".format(file))

    with open(file, 'rt') as f:
        data = csv.reader(f)
        for ri, row in enumerate(data):

            if ri < 2:
                # Skip first two rows
                continue

            if len(row) == 0:
                # Skip empty rows
                continue

            """
            XXX ajs 11/Sep/2019 There seems to be bug with
            csv.DictWriter.writerow() as used in
            stable_baselines/bench/monitor.py:98 - we occasionally see reward
            values that are missing the mantissa, e.g.
            'XXX.XXe-5' will simply appear as 'e-5' in the CSV cell.
            To work-around this, we detect these cases and replace with 0
            """
            if row[0].split("e")[0] == '':
                warnings.warn(
                    "Got malformed reward (no mantissa) "
                    "at line {} of {}: {}, replacing with 0.0".format(
                        ri,
                        file,
                        row[0]
                    )
                )
                reward.append(0.0)
                continue

            reward.append(float(row[0]))

    return np.array(reward, dtype=float)


def plot_csv_glob(fileglob, window, *, quartiles=[10, 50, 90], **kwargs):
    """Plot performance over many seeds into the current axes

    Args:
        fileglob (str): File path glob matching one or more monitor.csv files
        window (int): Moving average window

        kwargs (dict): Plotting keyword args
    """

    files = list(glob.glob(fileglob))

    if len(files) == 0:
        warnings.warn("Glob '{}' matched 0 files".format(fileglob))
        sys.exit(-1)

    print("Loading rewards from {} files".format(len(files)))
    rewards = [
        get_reward_from_csv(f)
        for f in files
    ]

    reward_len = [
        len(r)
        for r in rewards
    ]
    print("Reward lengths:")
    pprint(list(zip(files, reward_len)))
    min_len = min(reward_len)
    max_len = max(reward_len)
    print("Min training length: {}, max: {}".format(
        min_len,
        max_len
    ))

    # Crop all rewards to the minimum length
    rewards = np.array([
        r[0:min_len]
        for r in rewards
    ], dtype=float)
    x = np.arange(0, rewards.shape[1])

    # Smooth
    q1, q2, q3 = (
        ma(q, window)
        for q in np.percentile(rewards, q=quartiles, axis=0)
    )

    x = np.arange(window // 2, window // 2 + len(q1))

    p0 = plt.plot(x, q2, **kwargs)
    plt.fill_between(
        x,
        q1,
        q3,
        color=p0[0].get_color(),
        alpha=0.1,
        lw=0
    )

    return rewards


def main(fileglob, window, **kwargs):
    """Plot model training progress"""

    # Ensure we are using a GUI frontend so X-Forwarding works
    import matplotlib
    matplotlib.use('tkagg')

    plt.figure()
    r = plot_csv_glob(fileglob, window, **kwargs)
    plt.title("{} ({} files)".format(fileglob, len(r)))
    plt.xlabel("1e3 timesteps")
    plt.ylabel("Reward")
    plt.ylim(-50, 1050)
    plt.xlim(0, 6e6 / 1e3)
    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == '__main__':

    fileglob = sys.argv[1]
    window = int(sys.argv[2])

    main(fileglob, window)
