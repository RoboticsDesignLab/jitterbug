
import sys
import csv
import numpy as np
from matplotlib import pyplot as plt


def simpleMovingAverage(data, window, mode):
    """Smooth the highly variable data by computing the simple moving
    average.

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


def main(file, window):
    """Plot model training progress"""

    reward = []
    with open(file, 'rt') as f:
        data = csv.reader(f)
        for ri, row in enumerate(data):

            if ri < 2:
                # Skip first two rows
                continue

            if len(row) == 0:
                # Skip empty rows
                continue

            reward.append(float(row[0]))

    y = np.array(reward, dtype=float)
    y_smoothed = simpleMovingAverage(
        y,
        window,
        'valid'
    )
    x_smoothed = np.array(list(range(0, len(y_smoothed))), dtype=int)
    x = np.array(range(
        -window//2,
        -window//2 + len(y)
    ))

    p0 = plt.plot(
        x_smoothed,
        y_smoothed
    )
    plt.plot(
        x,
        y,
        color=p0[0].get_color(),
        alpha=0.1,
        lw=1
    )

    plt.title(file)
    plt.xlabel("1e3 timesteps")
    plt.ylabel("Reward")
    plt.ylim(-50, 1050)
    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == '__main__':

    file = sys.argv[1]
    window = int(sys.argv[2])

    main(file, window)
