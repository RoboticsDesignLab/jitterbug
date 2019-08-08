import csv
from matplotlib import pyplot as plt
import numpy as np

def simpleMovingAverage(data,window,mode):
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


reward = []
with open('/tmp/gym/ddpg/10/monitor.csv','rt')as f:
#with open('./ddpg-results/1/monitor.csv','rt') as f:
  data = csv.reader(f)
  i = 0
  for row in data:
    if i>1:
        reward.append(float(row[0]))
    i+=1

#reward2 = []
#with open('./ddpg-results/4/monitor.csv','rt') as f:
#  data2 = csv.reader(f)
#  i = 0
#  for row in data2:
#    if i>1:
#        reward2.append(float(row[0]))
#    i+=1



reward_smoothed = simpleMovingAverage(reward,100,'valid')
x_smoothed = range(len(reward_smoothed))
x = range(len(reward))


#reward_smoothed2 = simpleMovingAverage(reward2,100,'valid')
#x_smoothed2 = range(len(reward_smoothed2))
#x2 = range(len(reward2))

#plt.plot(x,reward_smoothed,x2,reward_smoothed2)
p1 = plt.plot(x_smoothed,reward_smoothed, label="Input dimension = 16")
p2 = plt.plot(x, reward, color=p1[0].get_color(), alpha=0.1)
#p3 = plt.plot(x_smoothed2,reward_smoothed2, label="Input dimension = 8")
#p4 = plt.plot(x2,reward2, color=p3[0].get_color(), alpha=0.1)
plt.title("Training of a DDPG Jitterbug Agent for the move_in_direction task. Moving average window = 100")
plt.xlabel("1e3 timesteps")

plt.ylabel("Cumulative Reward")
#plt.legend()
plt.grid()
plt.show()
