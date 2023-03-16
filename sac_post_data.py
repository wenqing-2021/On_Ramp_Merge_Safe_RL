import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

sac_cost = pd.read_csv('sac_averagecost.csv')
sac_reward = pd.read_csv('sac_averagereward.csv')

av_cost = sac_cost['SAC-0317 - AverageCost10'][:]
av_cost_safe = sac_cost['SAC-0317_safe - AverageCost10'][:]

plt.figure(1)
plt.plot(av_cost, label='SAC-Lagrangian')
plt.plot(av_cost_safe, label='SACL-MPC')
plt.ylim((0, 1))

plt.show()




