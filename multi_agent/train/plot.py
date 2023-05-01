import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from bokeh.palettes import Category10 as palette
# from bokeh.palettes import Category20 as palette2

baselines = [122.01]
desired_space = [10]
sns.set_theme(style="whitegrid")

for baseline, ds in zip(baselines, desired_space):
  # baseline =118.65
  plt.figure(dpi=150)
  ax = plt.gca()
  for i in ["0", "02", "04" , "06", "08", "1"]:
    path = "/home/diana/journal/multi_agent/train/models/alpha_{alpha}/exit_time_{d}.npy".format(alpha=i, d=ds)
    # path = "/content/drive/MyDrive/Research/plots/alpha_{alpha}/exit_time_{d}.npy".format(alpha=i, d=ds)
    time = np.load(path)
    t_data = np.column_stack((time[:,13] - baseline, time[:,12]))

    df = pd.DataFrame(data = t_data, columns = ['time', 'Configuration'])
    sns.lineplot(data=df, x="Configuration", y='time', ax=ax)

  # ax.set_xlim(3, 10)
  # ax.set_ylim(-7, 0)
  # plt.tight_layout()
  ax.set_xlim(3, 10)
  plt.title("Change in the Group's Exiting Time per Configuration")
  plt.xlabel("Configuration \n (No. of AVs in queue)")
  plt.ylabel("Time, s")
  plt.savefig("/home/diana/journal/multi_agent/train/group_exit_time_{}.png".format(ds))