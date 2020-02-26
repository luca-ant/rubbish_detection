import pandas as pd
import matplotlib.pyplot as plt

logfile='training_log.csv'
results = pd.read_csv(logfile, sep=';')
print(results)



show1 = 'val_accuracy'
show2 = 'val_loss'


t = results['epoch']
data1 = results[show1]
data2 = results[show2]

fig, ax1 = plt.subplots()

plt.title("Trainig results")
color = 'tab:red'
ax1.set_xlabel('epoch')
ax1.set_ylabel(show1, color=color)
ax1.plot(t, data1, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel(show2, color=color)  # we already handled the x-label with ax1
ax2.plot(t, data2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped


#plt.show()
plt.savefig("training.png")

