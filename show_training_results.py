import config
import pandas as pd
import matplotlib.pyplot as plt

logfile=config.train_log_file
model_name = ''.join(config.train_log_file.split('/')[-1].split('_')[0])
results = pd.read_csv(logfile, sep=';')
print(results)



plt.title("Trainig results of "+model_name+": accuracy")

x = results['epoch']
plt.plot(x, results['accuracy'])
plt.plot(x, results['val_accuracy'])
plt.legend(['train_accuracy', 'val_accuracy'], loc='lower right')
plt.xlabel('epoch')
#plt.show()
plt.savefig(config.train_log_dir+ model_name+"_accuracy.png")


plt.clf()


plt.title("Trainig results of "+model_name+": loss")
x = results['epoch']
plt.plot(x, results['loss'])
plt.plot(x, results['val_loss'])
plt.legend(['train_loss', 'val_loss'], loc='upper right')
plt.xlabel('epoch')
#plt.show()
plt.savefig(config.train_log_dir+ model_name+"_loss.png")
