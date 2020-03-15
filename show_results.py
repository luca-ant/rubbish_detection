import config
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


if os.path.isdir(config.train_log_dir):

    with os.scandir(config.train_log_dir) as entries:

        for e in entries:
            if e.is_file() and e.name.endswith('.csv'):
                logfile = e.name
                model_name = ''.join(logfile.split('_')[0])


                print('MODEL: ' + model_name)
                results = pd.read_csv(config.train_log_dir+logfile, sep=';')
                print(results)


                x = results['epoch']

                fig, ax1 = plt.subplots()

                plt.title("Trainig results of "+model_name+": accuracy")
                ax1.set_xlabel('epoch')
                ax1.set_ylabel('accuracy')
                ax1.plot(x, results['accuracy'])
                ax1.plot(x, results['val_accuracy'])
                ax1.legend(['train_accuracy', 'val_accuracy'], loc='center right')
                ax1.tick_params(axis='y' )
                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

                color = 'tab:red'
                ax2.set_ylabel('learning rate', color=color)  # we already handled the x-label with ax1
                ax2.plot(x, results['lr'], color=color)
                ax2.tick_params(axis='y', labelcolor=color)
                fig.tight_layout()  # otherwise the right y-label is slightly clipped
                #plt.show()
                plt.savefig(config.train_log_dir+ model_name+"_accuracy.png")

                plt.clf()


                fig, ax1 = plt.subplots()

                plt.title("Trainig results of "+model_name+": loss")
                ax1.set_xlabel('epoch')
                ax1.set_ylabel('loss')
                ax1.plot(x, results['loss'])
                ax1.plot(x, results['val_loss'])
                ax1.legend(['train_loss', 'val_loss'], loc='center right')
                ax1.tick_params(axis='y' )
                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

                color = 'tab:red'
                ax2.set_ylabel('learning rate', color=color)  # we already handled the x-label with ax1
                ax2.plot(x, results['lr'], color=color)
                ax2.tick_params(axis='y', labelcolor=color)
                fig.tight_layout()  # otherwise the right y-label is slightly clipped
                #plt.show()
                plt.savefig(config.train_log_dir+ model_name+"_loss.png")

                plt.clf()


                plt.title("Trainig results of "+model_name+": learning rate")
                x = results['epoch']
                plt.plot(x, results['lr'])
                #plt.legend(['learning rate'], loc='upper right')
                plt.xlabel('epoch')
                plt.ylabel('learning rate')
                plt.tight_layout()  
                #plt.show()
                plt.savefig(config.train_log_dir+ model_name+"_lr.png")

else:
    print("Logs not found in {}".format(config.train_log_dir))



