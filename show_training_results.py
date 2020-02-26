import pandas as pd

logfile='training_log.csv'
results = pd.read_csv(logfile, sep=';')
print(results)
