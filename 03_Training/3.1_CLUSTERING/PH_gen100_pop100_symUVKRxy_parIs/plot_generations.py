import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('study_log_data_algo1.csv')
print(df)

x = df['gen']
y = df['minFit']

plt.plot(x,y)
plt.savefig('gen_minFit.png')
plt.show()
plt.close()
