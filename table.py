import pandas as pd


data={'City': ['Ningbo','Chengdu Deyang', 'Beijing Lafang','Changzhou','Bengalore','Kolkata','Paris','Bangkok','Cairo','Guatemala City','Johannesburg','London','Mexico City','Nairobi','Santiago','Sao Paulo','Tehran','Las Vegas','Atlanta'],
      'alpha':[0.56, 0.53, 0.54, 0.54, 0.55, 0.52, 0.52 ,0.53, 0.53, 0.52, 0.58, 0.54, 0.55, 0.56, 0.58, 0.51, 0.55, 0.55, 0.56],
      'beta': [0.44, 0.68, 0.41, 0.37, 0.83, 0.34, 0.56, 1.01, 0.37, 0.37, 0.07, 0.01, 0.04, 0.28, 0.62, 0.89, 0.10, 0.41, 0.04],
       '1/z': [0.58, 0.68, 0.58, 0.56, 0.72, 0.74, 0.76, 0.80, 0.54, 0.33, 0.27, 0.21, 0.25, 0.27, 0.74, 0.41, 0.54, 0.37, 0.52]}





df=pd.DataFrame(data)



df['anisotropy']=[0.3, 0.45, 0.45, 0.25, 0.7, 1.45, 1.7, 0.85, 0.5, 0.7, 0.75,0.82, 0.80, 0.95, 1.05, 0.8, 1.1, 0.5, 1.2 ]



import matplotlib.pyplot as plt
from scipy import stats


corr=stats.spearmanr(df['anisotropy'],df['beta'])
print(corr)

plt.figure()
plt.plot(df['anisotropy'],df['beta'],'o')
plt.show()

