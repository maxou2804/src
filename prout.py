import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df=pd.read_csv('/Users/mika/Documents/DATA/src/gruyere_metric_2000.csv')


plt.figure()
plt.plot(df['beta'],df['metric_perimeter'],'o')
plt.xlabel('beta')
plt.ylabel('hull non urba')
plt.show()

res_radius=stats.spearmanr(df['beta'],df['metric_perimeter'])
print(res_radius)