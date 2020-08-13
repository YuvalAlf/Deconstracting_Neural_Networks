import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

font = {'family' : 'Calibri Light',
        'weight' : 'bold',
        'size'   : 12}

matplotlib.rc('font', **font)

# import mpld3

# mpld3.enable_notebook()

sns.set_style('white')

df = pd.read_csv('Predictions.csv')
# df = df[df['MSE'] < 0.2]
df['Median'] = df['Features'].apply(lambda f: df[df['Features'] == f]['MSE'].mean())
df = df.sort_values('Median')
unique_df = df.drop_duplicates(['Features'])


plt.subplots(figsize=(12, 8))
chart = sns.boxplot(x='Features', y='MSE', data=df, linewidth=1.0, fliersize=2) # , inner=None
# dup_df = df.drop_duplicates(subset=['Features'], keep='first', inplace=False)
# chart = sns.swarmplot(x='Features', y='MSE', data=df, hue='Correlation', linewidth=1.0, palette='Reds')
palette = matplotlib.cm.get_cmap('BrBG')
min_val = -1.0#unique_df['Correlation'].min()
max_val = 1.0#unique_df['Correlation'].max()
# print(min_val)
# print(max_val)
for i, box in enumerate(chart.artists):
        corr = unique_df.iloc[i]['Correlation']
        box.set_facecolor(palette((corr - min_val) / (max_val - min_val)))
        #if corr < 0.0:
        #        box.set_facecolor(palette((-corr / (2 * min_val) + 0.5)))
        #else:
        #        box.set_facecolor(palette((corr / (2 * max_val) + 0.5)))

chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

plt.savefig('MSE_Median.png', dpi=300, bbox_inches='tight')
plt.show()
