
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# import mpld3

# mpld3.enable_notebook()

sns.set_style('white')

df = pd.read_csv('Predictions.csv')
df['Average'] = df['Features'].apply(lambda f: df[df['Features'] == f].mean())
df = df.sort_values('Average')

plt.subplots(figsize=(10, 6))
chart = sns.boxplot(x='Features', y='MSE', data=df, palette='Blues', linewidth=1.0, fliersize=3)

chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
plt.savefig('MSE.png', dpi=300, bbox_inches='tight')
plt.show()
