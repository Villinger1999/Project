import pandas as pd
import numpy as np

df=pd.read_csv('HR_data.csv')
df.drop(['Unnamed: 0'],axis=1,inplace=True)
columns_to_log = ['HR_std', 'HR_AUC']

for column in columns_to_log:
    df[column] = np.log(df[column])

median_value = df['Frustrated'].median()
df['Frus_Group'] = pd.cut(df['Frustrated'], bins=[float('-inf'), median_value, float('inf')], labels=[0, 1])

df