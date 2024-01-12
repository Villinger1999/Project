import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from sklearn import metrics
from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy.stats import kruskal
from scipy.stats import chi2_contingency
from scipy.stats import shapiro
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

df=pd.read_csv('HR_data.csv')
df.drop(['Unnamed: 0'],axis=1,inplace=True)
columns_to_log = ['HR_std']

for column in columns_to_log:
    df[column] = np.log(df[column])

median_value = df['Frustrated'].median()
df['Frus_Group'] = pd.cut(df['Frustrated'], bins=[float('-inf'), median_value, float('inf')], labels=[0, 1])