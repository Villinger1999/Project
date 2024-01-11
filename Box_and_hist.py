from Load_data import*
import matplotlib.pyplot as plt
import seaborn as sns

plt.hist(df["Frustrated"], bins=20)
plt.show()

sns.boxplot(x = df["Frustrated"], y = df["HR_Mean"])
plt.show()

sns.boxplot(x = df["Round"], y = df["Frustrated"])
plt.show()

sns.boxplot(x = df["Round"], y = df["HR_Mean"])
plt.show()

sns.boxplot(x = df["Phase"], y = df["Frustrated"])
plt.show()

sns.boxplot(x = df["Phase"], y = df["HR_Mean"])
plt.show()