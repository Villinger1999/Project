from Load_data import*

model = ols('HR_Mean ~ C(Frustrated)', data=df).fit()
anova_table = sm.stats.anova_lm(model)
anova_table
variables = ['HR_Mean', 'HR_Median', 'HR_std', 'HR_Min']

for variable in variables:
    model = ols(f'({variable}) ~ C(Frustrated)', data=df).fit()
    anova_table = sm.stats.anova_lm(model)
    print(f"ANOVA Table for {variable}:")
    display(anova_table)
from scipy.stats import kruskal
variables = ['HR_Max', 'HR_AUC']

for variable in variables:
    groups = [df[df[variable] == i]['Frustrated'] for i in df[variable].unique()]
    statistic, p_value = kruskal(*groups)
    print(f"Kruskal-Wallis Test for {variable}:")
    print(f"Statistic: {statistic}")
    print(f"P-value: {p_value}")
variables = ['HR_Mean', 'HR_Median', 'HR_std', 'HR_Min']

for variable in variables:
    model = ols(f'({variable}) ~ C(Puzzler)', data=df).fit()
    anova_table = sm.stats.anova_lm(model)
    print(f"ANOVA Table for {variable}:")
    display(anova_table)
from scipy.stats import kruskal
variables = ['HR_Max', 'HR_AUC']

for variable in variables:
    groups = [df[df[variable] == i]['Puzzler'] for i in df[variable].unique()]
    statistic, p_value = kruskal(*groups)
    print(f"Kruskal-Wallis Test for {variable}:")
    print(f"Statistic: {statistic}")
    print(f"P-value: {p_value}")
model = ols('Frustrated ~ C(Puzzler)', data=df).fit()
anova_table = sm.stats.anova_lm(model)
anova_table
model = ols('Frustrated ~ C(Puzzler)', data=df).fit()
anova_table = sm.stats.anova_lm(model)
groups = [df[df['Puzzler'] == i]['Frustrated'] for i in df['Puzzler'].unique()]
statistic, p_value = stats.kruskal(*groups)
print("Kruskal-Wallis Test:")
print("Statistic:", statistic)
print("P-value:", p_value)

model = ols('Puzzler ~ C(Frustrated)', data=df).fit()
anova_table = sm.stats.anova_lm(model)
anova_table
model = ols('Puzzler ~ C(Frustrated)', data=df).fit()
anova_table = sm.stats.anova_lm(model)
groups = [df[df['Puzzler'] == i]['Frustrated'] for i in df['Puzzler'].unique()]
statistic, p_value = stats.kruskal(*groups)
print("Kruskal-Wallis Test:")
print("Statistic:", statistic)
print("P-value:", p_value)