from Load_data import*
from scipy.stats import kruskal

variables = ['HR_Mean', 'HR_Median', 'HR_std', 'HR_Min']

for variable in variables:
    model = ols(f'({variable}) ~ C(Puzzler)', data=df).fit()
    anova_table = sm.stats.anova_lm(model)
    print(f"ANOVA Table for {variable}:")
    display(anova_table)

model = ols('Frustrated ~ C(Puzzler)', data=df).fit()
anova_table = sm.stats.anova_lm(model)
print(anova_table)

variables = ['HR_Max', 'HR_AUC']

for variable in variables:
    groups = [df[df[variable] == i]['Puzzler'] for i in df[variable].unique()]
    statistic, p_value = kruskal(*groups)
    print(f"Kruskal-Wallis Test for {variable}:")
    print(f"Statistic: {statistic}")
    print(f"P-value: {p_value}")