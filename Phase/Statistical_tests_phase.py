from Load_data_phase import*

variables = ['HR_Mean', 'HR_Median', 'HR_std', 'HR_Min']

df['Phase'] = df['Phase'].map({'phase1': 1, 'phase2': 2, 'phase3': 3})  # Convert Phase to numeric
df['Round'] = df['Round'].map({'round_1': 1, 'round_2': 2, 'round_3': 3, 'round_4' : 4})  # Convert Round to numeric

for variable in variables:
    model = ols(f'Phase ~ ({variable})', data=df).fit()
    anova_table = sm.stats.anova_lm(model)
    print(f"ANOVA Table for {variable}:")
    display(anova_table)


model = ols('Phase ~ HR_Mean + HR_Median + HR_std + HR_Min + HR_Max + HR_AUC + C(Frustrated)', data=df).fit()
anova_table = sm.stats.anova_lm(model)
print("ANOVA Table for all variables:")
display(anova_table)


variables = ['HR_Max', 'HR_AUC']

for variable in variables:
    groups = [df[df['Phase'] == i][variable] for i in df['Phase'].unique()]
    statistic, p_value = kruskal(*groups)
    print(f"Kruskal-Wallis Test for {variable}:")
    print(f"Statistic: {statistic}")
    print(f"P-value: {p_value}")


variables = ['HR_Mean', 'HR_Median', 'HR_std', 'HR_Max', 'HR_Min']

for variable in variables:
    model = ols(f'Phase ~ HR_AUC * ({variable})', data=df).fit()
    anova_table = sm.stats.anova_lm(model)
    print(f"ANOVA Table for {variable}:")
    display(anova_table)

variables = ['Round', 'Puzzler', 'Frustrated']

for variable in variables:
    model = ols(f'Phase ~ C({variable})', data=df).fit()
    anova_table = sm.stats.anova_lm(model)
    print(f"ANOVA Table for {variable}:")
    display(anova_table)

variables = ['Round', 'Puzzler', 'Phase']

for variable in variables:
    model = ols(f'Frustrated ~ C({variable})', data=df).fit()
    anova_table = sm.stats.anova_lm(model)
    print(f"ANOVA Table for {variable}:")
    display(anova_table) 


model = ols('Phase ~ HR_AUC * C(Frustrated)', data=df).fit()
anova_table = sm.stats.anova_lm(model)
print(f"ANOVA Table for Frustrated:")
display(anova_table)

