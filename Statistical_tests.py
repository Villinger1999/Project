from Load_data import*
from statsmodels.formula.api import ols
import statsmodels.api as sm
from IPython.display import display
from scipy.stats import kruskal
from scipy.stats import chi2_contingency
from scipy.stats import shapiro

variables = ['HR_Mean', 'HR_Median', 'HR_std', 'HR_Max', 'HR_Min', 'HR_AUC']

pvalue = []

for variable in variables:
    # Perform Shapiro-Wilk test
    statistic, p_value = shapiro(df[variable])
    pvalue.append(p_value)

    # Check if data is parametric or non-parametric
    if p_value < 0.05:
        print(f"{variable}: Data is non-parametric")
    else:
        print(f"{variable}: Data is parametric")

pvalue


variables = ['HR_Mean', 'HR_Median', 'HR_std', 'HR_Min', 'HR_Max', 'HR_AUC', 'Puzzler', 'Frustrated']

for variable in variables:
    contingency_table = pd.crosstab(df[variable], df['Frustrated'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-square Test of Independence for {variable}:")
    print(f"Chi-square statistic: {chi2}")
    print(f"P-value: {p_value}")

# If the p-value is less than 0.05, we reject the null hypothesis and conclude that the variables are not independent. 
# If the p-value is larger than 0.05, we cannot reject the null hypothesis and conclude that the variables are independent.


variables = ['HR_std', 'HR_Max', 'HR_AUC']

for variable in variables:
    groups = [df[df['Frustrated'] == i][variable] for i in df['Frustrated'].unique()]
    statistic, p_value = kruskal(*groups)
    print(f"Kruskal-Wallis Test for {variable}:")
    print(f"Statistic: {statistic}")
    print(f"P-value: {p_value}")


variables = ['Round', 'Phase', 'Puzzler']

for variable in variables:
    model = ols(f'Frustrated ~ C({variable})', data=df).fit()
    anova_table = sm.stats.anova_lm(model)
    print(f"ANOVA Table for {variable}:")
    display(anova_table)
    

variables = ['HR_Mean', 'HR_Median', 'HR_std', 'HR_Min', 'HR_Max', 'HR_AUC']

for variable in variables:
    model = ols(f'Frustrated ~ C(Phase) + ({variable})', data=df).fit()
    anova_table = sm.stats.anova_lm(model)
    print(f"ANOVA Table for {variable}:")
    display(anova_table)


variables = ['Round', 'Puzzler']

for variable in variables:
    model = ols(f'Frustrated ~ C(Phase) + C({variable})', data=df).fit()
    anova_table = sm.stats.anova_lm(model)
    print(f"ANOVA Table for {variable}:")
    display(anova_table)


variables = ['HR_Mean', 'HR_Median', 'HR_std', 'HR_Max', 'HR_Min', 'HR_AUC']

for variable in variables:
    model = ols(f'Frustrated ~ C(Phase) * ({variable})', data=df).fit()
    anova_table = sm.stats.anova_lm(model)
    print(f"ANOVA Table for {variable}:")
    display(anova_table)


variables = ['Round', 'Puzzler']

for variable in variables:
    model = ols(f'Frustrated ~ C(Phase) * C({variable})', data=df).fit()
    anova_table = sm.stats.anova_lm(model)
    print(f"ANOVA Table for {variable}:")
    display(anova_table)

