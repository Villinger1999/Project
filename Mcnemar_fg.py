from statsmodels.stats.contingency_tables import mcnemar
from Kfold_CV import *

log_table = pd.DataFrame(dict(y_true=y_test_fg, y_pred=y_pred_fg_kf))
log_table['log_tfg'] = log_table.y_true == log_table.y_pred
log_tfg = log_table.log_tfg.value_counts()

base_table = pd.DataFrame(dict(y_true=y_test_fg, y_pred=y_pred_base_fg_kf))
base_table['base_tfg'] = base_table.y_true == base_table.y_pred
base_tfg = base_table.base_tfg.value_counts()

dt_table = pd.DataFrame(dict(y_true=y_test_fg, y_pred=y_pred_dt_fg_kf))
dt_table['dt_tfg'] = dt_table.y_true == dt_table.y_pred
dt_tfg = dt_table.dt_tfg.value_counts()

table_lbfg = [[log_tfg[1]+base_tfg[1], log_tfg[1]+base_tfg[0]],
         [log_tfg[0]+base_tfg[1], log_tfg[0]+base_tfg[0]]]

table_lbfg = pd.DataFrame(table_lbfg, index=['Logistic Correct', 'Logistic Incorrect'], 
                     columns=['Baseline Correct', 'Baseline Incorrect'])

table_ldtfg = [[log_tfg[1]+dt_tfg[1], log_tfg[1]+dt_tfg[0]],
         [log_tfg[0]+dt_tfg[1], log_tfg[0]+dt_tfg[0]]]

table_ldtfg = pd.DataFrame(table_ldtfg, index=['Logistic Correct', 'Logistic Incorrect'], 
                     columns=['Dt Correct', 'Dt Incorrect'])

table_bdtfg = [[base_tfg[1]+dt_tfg[1], base_tfg[1]+dt_tfg[0]],[base_tfg[0]+dt_tfg[1], base_tfg[0]+dt_tfg[0]]]

table_bdtfg = pd.DataFrame(table_bdtfg, index=['Logistic Correct', 'Logistic Incorrect'], 
                     columns=['Dt', 'Dt Incorrect'])

# Logistic vs Baseline

sns.heatmap(table_lbfg, annot=True, fmt=".1f", annot_kws={"size": 16})
plt.yticks(rotation=0)
b, t = plt.ylim() 
b += 0.5 
t -= 0.5 
plt.ylim(b, t) 
plt.show()

# calculate mcnemar test
result_lbfg = mcnemar(table_lbfg.values, exact=False, correction=True)

# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result_lbfg.statistic, result_lbfg.pvalue))

# interpret the p-value
alpha = 0.05
if result_lbfg.pvalue > alpha:
    print('Same proportions of errors (fail to reject H0)')
else:
    print('Different proportions of errors (reject H0)')

# Logistic vs Decision Tree
sns.heatmap(table_ldtfg, annot=True, fmt=".1f", annot_kws={"size": 16})
plt.yticks(rotation=0)
b, t = plt.ylim() 
b += 0.5 
t -= 0.5 
plt.ylim(b, t) 
plt.show()

# calculate mcnemar test
result_ldtfg = mcnemar(table_ldtfg.values, exact=False, correction=True)

# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result_ldtfg.statistic, result_ldtfg.pvalue))

# interpret the p-value
alpha = 0.05
if result_ldtfg.pvalue > alpha:
    print('Same proportions of errors (fail to reject H0)')
else:
    print('Different proportions of errors (reject H0)')

# Baseline vs Decision Tree
sns.heatmap(table_bdtfg, annot=True, fmt=".1f", annot_kws={"size": 16})
plt.yticks(rotation=0)
b, t = plt.ylim()
b += 0.5
t -= 0.5
plt.ylim(b, t)
plt.show()

# calculate mcnemar test
result_bdtfg = mcnemar(table_bdtfg.values, exact=False, correction=True)

# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result_bdtfg.statistic, result_bdtfg.pvalue))

# interpret the p-value
alpha = 0.05
if result_bdtfg.pvalue > alpha:
    print('Same proportions of errors (fail to reject H0)')
else:
    print('Different proportions of errors (reject H0)')