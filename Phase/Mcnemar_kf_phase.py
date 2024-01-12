from Kfold_CV_phase import *

log_table = pd.DataFrame(dict(y_true=y_test_f, y_pred=y_pred_f_kf))
log_table['log_tf'] = log_table.y_true == log_table.y_pred
log_tf = log_table.log_tf.value_counts()

base_table = pd.DataFrame(dict(y_true=y_test_f, y_pred=y_pred_base_f_kf))
base_table['base_tf'] = base_table.y_true == base_table.y_pred
base_tf = base_table.base_tf.value_counts()

dt_table = pd.DataFrame(dict(y_true=y_test_f, y_pred=y_pred_dt_f_kf))
dt_table['dt_tf'] = dt_table.y_true == dt_table.y_pred
dt_tf = dt_table.dt_tf.value_counts()

table_lbf = [[log_tf[1]+base_tf[1], log_tf[1]+base_tf[0]],
         [log_tf[0]+base_tf[1], log_tf[0]+base_tf[0]]]

table_lbf = pd.DataFrame(table_lbf, index=['Logistic Correct', 'Logistic Incorrect'], 
                     columns=['Baseline Correct', 'Baseline Incorrect'])

table_ldtf = [[log_tf[1]+dt_tf[1], log_tf[1]+dt_tf[0]],
         [log_tf[0]+dt_tf[1], log_tf[0]+dt_tf[0]]]

table_ldtf = pd.DataFrame(table_ldtf, index=['Logistic Correct', 'Logistic Incorrect'], 
                     columns=['Dt Correct', 'Dt Incorrect'])

table_bdtf = [[base_tf[1]+dt_tf[1], base_tf[1]+dt_tf[0]],[base_tf[0]+dt_tf[1], base_tf[0]+dt_tf[0]]]

table_bdtf = pd.DataFrame(table_bdtf, index=['Logistic Correct', 'Logistic Incorrect'], 
                     columns=['Dt', 'Dt Incorrect'])

# Logistic vs Baseline

sns.heatmap(table_lbf, annot=True, fmt=".1f", annot_kws={"size": 16})
plt.yticks(rotation=0)
b, t = plt.ylim() 
b += 0.5 
t -= 0.5 
plt.ylim(b, t) 
plt.show()

# calculate mcnemar test
result_lbf = mcnemar(table_lbf.values, exact=False, correction=True)

# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result_lbf.statistic, result_lbf.pvalue))

# interpret the p-value
alpha = 0.05
if result_lbf.pvalue > alpha:
    print('Same proportions of errors (fail to reject H0)')
else:
    print('Different proportions of errors (reject H0)')

# Logistic vs Decision Tree
sns.heatmap(table_ldtf, annot=True, fmt=".1f", annot_kws={"size": 16})
plt.yticks(rotation=0)
b, t = plt.ylim() 
b += 0.5 
t -= 0.5 
plt.ylim(b, t) 
plt.show()

# calculate mcnemar test
result_ldtf = mcnemar(table_ldtf.values, exact=False, correction=True)

# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result_ldtf.statistic, result_ldtf.pvalue))

# interpret the p-value
alpha = 0.05
if result_ldtf.pvalue > alpha:
    print('Same proportions of errors (fail to reject H0)')
else:
    print('Different proportions of errors (reject H0)')

# Baseline vs Decision Tree
sns.heatmap(table_bdtf, annot=True, fmt=".1f", annot_kws={"size": 16})
plt.yticks(rotation=0)
b, t = plt.ylim()
b += 0.5
t -= 0.5
plt.ylim(b, t)
plt.show()

# calculate mcnemar test
result_bdtf = mcnemar(table_bdtf.values, exact=False, correction=True)

# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result_bdtf.statistic, result_bdtf.pvalue))

# interpret the p-value
alpha = 0.05
if result_bdtf.pvalue > alpha:
    print('Same proportions of errors (fail to reject H0)')
else:
    print('Different proportions of errors (reject H0)')