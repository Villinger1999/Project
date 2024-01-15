from ML_prepare import *
from Kfold_CV import *
from sklearn.metrics import classification_report


df_cohorts_1 = df[df['Cohort'] == 'D1_1']
df_cohorts_2 = df[df['Cohort'] == 'D1_2']

cohort_1_log = log_model_f.score(df_cohorts_1.iloc[:, 4:5], df_cohorts_1['Frustrated'])
cohort_2_log = log_model_f.score(df_cohorts_2.iloc[:, 4:5], df_cohorts_2['Frustrated'])
cohort_1_baseline = base_model_f.score(df_cohorts_1.iloc[:, 4:5], df_cohorts_1['Frustrated'])
cohort_2_baseline = base_model_f.score(df_cohorts_2.iloc[:, 4:5], df_cohorts_2['Frustrated'])
cohort_1_decision = dt_model_f.score(df_cohorts_1.iloc[:, 4:5], df_cohorts_1['Frustrated'])
cohort_2_decision = dt_model_f.score(df_cohorts_2.iloc[:, 4:5], df_cohorts_2['Frustrated'])


pred_cohort_1_log = log_model_f.predict(df_cohorts_1.iloc[:, 4:5])
pred_cohort_2_log = log_model_f.predict(df_cohorts_2.iloc[:, 4:5])
pred_cohort_1_baseline = base_model_f.predict(df_cohorts_1.iloc[:, 4:5])
pred_cohort_2_baseline = base_model_f.predict(df_cohorts_2.iloc[:, 4:5])
pred_cohort_1_decision = dt_model_f.predict(df_cohorts_1.iloc[:, 4:5])
pred_cohort_2_decision = dt_model_f.predict(df_cohorts_2.iloc[:, 4:5])

print("Logistic Regression Cohort 1")
print(classification_report(df_cohorts_1['Frustrated'], pred_cohort_1_log, zero_division=0))

print("Logistic Regression Cohort 2")
print(classification_report(df_cohorts_2['Frustrated'], pred_cohort_2_log, zero_division=0))

print("Baseline Cohort 1")
print(classification_report(df_cohorts_1['Frustrated'], pred_cohort_1_baseline, zero_division=0))

print("Baseline Cohort 2")
print(classification_report(df_cohorts_2['Frustrated'], pred_cohort_2_baseline, zero_division=0))

print("Decision Tree Cohort 1")
print(classification_report(df_cohorts_1['Frustrated'], pred_cohort_1_decision, zero_division=0))

print("Decision Tree Cohort 2")
print(classification_report(df_cohorts_2['Frustrated'], pred_cohort_2_decision, zero_division=0))





