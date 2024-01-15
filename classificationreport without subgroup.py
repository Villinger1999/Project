from sklearn.metrics import classification_report
from Kfold_CV import *

print("Logistic Regression")
print(classification_report(y_test_f, y_pred_f_kf, zero_division=0))

print("Logistic Regression grouped")
print(classification_report(y_test_fg, y_pred_fg_kf, zero_division=0))

print("Baseline")
print(classification_report(y_test_f, y_pred_base_f_kf, zero_division=0))

print("Baseline grouped")
print(classification_report(y_test_fg, y_pred_base_fg_kf, zero_division=0))

print("Decision Tree")
print(classification_report(y_test_f, y_pred_dt_f_kf, zero_division=0))

print("Decision Tree grouped")
print(classification_report(y_test_fg, y_pred_dt_fg_kf, zero_division=0))

