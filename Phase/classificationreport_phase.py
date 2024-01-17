from sklearn.metrics import classification_report
from Kfold_CV_phase import *

print("Baseline")
print(classification_report(y_test_f, y_pred_base_f_kf, zero_division=0))

print("Descision Tree")
print(classification_report(y_test_f, y_pred_dt_f_kf, zero_division=0))

print("Logistic Regression")
print(classification_report(y_test_f, y_pred_f_kf, zero_division=0))