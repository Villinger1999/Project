import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn import metrics

# Confusion matrix for Frustrated logistic
cm_log_f = metrics.confusion_matrix(y_test_f, y_pred_f_kf,labels=[0,1,2,3,4,5,6,7,8,9,10])
cm_display_log_f = metrics.ConfusionMatrixDisplay(cm_log_f,display_labels=[0,1,2,3,4,5,6,7,8,9,10])
cm_display_log_f.plot()