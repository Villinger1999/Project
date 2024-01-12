import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn import metrics
<<<<<<< HEAD
from Stratified_CV import *

fig, axs = plt.subplots(2, 3, figsize=(12, 8))

# Confusion matrix for Frustrated
cm_log_f_skf = metrics.confusion_matrix(y_test_f, y_pred_f_skf,labels=[0,1,2,3,4,5,6,7,8,9,10])
cm_display_log_f_skf = metrics.ConfusionMatrixDisplay(cm_log_f_skf,display_labels=[0,1,2,3,4,5,6,7,8,9,10])
cm_display_log_f_skf.plot(ax=axs[0, 0])
axs[0, 0].set_title("Logistic Regression")

cm_base_f_skf = metrics.confusion_matrix(y_test_f, y_pred_base_f_skf,labels=[0,1,2,3,4,5,6,7,8,9,10])
cm_display_base_f_skf = metrics.ConfusionMatrixDisplay(cm_base_f_skf,display_labels=[0,1,2,3,4,5,6,7,8,9,10])
cm_display_base_f_skf.plot(ax=axs[0, 1])
axs[0, 1].set_title("Baseline")

cm_dt_f_skf = metrics.confusion_matrix(y_test_f, y_pred_dt_f_skf,labels=[0,1,2,3,4,5,6,7,8,9,10])
cm_display_dt_f_skf = metrics.ConfusionMatrixDisplay(cm_dt_f_skf,display_labels=[0,1,2,3,4,5,6,7,8,9,10])
cm_display_dt_f_skf.plot(ax=axs[0, 2])
axs[0, 2].set_title("Decision Tree")

# Confusion matrix for Frus_Group baseline
cm_log_fg_skf = metrics.confusion_matrix(y_test_fg, y_pred_fg_skf,labels=[0,1])
cm_display_log_fg_skf = metrics.ConfusionMatrixDisplay(cm_log_fg_skf,display_labels=[0,1])
cm_display_log_fg_skf.plot(ax=axs[1, 0])
axs[1, 0].set_title("Logistic Regression")

cm_base_fg_skf = metrics.confusion_matrix(y_test_fg, y_pred_base_fg_skf,labels=[0,1])
cm_display_base_fg_skf = metrics.ConfusionMatrixDisplay(cm_base_fg_skf,display_labels=[0,1])
cm_display_base_fg_skf.plot(ax=axs[1, 1])
axs[1, 1].set_title("Baseline")

cm_dt_fg_skf = metrics.confusion_matrix(y_test_fg, y_pred_dt_fg_skf,labels=[0,1])
cm_display_dt_fg_skf = metrics.ConfusionMatrixDisplay(cm_dt_fg_skf,display_labels=[0,1])
cm_display_dt_fg_skf.plot(ax=axs[1, 2])
axs[1, 2].set_title("Decision Tree")
=======
from Kfold_CV import*
from Leave_one_out import*
from Stratified_CV import*
>>>>>>> 46de917e66bec0f345078d0e24278b183455827c

