import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from Stratified_CV import model_frustrated, model_frus_group, baseline_frustrated, baseline_frus, X_test_frustrated, y_test_frustrated, X_test_frus_group, y_test_frus_group, model_decision_tree_frus_group, model_decision_tree_frustrated

# Create subplots with 2 rows and 2 columns
fig, axs = plt.subplots(2, 3, figsize=(12, 8))

# Confusion matrix for Frustrated baseline
cm_baseline_frustrated_baseline = metrics.confusion_matrix(y_test_frustrated, baseline_frustrated.predict(X_test_frustrated),labels=[0,1,2,3,4,5,6,7,8,9,10])
cm_display_frustrated_baseline = metrics.ConfusionMatrixDisplay(cm_baseline_frustrated_baseline,display_labels=[0,1,2,3,4,5,6,7,8,9,10])
cm_display_frustrated_baseline.plot(ax=axs[0, 0])
axs[0, 0].set_title("Frustrated Baseline")

# Confusion matrix for Frus_Group baseline
cm_baseline_frus_group_baseline = metrics.confusion_matrix(y_test_frus_group, baseline_frus.predict(X_test_frus_group))
cm_display_frus_group_baseline = metrics.ConfusionMatrixDisplay(cm_baseline_frus_group_baseline)
cm_display_frus_group_baseline.plot(ax=axs[1, 0])
axs[1, 0].set_title("Frus_Group Baseline")

# Confusion matrix for Frustrated logistic regression model
cm_model_frustrated = metrics.confusion_matrix(y_test_frustrated, model_frustrated.predict(X_test_frustrated),labels=[0,1,2,3,4,5,6,7,8,9,10])
cm_display_frustrated = metrics.ConfusionMatrixDisplay(cm_model_frustrated,display_labels=[0,1,2,3,4,5,6,7,8,9,10])
cm_display_frustrated.plot(ax=axs[0, 1])
axs[0, 1].set_title("Frustrated Logistic Regression Model")

# Confusion matrix for Frus_Group logistic regression model
cm_model_frus_group = metrics.confusion_matrix(y_test_frus_group, model_frus_group.predict(X_test_frus_group))
cm_display_frus_group = metrics.ConfusionMatrixDisplay(cm_model_frus_group)
cm_display_frus_group.plot(ax=axs[1, 1])
axs[1, 1].set_title("Frus_Group Logistic Regression Model")

# Confusion matrix for Frustrated decision tree model
cm_decision_tree_frustrated = metrics.confusion_matrix(y_test_frustrated, model_decision_tree_frustrated.predict(X_test_frustrated),labels=[0,1,2,3,4,5,6,7,8,9,10])
cm_display_decision_tree_frustrated = metrics.ConfusionMatrixDisplay(cm_decision_tree_frustrated,display_labels=[0,1,2,3,4,5,6,7,8,9,10])
cm_display_decision_tree_frustrated.plot(ax=axs[0, 2])
axs[0, 2].set_title("Frustrated Decision Tree Model")

# Confusion matrix for Frus_Group decision tree model
cm_decision_tree_frus_group = metrics.confusion_matrix(y_test_frus_group, model_decision_tree_frus_group.predict(X_test_frus_group))
cm_display_decision_tree_frus_group = metrics.ConfusionMatrixDisplay(cm_decision_tree_frus_group)
cm_display_decision_tree_frus_group.plot(ax=axs[1, 2])
axs[1, 2].set_title("Frus_Group Decision Tree Model")

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()