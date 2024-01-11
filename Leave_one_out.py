from ML_prepare import *
from sklearn.model_selection import LeaveOneOut
from matplotlib import pyplot as plt


# Leave one out
loo = LeaveOneOut()

# Train and evaluate the model using stratified k-fold cross-validation for Frustrated
acc_log_f = []
acc_base_f = []
acc_dtree_f = []
for train_index, val_index in loo.split(X_train_f, y_train_f):
    X_train_fold, X_val_fold = X_train_f.iloc[train_index], X_train_f.iloc[val_index]
    y_train_fold, y_val_fold = y_train_f.iloc[train_index], y_train_f.iloc[val_index]
    
    # Train the model on the training fold
    log_model_f.fit(X_train_fold, y_train_fold)
    base_model_f.fit(X_train_fold, y_train_fold)
    dt_model_f.fit(X_train_fold, y_train_fold)
    
    
    # Evaluate the model on the validation fold
    acc_l = log_model_f.score(X_val_fold, y_val_fold)
    acc_b = base_model_f.score(X_val_fold, y_val_fold)
    acc_dt = dt_model_f.score(X_val_fold, y_val_fold)

    acc_log_f.append(acc_l)
    acc_base_f.append(acc_b)
    acc_dtree_f.append(acc_dt)

# Train and evaluate the model using stratified k-fold cross-validation for Frus_Group
acc_log_fg = []
acc_base_fg = []
acc_dtree_fg = []

for train_index, val_index in loo.split(X_train_fg, y_train_fg):
    X_train_fold, X_val_fold = X_train_fg.iloc[train_index], X_train_fg.iloc[val_index]
    y_train_fold, y_val_fold = y_train_fg.iloc[train_index], y_train_fg.iloc[val_index]
    
    # Train the model on the training fold
    log_model_fg.fit(X_train_fold, y_train_fold)
    base_model_fg.fit(X_train_fold, y_train_fold)
    dt_model_fg.fit(X_train_fold, y_train_fold)
    
    
    # Evaluate the model on the validation fold
    acc_l = log_model_fg.score(X_val_fold, y_val_fold)
    acc_b = base_model_fg.score(X_val_fold, y_val_fold)
    acc_dt = dt_model_fg.score(X_val_fold, y_val_fold)

    acc_log_fg.append(acc_l)
    acc_base_fg.append(acc_b)
    acc_dtree_fg.append(acc_dt)

# Calculate the average accuracy across all folds for Frustrated
av_acc_log_f = sum(acc_log_f) / len(acc_log_f)
av_acc_base_f = sum(acc_base_f) / len(acc_base_f)
av_acc_dt_f = sum(acc_dtree_f) / len(acc_dtree_f)

# Calculate the average accuracy across all folds for Frus_Group
av_acc_log_fg = sum(acc_log_fg) / len(acc_log_fg)
av_acc_base_fg = sum(acc_base_fg) / len(acc_base_fg)
av_acc_dt_fg = sum(acc_dtree_fg) / len(acc_dtree_fg)



# Print predictions for Frustrated
y_pred_f = log_model_f.predict(X_test_f)

# Print predictions for Frus_Group
y_pred_fg = log_model_fg.predict(X_test_fg)

y_pred_base_f = base_model_f.predict(X_test_f)
y_pred_base_fg = base_model_fg.predict(X_test_fg)

# Calculate the accuracy for the test data for Frustrated
test_acc_f = log_model_f.score(X_test_f, y_test_f)

# Calculate the accuracy for the test data for Frus_Group
test_acc_fg = log_model_f.score(X_test_fg, y_test_fg)

# Calculate the accuracy for the test data for Frustrated Baseline
test_acc_base_f = base_model_f.score(X_test_f, y_test_f)

# Calculate the accuracy for the test data for Frus_Group Baseline
test_acc_base_fg= base_model_fg.score(X_test_fg, y_test_fg)

# Calculate the accuracy for the test data for Frustrated Decision Tree
test_acc_dt_f = dt_model_f.score(X_test_f, y_test_f)

# Calculate the accuracy for the test data for Frus_Group Decision Tree
test_acc_dt_fg = dt_model_fg.score(X_test_fg, y_test_fg)


# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Plot the average accuracy for Frustrated
axes[0, 0].bar(['Logistic', 'Baseline', 'Decision Tree'], [av_acc_log_f, av_acc_base_f, av_acc_dt_f])
axes[0, 0].set_title('Average Accuracy for Frustrated')

# Plot the average accuracy for Frus_Group
axes[1, 0].bar(['Logistic', 'Baseline', 'Decision Tree'], [av_acc_log_fg, av_acc_base_fg, av_acc_dt_fg])
axes[1, 0].set_title('Average Accuracy for Frus_Group')

# Plot the accuracy for the test data for Frustrated
axes[0, 1].bar(['Logistic', 'Baseline', 'Decision Tree'], [test_acc_f, test_acc_base_f, test_acc_dt_f])
axes[0, 1].set_title('Accuracy for Frustrated (Test Data)')

# Plot the accuracy for the test data for Frus_Group
axes[1, 1].bar(['Logistic', 'Baseline', 'Decision Tree'], [test_acc_fg, test_acc_base_fg, test_acc_dt_fg])
axes[1, 1].set_title('Accuracy for Frus_Group (Test Data)')

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()