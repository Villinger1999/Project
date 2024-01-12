from ML_prepare_phase import *

# K-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Train and evaluate the model using stratified k-fold cross-validation for Frustrated
acc_log_f_kf = []
acc_base_f_kf = []
acc_dtree_f_kf = []
for train_index, val_index in kfold.split(X_train_f, y_train_f):
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

    acc_log_f_kf.append(acc_l)
    acc_base_f_kf.append(acc_b)
    acc_dtree_f_kf.append(acc_dt)


# Train and evaluate the model using stratified k-fold cross-validation for Frus_Group
acc_log_fg_kf = []
acc_base_fg_kf = []
acc_dtree_fg_kf = []

for train_index, val_index in kfold.split(X_train_fg, y_train_fg):
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

    acc_log_fg_kf.append(acc_l)
    acc_base_fg_kf.append(acc_b)
    acc_dtree_fg_kf.append(acc_dt)

# Calculate the average accuracy across all folds for Frustrated
av_acc_log_f_kf = sum(acc_log_f_kf) / len(acc_log_f_kf)
av_acc_base_f_kf = sum(acc_base_f_kf) / len(acc_base_f_kf)
av_acc_dt_f_kf = sum(acc_dtree_f_kf) / len(acc_dtree_f_kf)

# Calculate the average accuracy across all folds for Frus_Group
av_acc_log_fg_kf = sum(acc_log_fg_kf) / len(acc_log_fg_kf)
av_acc_base_fg_kf = sum(acc_base_fg_kf) / len(acc_base_fg_kf)
av_acc_dt_fg_kf = sum(acc_dtree_fg_kf) / len(acc_dtree_fg_kf)



# Print predictions for Frustrated
y_pred_f_kf = log_model_f.predict(X_test_f)

# Print predictions for Frus_Group
y_pred_fg_kf = log_model_fg.predict(X_test_fg)

y_pred_base_f_kf = base_model_f.predict(X_test_f)
y_pred_base_fg_kf = base_model_fg.predict(X_test_fg)

y_pred_dt_f_kf = dt_model_f.predict(X_test_f)
y_pred_dt_fg_kf = dt_model_fg.predict(X_test_fg)

# Calculate the accuracy for the test data for Frustrated
test_acc_f_kf = log_model_f.score(X_test_f, y_test_f)

# Calculate the accuracy for the test data for Frus_Group
test_acc_fg_kf = log_model_f.score(X_test_fg, y_test_fg)

# Calculate the accuracy for the test data for Frustrated Baseline
test_acc_base_f_kf = base_model_f.score(X_test_f, y_test_f)

# Calculate the accuracy for the test data for Frus_Group Baseline
test_acc_base_fg_kf = base_model_fg.score(X_test_fg, y_test_fg)

# Calculate the accuracy for the test data for Frustrated Decision Tree
test_acc_dt_f_kf = dt_model_f.score(X_test_f, y_test_f)

# Calculate the accuracy for the test data for Frus_Group Decision Tree
test_acc_dt_fg_kf = dt_model_fg.score(X_test_fg, y_test_fg)


# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Plot the average accuracy for Frustrated
axes[0, 0].bar(['Logistic', 'Baseline', 'Decision Tree'], [av_acc_log_f_kf, av_acc_base_f_kf, av_acc_dt_f_kf])
axes[0, 0].set_title('Average Accuracy for Frustrated')

# Plot the average accuracy for Frus_Group
axes[1, 0].bar(['Logistic', 'Baseline', 'Decision Tree'], [av_acc_log_fg_kf, av_acc_base_fg_kf, av_acc_dt_fg_kf])
axes[1, 0].set_title('Average Accuracy for Frus_Group')

# Plot the accuracy for the test data for Frustrated
axes[0, 1].bar(['Logistic', 'Baseline', 'Decision Tree'], [test_acc_f_kf, test_acc_base_f_kf, test_acc_dt_f_kf])
axes[0, 1].set_title('Accuracy for Frustrated (Test Data)')

# Plot the accuracy for the test data for Frus_Group
axes[1, 1].bar(['Logistic', 'Baseline', 'Decision Tree'], [test_acc_fg_kf, test_acc_base_fg_kf, test_acc_dt_fg_kf])
axes[1, 1].set_title('Accuracy for Frus_Group (Test Data)')

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()