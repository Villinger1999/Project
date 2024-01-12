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

# Calculate the average accuracy across all folds for Frustrated
av_acc_log_f_kf = sum(acc_log_f_kf) / len(acc_log_f_kf)
av_acc_base_f_kf = sum(acc_base_f_kf) / len(acc_base_f_kf)
av_acc_dt_f_kf = sum(acc_dtree_f_kf) / len(acc_dtree_f_kf)

# Print predictions for Frustrated
import matplotlib.pyplot as plt

y_pred_f_kf = log_model_f.predict(X_test_f)

y_pred_base_f_kf = base_model_f.predict(X_test_f)

y_pred_dt_f_kf = dt_model_f.predict(X_test_f)

# Calculate the accuracy for the test data for Frustrated
test_acc_f_kf = log_model_f.score(X_test_f, y_test_f)

# Calculate the accuracy for the test data for Frustrated Baseline
test_acc_base_f_kf = base_model_f.score(X_test_f, y_test_f)

# Calculate the accuracy for the test data for Frustrated Decision Tree
test_acc_dt_f_kf = dt_model_f.score(X_test_f, y_test_f)

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(1,2, figsize=(10, 8))

# Plot the average accuracy for Frustrated
axes[0].bar(['Logistic', 'Baseline', 'Decision Tree'], [av_acc_log_f_kf, av_acc_base_f_kf, av_acc_dt_f_kf])
axes[0].set_title('Average Accuracy for Phase')

# Plot the accuracy for the test data for Frustrated
axes[1].bar(['Logistic', 'Baseline', 'Decision Tree'], [test_acc_f_kf, test_acc_base_f_kf, test_acc_dt_f_kf])
axes[1].set_title('Accuracy for Phase (Test Data)')

# Display the plot
plt.show()
