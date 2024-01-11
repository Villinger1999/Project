from ML_prepare import X_train_frustrated, y_train_frustrated, X_test_frustrated, y_test_frustrated, X_train_frus_group, y_train_frus_group, X_test_frus_group, y_test_frus_group, model_frustrated, model_frus_group, baseline_frustrated, baseline_frus, model_decision_tree_frustrated, model_decision_tree_frus_group
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt

# K-fold cross-validation
kfold = KFold(n_splits=2, shuffle=True, random_state=42)



# Train and evaluate the model using k-fold cross-validation for Frustrated
accuracies_frustrated_k = []
accuracies_frustrated_baseline_k = []
accuracies_frustrated_decision_tree_k = []
for train_index, val_index in kfold.split(X_train_frustrated):
    X_train_fold, X_val_fold = X_train_frustrated.iloc[train_index], X_train_frustrated.iloc[val_index]
    y_train_fold, y_val_fold = y_train_frustrated.iloc[train_index], y_train_frustrated.iloc[val_index]
    
    # Train the model on the training fold
    model_frustrated.fit(X_train_fold, y_train_fold)
    baseline_frustrated.fit(X_train_fold, y_train_fold)
    model_decision_tree_frustrated.fit(X_train_fold, y_train_fold)
    
    # Evaluate the model on the validation fold
    accuracy_k = model_frustrated.score(X_val_fold, y_val_fold)
    accuracy_baseline_k = baseline_frustrated.score(X_val_fold, y_val_fold)
    accuracy_decision_tree_k = model_decision_tree_frustrated.score(X_val_fold, y_val_fold)

    accuracies_frustrated_k.append(accuracy_k)
    accuracies_frustrated_baseline_k.append(accuracy_baseline_k)
    accuracies_frustrated_decision_tree_k.append(accuracy_decision_tree_k)

# Train and evaluate the model using k-fold cross-validation for Frus_Group
accuracies_frus_group_k = []
accuracies_frus_group_baseline_k = []
accuracies_frus_group_decision_tree_k = []
for train_index, val_index in kfold.split(X_train_frus_group):
    X_train_fold, X_val_fold = X_train_frus_group.iloc[train_index], X_train_frus_group.iloc[val_index]
    y_train_fold, y_val_fold = y_train_frus_group.iloc[train_index], y_train_frus_group.iloc[val_index]
    
    # Train the model on the training fold
    model_frus_group.fit(X_train_fold, y_train_fold)
    baseline_frus.fit(X_train_fold, y_train_fold)
    model_decision_tree_frus_group.fit(X_train_fold, y_train_fold)
    
    # Evaluate the model on the validation fold
    accuracy_k = model_frus_group.score(X_val_fold, y_val_fold)
    accuracy_baseline_k = baseline_frus.score(X_val_fold, y_val_fold)
    accuracy_decision_tree_k = model_decision_tree_frus_group.score(X_val_fold, y_val_fold)

    accuracies_frus_group_k.append(accuracy_k)
    accuracies_frus_group_baseline_k.append(accuracy_baseline_k)
    accuracies_frus_group_decision_tree_k.append(accuracy_decision_tree_k)


# Calculate the average accuracy across all folds for Frustrated
average_accuracy_frustrated_k = sum(accuracies_frustrated_k) / len(accuracies_frustrated_k)
average_accuracy_frustrated_baseline_k = sum(accuracies_frustrated_baseline_k) / len(accuracies_frustrated_baseline_k)
average_accuracy_decision_tree_frustrated_k = sum(accuracies_frustrated_decision_tree_k) / len(accuracies_frustrated_decision_tree_k)

# Calculate the average accuracy across all folds for Frus_Group
average_accuracy_frus_group_k = sum(accuracies_frus_group_k) / len(accuracies_frus_group_k)
average_accuracy_frus_group_baseline_k = sum(accuracies_frus_group_baseline_k) / len(accuracies_frus_group_baseline_k)
average_accuracy_decision_tree_frus_group_k = sum(accuracies_frus_group_decision_tree_k) / len(accuracies_frus_group_decision_tree_k)



# Print predictions for Frustrated
y_pred_frustrated_k = model_frustrated.predict(X_test_frustrated)

# Print predictions for Frus_Group
y_pred_frus_group_k = model_frus_group.predict(X_test_frus_group)

y_pred_frustrated_baseline_k = baseline_frustrated.predict(X_test_frustrated)
y_pred_frus_group_baseline_k = baseline_frus.predict(X_test_frus_group)

# Calculate the accuracy for the test data for Frustrated
test_accuracy_frustrated_k = model_frustrated.score(X_test_frustrated, y_test_frustrated)

# Calculate the accuracy for the test data for Frus_Group
test_accuracy_frus_group_k = model_frus_group.score(X_test_frus_group, y_test_frus_group)

# Calculate the accuracy for the test data for Frustrated Baseline
test_accuracy_frustrated_baseline_k = baseline_frustrated.score(X_test_frustrated, y_test_frustrated)

# Calculate the accuracy for the test data for Frus_Group Baseline
test_accuracy_frus_group_baseline_k = baseline_frus.score(X_test_frus_group, y_test_frus_group)

# Calculate the accuracy for the test data for Frustrated Decision Tree
test_accuracy_decision_tree_frustrated_k = model_decision_tree_frustrated.score(X_test_frustrated, y_test_frustrated)

# Calculate the accuracy for the test data for Frus_Group Decision Tree
test_accuracy_decision_tree_frus_group_k = model_decision_tree_frus_group.score(X_test_frus_group, y_test_frus_group)


# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Plot the average accuracy for Frustrated
axes[0, 0].bar(['Logistic', 'Baseline', 'Decision Tree'], [average_accuracy_frustrated_k, average_accuracy_frustrated_baseline_k, average_accuracy_decision_tree_frustrated_k])
axes[0, 0].set_title('Average Accuracy for Frustrated')

# Plot the average accuracy for Frus_Group
axes[1, 0].bar(['Logistic', 'Baseline', 'Decision Tree'], [average_accuracy_frus_group_k, average_accuracy_frus_group_baseline_k, average_accuracy_decision_tree_frus_group_k])
axes[1, 0].set_title('Average Accuracy for Frus_Group')

# Plot the accuracy for the test data for Frustrated
axes[0, 1].bar(['Logistic', 'Baseline', 'Decision Tree'], [test_accuracy_frustrated_k, test_accuracy_frustrated_baseline_k, test_accuracy_decision_tree_frustrated_k])
axes[0, 1].set_title('Accuracy for Frustrated (Test Data)')

# Plot the accuracy for the test data for Frus_Group
axes[1, 1].bar(['Logistic', 'Baseline', 'Decision Tree'], [test_accuracy_frus_group_k, test_accuracy_frus_group_baseline_k, test_accuracy_decision_tree_frus_group_k])
axes[1, 1].set_title('Accuracy for Frus_Group (Test Data)')

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()