from Load_data import*
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt

# Logistic regression model for Frustrated
X_frustrated = df[['HR_Mean', 'HR_Median', 'HR_std', 'HR_Min', 'HR_Max','HR_AUC']]
y_frustrated = df['Frustrated']

# Logistic regression model for Frus_Group
X_frus_group = df[['HR_Mean', 'HR_Median', 'HR_std', 'HR_Min', 'HR_Max','HR_AUC']]
y_frus_group = df['Frus_Group']

# Split the data into training and testing sets for Frustrated
X_train_frustrated, X_test_frustrated, y_train_frustrated, y_test_frustrated = train_test_split(X_frustrated, y_frustrated, test_size=0.1, random_state=42)

# Split the data into training and testing sets for Frus_Group
X_train_frus_group, X_test_frus_group, y_train_frus_group, y_test_frus_group = train_test_split(X_frus_group, y_frus_group, test_size=0.1, random_state=42)

# Create a logistic regression model for Frustrated
model_frustrated = LogisticRegression(multi_class='multinomial', max_iter=1000000)

# Create a logistic regression model for Frus_Group
model_frus_group = LogisticRegression(multi_class='multinomial', max_iter=1000000)

baseline_frustrated = DummyClassifier(strategy='most_frequent', random_state=1)

baseline_frus = DummyClassifier(strategy='most_frequent', random_state=1)

model_decision_tree_frustrated = DecisionTreeClassifier(max_depth=10)

model_decision_tree_frus_group = DecisionTreeClassifier(max_depth=10)

# Leave one out
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# Train and evaluate the model using stratified k-fold cross-validation for Frustrated
accuracies_frustrated = []
accuracies_frustrated_baseline = []
accuracies_frustrated_decision_tree = []
for train_index, val_index in skf.split(X_train_frustrated, y_train_frustrated):
    X_train_fold, X_val_fold = X_train_frustrated.iloc[train_index], X_train_frustrated.iloc[val_index]
    y_train_fold, y_val_fold = y_train_frustrated.iloc[train_index], y_train_frustrated.iloc[val_index]
    
    # Train the model on the training fold
    model_frustrated.fit(X_train_fold, y_train_fold)
    baseline_frustrated.fit(X_train_fold, y_train_fold)
    model_decision_tree_frustrated.fit(X_train_fold, y_train_fold)
    
    
    # Evaluate the model on the validation fold
    accuracy = model_frustrated.score(X_val_fold, y_val_fold)
    accuracy_baseline = baseline_frustrated.score(X_val_fold, y_val_fold)
    accuracy_decision_tree = model_decision_tree_frustrated.score(X_val_fold, y_val_fold)

    accuracies_frustrated.append(accuracy)
    accuracies_frustrated_baseline.append(accuracy_baseline)
    accuracies_frustrated_decision_tree.append(accuracy_decision_tree)

# Train and evaluate the model using stratified k-fold cross-validation for Frus_Group
accuracies_frus_group = []
accuracies_frus_group_baseline = []
accuracies_frus_group_decision_tree = []
for train_index, val_index in skf.split(X_train_frus_group, y_train_frus_group):
    X_train_fold, X_val_fold = X_train_frus_group.iloc[train_index], X_train_frus_group.iloc[val_index]
    y_train_fold, y_val_fold = y_train_frus_group.iloc[train_index], y_train_frus_group.iloc[val_index]
    
    # Train the model on the training fold
    model_frus_group.fit(X_train_fold, y_train_fold)
    baseline_frus.fit(X_train_fold, y_train_fold)
    model_decision_tree_frus_group.fit(X_train_fold, y_train_fold)
    
    # Evaluate the model on the validation fold
    accuracy = model_frus_group.score(X_val_fold, y_val_fold)
    accuracy_baseline = baseline_frus.score(X_val_fold, y_val_fold)
    accuracy_decision_tree = model_decision_tree_frus_group.score(X_val_fold, y_val_fold)

    accuracies_frus_group.append(accuracy)
    accuracies_frus_group_baseline.append(accuracy_baseline)
    accuracies_frus_group_decision_tree.append(accuracy_decision_tree)


# Calculate the average accuracy across all folds for Frustrated
average_accuracy_frustrated = sum(accuracies_frustrated) / len(accuracies_frustrated)
average_accuracy_frustrated_baseline = sum(accuracies_frustrated_baseline) / len(accuracies_frustrated_baseline)
average_accuracy_decision_tree_frustrated = sum(accuracies_frustrated_decision_tree) / len(accuracies_frustrated_decision_tree)
# print("Average Accuracy for Frustrated:", average_accuracy_frustrated)
# print("Average Accuracy for Frustrated Baseline:", average_accuracy_frustrated_baseline)
# print("Average Accuracy for Frustrated Decision Tree:", average_accuracy_decision_tree_frustrated)

# Calculate the average accuracy across all folds for Frus_Group
average_accuracy_frus_group = sum(accuracies_frus_group) / len(accuracies_frus_group)
average_accuracy_frus_group_baseline = sum(accuracies_frus_group_baseline) / len(accuracies_frus_group_baseline)
average_accuracy_decision_tree_frus_group = sum(accuracies_frus_group_decision_tree) / len(accuracies_frus_group_decision_tree)
# print("Average Accuracy for Frus_Group:", average_accuracy_frus_group)
# print("Average Accuracy for Frus_Group Baseline:", average_accuracy_frus_group_baseline)
# print("Average Accuracy for Frus_Group Decision Tree:", average_accuracy_decision_tree_frus_group)



# Print predictions for Frustrated
y_pred_frustrated = model_frustrated.predict(X_test_frustrated)
# print("Number of predictions for Frustrated:", len(y_pred_frustrated), len(y_test_frustrated))

# Print predictions for Frus_Group
y_pred_frus_group = model_frus_group.predict(X_test_frus_group)
# print("Number of predictions for Frus_Group:", len(y_pred_frus_group), len(y_test_frus_group))

y_pred_frustrated_baseline = baseline_frustrated.predict(X_test_frustrated)
y_pred_frus_group_baseline = baseline_frus.predict(X_test_frus_group)

# Calculate the accuracy for the test data for Frustrated
test_accuracy_frustrated = model_frustrated.score(X_test_frustrated, y_test_frustrated)
# print("Accuracy for Frustrated (Test Data):", test_accuracy_frustrated)

# Calculate the accuracy for the test data for Frus_Group
test_accuracy_frus_group = model_frus_group.score(X_test_frus_group, y_test_frus_group)
# print("Accuracy for Frus_Group (Test Data):", test_accuracy_frus_group)

# Calculate the accuracy for the test data for Frustrated Baseline
test_accuracy_frustrated_baseline = baseline_frustrated.score(X_test_frustrated, y_test_frustrated)
# print("Accuracy for Frustrated Baseline (Test Data):", test_accuracy_frustrated_baseline)

# Calculate the accuracy for the test data for Frus_Group Baseline
test_accuracy_frus_group_baseline = baseline_frus.score(X_test_frus_group, y_test_frus_group)
# print("Accuracy for Frus_Group Baseline (Test Data):", test_accuracy_frus_group_baseline)

# Calculate the accuracy for the test data for Frustrated Decision Tree
test_accuracy_decision_tree_frustrated = model_decision_tree_frustrated.score(X_test_frustrated, y_test_frustrated)
# print("Accuracy for Frustrated Decision Tree (Test Data):", test_accuracy_decision_tree_frustrated)

# Calculate the accuracy for the test data for Frus_Group Decision Tree
test_accuracy_decision_tree_frus_group = model_decision_tree_frus_group.score(X_test_frus_group, y_test_frus_group)
# print("Accuracy for Frus_Group Decision Tree (Test Data):", test_accuracy_decision_tree_frus_group)


# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Plot the average accuracy for Frustrated
axes[0, 0].bar(['Model', 'Baseline', 'Decision Tree'], [average_accuracy_frustrated, average_accuracy_frustrated_baseline, average_accuracy_decision_tree_frustrated])
axes[0, 0].set_title('Average Accuracy for Frustrated')

# Plot the average accuracy for Frus_Group
axes[0, 1].bar(['Model', 'Baseline', 'Decision Tree'], [average_accuracy_frus_group, average_accuracy_frus_group_baseline, average_accuracy_decision_tree_frus_group])
axes[0, 1].set_title('Average Accuracy for Frus_Group')

# Plot the accuracy for the test data for Frustrated
axes[1, 0].bar(['Model', 'Baseline', 'Decision Tree'], [test_accuracy_frustrated, test_accuracy_frustrated_baseline, test_accuracy_decision_tree_frustrated])
axes[1, 0].set_title('Accuracy for Frustrated (Test Data)')

# Plot the accuracy for the test data for Frus_Group
axes[1, 1].bar(['Model', 'Baseline', 'Decision Tree'], [test_accuracy_frus_group, test_accuracy_frus_group_baseline, test_accuracy_decision_tree_frus_group])
axes[1, 1].set_title('Accuracy for Frus_Group (Test Data)')

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()