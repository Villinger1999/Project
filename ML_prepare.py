from Load_data import df
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

# Models
X_frustrated = df[['HR_Mean', 'HR_Median', 'HR_std', 'HR_Min', 'HR_Max','HR_AUC']]
y_frustrated = df['Frustrated']

# Models
X_frus_group = df[['HR_Mean', 'HR_Median', 'HR_std', 'HR_Min', 'HR_Max','HR_AUC']]
y_frus_group = df['Frus_Group']

# Split the data into training and testing sets for Frustrated
X_train_frustrated, X_test_frustrated, y_train_frustrated, y_test_frustrated = train_test_split(X_frustrated, y_frustrated, test_size=0.1, random_state=42)

# Split the data into training and testing sets for Frus_Group
X_train_frus_group, X_test_frus_group, y_train_frus_group, y_test_frus_group = train_test_split(X_frus_group, y_frus_group, test_size=0.1, random_state=42)

# Create a logistic regression model for Frustrated
model_frustrated = LogisticRegression(multi_class='multinomial', max_iter=1000000)

# Create a logistic regression model for Frus_Group
model_frus_group = LogisticRegression(max_iter=1000000)

baseline_frustrated = DummyClassifier(strategy='most_frequent', random_state=1)

baseline_frus = DummyClassifier(strategy='most_frequent', random_state=1)

model_decision_tree_frustrated = DecisionTreeClassifier(max_depth=10)

model_decision_tree_frus_group = DecisionTreeClassifier(max_depth=10)