from Load_data_phase import *

# Models
X_f = df[['HR_AUC', 'Frustrated']]
y_f = df['Phase']


# Split the data into training and testing sets for Frustrated
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_f, y_f, test_size=0.2, random_state=42)

# Create a logistic regression model for Frustrated
log_model_f = LogisticRegression(multi_class='multinomial', max_iter=1000)

base_model_f = DummyClassifier(strategy='most_frequent', random_state=1)

dt_model_f = DecisionTreeClassifier(max_depth=10)



