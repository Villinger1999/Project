from Load_data import*

# Models
X_f = df[['HR_Max']]
y_f = df['Frustrated']

# Models
X_fg = df[['HR_Max']]
y_fg = df['Frus_Group']

# Split the data into training and testing sets for Frustrated
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_f, y_f, test_size=0.2, random_state=40)

# Split the data into training and testing sets for Frus_Group
X_train_fg, X_test_fg, y_train_fg, y_test_fg = train_test_split(X_fg, y_fg, test_size=0.2, random_state=39)

# Create a logistic regression model for Frustrated
log_model_f = LogisticRegression(multi_class='multinomial', max_iter=1000000)

# Create a logistic regression model for Frus_Group
log_model_fg = LogisticRegression(max_iter=1000000)

base_model_f = DummyClassifier(strategy='most_frequent', random_state=1)

base_model_fg = DummyClassifier(strategy='most_frequent', random_state=1)

dt_model_f = DecisionTreeClassifier(max_depth=10)

dt_model_fg = DecisionTreeClassifier(max_depth=10)

