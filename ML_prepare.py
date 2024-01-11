from Load_data import df
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

# Models
X_f = df[['HR_Mean', 'HR_Median', 'HR_std', 'HR_Min', 'HR_Max','HR_AUC']]
y_f = df['Frustrated']

# Models
X_fg = df[['HR_Mean', 'HR_Median', 'HR_std', 'HR_Min', 'HR_Max','HR_AUC']]
y_fg = df['Frus_Group']

# Split the data into training and testing sets for Frustrated
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_f, y_f, test_size=0.1, random_state=42)

# Split the data into training and testing sets for Frus_Group
X_train_fg, X_test_fg, y_train_fg, y_test_fg = train_test_split(X_fg, y_fg, test_size=0.1, random_state=42)

# Create a logistic regression model for Frustrated
log_model_f = LogisticRegression(multi_class='multinomial', max_iter=1000000)

# Create a logistic regression model for Frus_Group
log_model_fg = LogisticRegression(multi_class='multinomial', max_iter=1000000)

base_f = DummyClassifier(strategy='most_frequent', random_state=1)

base_fg = DummyClassifier(strategy='most_frequent', random_state=1)

dt_model_f = DecisionTreeClassifier(max_depth=10)

dt_model_fg = DecisionTreeClassifier(max_depth=10)