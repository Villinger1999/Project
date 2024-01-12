from Load_data_phase import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss
from sklearn.metrics import accuracy_score

X = df[["HR_Mean", "HR_Median", "HR_std", "HR_Min", "HR_Max", "HR_AUC"]]
y = df["Phase"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
decisiontree = DecisionTreeClassifier()
logistic = LogisticRegression(max_iter=1000)
decisiontree.fit(X_train, y_train)
logistic.fit(X_train, y_train)
y_pred = decisiontree.predict(X_test)
y_pred2 = logistic.predict(X_test)
score = decisiontree.score(X_train, y_train)
score1 = decisiontree.score(X_test, y_test)
score4 = logistic.score(X_train, y_train)
score5 = logistic.score(X_test, y_test)

print(score, score1, score4, score5)