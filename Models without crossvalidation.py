from Load_data import *
from ML_prepare import *

X = df[["HR_Max"]]
y_f = df["Frustrated"]
y_fg = df["Frus_Group"]


#Frustrated
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X, y_f, test_size=0.2, random_state=42)
decisiontree = DecisionTreeClassifier()
logistic = LogisticRegression(max_iter=1000)
base = DummyClassifier(strategy='most_frequent', random_state=1)
decisiontree.fit(X_train_f, y_train_f)
logistic.fit(X_train_f, y_train_f)
base.fit(X_train_f, y_train_f)
y_pred = decisiontree.predict(X_test_f)
y_pred2 = logistic.predict(X_test_f)
score = base.score(X_train_f, y_train_f)
score1 = base.score(X_test_f, y_test_f)
score2 = decisiontree.score(X_train_f, y_train_f)
score3 = decisiontree.score(X_test_f, y_test_f)
score4 = logistic.score(X_train_f, y_train_f)
score5 = logistic.score(X_test_f, y_test_f)

print(score, score1, score2, score3, score4, score5)


#Frus_Group
X_train_fg, X_test_fg, y_train_fg, y_test_fg = train_test_split(X, y_fg, test_size=0.2, random_state=42)
decisiontree = DecisionTreeClassifier()
logistic = LogisticRegression(max_iter=1000)
base = DummyClassifier(strategy='most_frequent', random_state=1)
decisiontree.fit(X_train_fg, y_train_fg)
logistic.fit(X_train_fg, y_train_fg)
base.fit(X_train_fg, y_train_fg)
y_pred = decisiontree.predict(X_test_fg)
y_pred2 = logistic.predict(X_test_fg)
score = base.score(X_train_fg, y_train_fg)
score1 = base.score(X_test_fg, y_test_fg)
score2 = decisiontree.score(X_train_fg, y_train_fg)
score3 = decisiontree.score(X_test_fg, y_test_fg)
score4 = logistic.score(X_train_fg, y_train_fg)
score5 = logistic.score(X_test_fg, y_test_fg)

print(score, score1, score2, score3, score4, score5)