from ML_prepare_phase import *

base_model_f.fit(X_train_f, y_train_f)
dt_model_f.fit(X_train_f, y_train_f)
log_model_f.fit(X_train_f, y_train_f)

score = base_model_f.score(X_train_f, y_train_f)
score1 = base_model_f.score(X_test_f, y_test_f)
score2 = dt_model_f.score(X_train_f, y_train_f)
score3 = dt_model_f.score(X_test_f, y_test_f)
score4 = log_model_f.score(X_train_f, y_train_f)
score5 = log_model_f.score(X_test_f, y_test_f)

print(score, score1, score2, score3, score4, score5)