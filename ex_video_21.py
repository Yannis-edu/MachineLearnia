import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve, train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

titanic = pd.read_excel("data/titanic3.xls")
titanic = titanic[["survived", "age", "pclass", "sex"]].dropna()
y = titanic["survived"].copy()
X = titanic[["pclass", "age"]].copy()
X["sex"] = titanic["sex"].astype("category").cat.codes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

param_grid = {
    "n_neighbors": np.arange(1, 20),
    "metric": ["euclidean", "manhattan"],
    "weights": ["distance", "uniform"],
}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best score:", grid.best_score_)
print("Best params:", grid.best_params_)

model = grid.best_estimator_
print("Test score:", model.score(X_test, y_test))

N, train_score, val_score = learning_curve(
    model, X_train, y_train, train_sizes=np.linspace(0.2, 1.0, 5), cv=5
)

plt.plot(N, train_score.mean(axis=1), label="train")
plt.plot(N, val_score.mean(axis=1), label="validation")
plt.xlabel("Train sizes")
plt.legend()
plt.show()
