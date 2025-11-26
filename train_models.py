from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        'LogisticRegression': LogisticRegression(max_iter=500),
        'RandomForest': RandomForestClassifier(n_estimators=100),
        'SVM': SVC(probability=True)
    }
    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(f"{name} accuracy: {acc:.4f}")
        scores[name] = acc
    return models, scores, (X_train, X_test, y_train, y_test)

def tune_model(X_train, y_train):
    param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
    grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
    grid.fit(X_train, y_train)
    print("Best Params:", grid.best_params_)
    return grid.best_estimator_
