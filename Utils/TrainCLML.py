from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score


def train_ml(latent, targets):
    classifiers = {
        'RandomForest': RandomForestClassifier(),
        'KNeighbors': KNeighborsClassifier(),
        'LogisticRegression': LogisticRegression(),
        'DecisionTree': DecisionTreeClassifier()
    }

    for name, classifier in classifiers.items():
        classifier.fit(latent, targets)
        prediction = classifier.predict(latent)
        acc = accuracy_score(targets, prediction)
        print(f"{name} Accuracy: {acc}")
