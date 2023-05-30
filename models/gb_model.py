from sklearn.ensemble import GradientBoostingClassifier


class GBModel:
    def __init__(self):
        self.model = GradientBoostingClassifier()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
