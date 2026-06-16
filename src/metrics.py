from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class MetricsCalculator:

    def __init__(self, dataset, tree, nn):

        self.dataset = dataset
        self.tree = tree
        self.nn = nn
        
    def _evaluate(self, y_pred):
        
        return {
            "acuracia": accuracy_score(self.dataset.y_test, y_pred),
            "precisao": precision_score(self.dataset.y_test, y_pred),
            "recall": recall_score(self.dataset.y_test, y_pred),
            "f1": f1_score(self.dataset.y_test, y_pred),
            "y_pred": y_pred,
        }

    def calculate_tree(self):
        return self._evaluate(self.tree.predict(self.dataset.X_test))

    def calculate_nn(self):
        return self._evaluate(self.nn.predict(self.dataset.X_test_scaled))