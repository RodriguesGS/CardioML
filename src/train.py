import joblib

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from .config import MODELS_DIR

class ModelTrain:
    
    def __init__(self, dataset):
        
        self.dataset = dataset
        
        self.tree = None
        self.nn = None
        
    def train_tree(self):
        
        model = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        model.fit(self.dataset.X_train, self.dataset.y_train)
        
        self.tree = model
        return model
    
    def train_nn(self):
        
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
        )
        model.fit(self.dataset.X_train_scaled, self.dataset.y_train)
        
        self.nn = model
        return model
        
    def save_models(self):
        
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.tree, MODELS_DIR / 'decision_tree.pkl')
        joblib.dump(self.nn, MODELS_DIR / "mlp_classifier.pkl")
        joblib.dump(self.dataset.scaler, MODELS_DIR / "scaler.pkl")

    