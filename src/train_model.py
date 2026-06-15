import joblib

from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


def train_tree(X_train, y_train):
    
    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    return model

def train_nn(X_train_scaled, y_train):
    
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
    model.fit(X_train_scaled, y_train)
    
    return model

def save_models(tree, nn, scaler):
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(tree, MODELS_DIR / 'decision_tree.pkl')
    joblib.dump(nn, MODELS_DIR / "mlp_classifier.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    
def train_models(data):
    
    tree = train_tree(data["X_train"], data["y_train"])
    nn = train_nn(data["X_train_scaled"], data["y_train"])
    
    return tree, nn