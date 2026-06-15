import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from .config import DATA_PATH, DATA_URL, DATA_DIR

@dataclass
class Dataset:
    
    df: object
    
    X_train: object
    X_test: object
    
    y_train: object
    y_test: object
    
    X_train_scaled: object
    X_test_scaled: object
    
    scaler: object
    
class DataProcessor:
    
    def download_dataset(self):
        
        if DATA_PATH.exists():
            return pd.read_csv(DATA_PATH)
    
        df = pd.read_csv(DATA_URL)
        
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        
        return df
    
    def process(self):
        df = self.download_dataset()
    
        X = df.drop('target', axis=1)
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return Dataset(
            df=df,

            X_train=X_train,
            X_test=X_test,

            y_train=y_train,
            y_test=y_test,

            X_train_scaled=X_train_scaled,
            X_test_scaled=X_test_scaled,

            scaler=scaler
        )
        