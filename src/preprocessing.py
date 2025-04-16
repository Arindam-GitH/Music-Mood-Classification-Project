import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(data_path='../data/generated_dataset.csv', test_size=0.2, random_state=42):
    """
    Load and preprocess the music dataset.
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df.drop(['mood', 'genre'], axis=1)
    y = df['mood']
    
    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=test_size, random_state=random_state
    )
    
    # Apply PCA for visualization
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_pca': X_pca,
        'feature_names': X.columns,
        'label_encoder': le,
        'scaler': scaler,
        'pca': pca
    }

def get_feature_importance(model, feature_names):
    """
    Extract feature importance from models that support it.
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        return None
    
    # Create DataFrame with feature names and importance scores
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    return feature_importance.sort_values('importance', ascending=False)

if __name__ == "__main__":
    # Test the preprocessing
    data_dict = load_and_preprocess_data()
    print("Data preprocessing completed successfully!")
    print(f"Training set shape: {data_dict['X_train'].shape}")
    print(f"Test set shape: {data_dict['X_test'].shape}")
    print("\nFeature names:")
    print(data_dict['feature_names'].tolist()) 