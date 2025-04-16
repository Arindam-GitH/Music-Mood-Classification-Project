import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import json
from preprocessing import load_and_preprocess_data
from visualization import generate_all_visualizations

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

def train_and_evaluate_models(data_dict):
    """
    Train and evaluate multiple classification models.
    """
    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    feature_names = data_dict['feature_names']
    label_encoder = data_dict['label_encoder']
    
    # Initialize models
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(kernel='rbf', probability=True),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=1000,
            random_state=42
        )
    }
    
    # Dictionary to store results
    results = {}
    
    # Train and evaluate all models
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist()
        }
        
        # Get feature importance if available
        if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
            importance = get_feature_importance(model, feature_names)
            if importance is not None:
                results[name]['feature_importance'] = importance.to_dict()
    
    # Save results
    os.makedirs('../data', exist_ok=True)
    with open('../data/model_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def print_results(results):
    """
    Print the results of all models.
    """
    print("\nModel Performance Summary:")
    print("=" * 50)
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print("\nClassification Report:")
        print(result['classification_report'])
        print("\nConfusion Matrix:")
        print(np.array(result['confusion_matrix']))
        print("-" * 50)

if __name__ == "__main__":
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data_dict = load_and_preprocess_data()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    df = pd.read_csv('../data/generated_dataset.csv')
    generate_all_visualizations(df, data_dict)
    
    # Train and evaluate models
    print("\nTraining and evaluating models...")
    results = train_and_evaluate_models(data_dict)
    
    # Print results
    print_results(results) 