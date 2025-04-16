import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os

def create_radar_chart(df, mood_classes, feature_cols, save_path='../data/radar_chart.png'):
    """
    Create a radar chart showing average feature profiles per mood.
    """
    # Calculate mean values for each mood
    mood_means = df.groupby('mood')[feature_cols].mean()
    
    # Number of features
    num_features = len(feature_cols)
    
    # Compute angle for each axis
    angles = [n / float(num_features) * 2 * np.pi for n in range(num_features)]
    angles += angles[:1]  # Complete the circle
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot for each mood
    for mood in mood_classes:
        values = mood_means.loc[mood].values.flatten().tolist()
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, linewidth=2, label=mood)
        ax.fill(angles, values, alpha=0.25)
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_cols)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def create_3d_pca_plot(X_pca, y, label_encoder, save_path='../data/pca_3d.png'):
    """
    Create a 3D scatter plot using PCA components.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get unique labels and colors
    labels = label_encoder.classes_
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    
    # Plot each class
    for i, label in enumerate(labels):
        mask = (y == i)
        if len(mask) > len(X_pca):
            mask = mask[:len(X_pca)]
        elif len(mask) < len(X_pca):
            X_pca = X_pca[:len(mask)]
            
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                  c=[colors[i]], label=label, alpha=0.6)
    
    # Customize the plot
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('3D PCA Visualization of Music Moods')
    
    # Add legend
    ax.legend()
    
    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def create_stacked_bar_chart(df, save_path='../data/stacked_bar.png'):
    """
    Create a stacked bar chart showing mood distribution across genres.
    """
    # Create cross-tabulation
    mood_genre = pd.crosstab(df['genre'], df['mood'])
    
    # Create stacked bar chart
    ax = mood_genre.plot(kind='bar', stacked=True, figsize=(12, 6))
    
    # Customize the plot
    plt.title('Mood Distribution Across Genres')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.legend(title='Mood', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def create_correlation_heatmap(df, feature_cols, save_path='../data/correlation_heatmap.png'):
    """
    Create a heatmap showing correlation between features and mood.
    """
    # Create correlation matrix
    corr_matrix = df[feature_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    
    # Customize the plot
    plt.title('Feature Correlation Heatmap')
    
    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def generate_all_visualizations(df, data_dict):
    """
    Generate all visualizations for the dataset.
    """
    feature_cols = ['tempo', 'energy', 'danceability', 'acousticness', 'valence', 'lyrics_sentiment']
    mood_classes = df['mood'].unique()
    
    # Create visualizations
    print("Creating radar chart...")
    create_radar_chart(df, mood_classes, feature_cols)
    
    print("Creating 3D PCA plot...")
    create_3d_pca_plot(data_dict['X_pca'], data_dict['y_train'], data_dict['label_encoder'])
    
    print("Creating stacked bar chart...")
    create_stacked_bar_chart(df)
    
    print("Creating correlation heatmap...")
    create_correlation_heatmap(df, feature_cols)
    
    print("All visualizations have been generated and saved in the data directory.")

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('../data/generated_dataset.csv')
    data_dict = load_and_preprocess_data()
    
    # Generate visualizations
    generate_all_visualizations(df, data_dict) 