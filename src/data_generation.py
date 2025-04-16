import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def generate_music_dataset(n_samples=10000, random_state=42):
    """
    Generate synthetic music dataset with realistic feature distributions.
    """
    np.random.seed(random_state)
    
    # Define mood classes
    moods = ['happy', 'sad', 'calm', 'energetic']
    
    # Generate features with mood-specific distributions
    data = []
    
    for _ in range(n_samples):
        mood = np.random.choice(moods)
        
        # Generate features based on mood
        if mood == 'happy':
            tempo = np.random.normal(120, 10)
            energy = np.random.normal(0.8, 0.1)
            danceability = np.random.normal(0.7, 0.1)
            acousticness = np.random.normal(0.3, 0.1)
            valence = np.random.normal(0.8, 0.1)
            lyrics_sentiment = np.random.normal(0.7, 0.1)
            
        elif mood == 'sad':
            tempo = np.random.normal(70, 10)
            energy = np.random.normal(0.3, 0.1)
            danceability = np.random.normal(0.3, 0.1)
            acousticness = np.random.normal(0.7, 0.1)
            valence = np.random.normal(0.2, 0.1)
            lyrics_sentiment = np.random.normal(0.2, 0.1)
            
        elif mood == 'calm':
            tempo = np.random.normal(85, 10)
            energy = np.random.normal(0.4, 0.1)
            danceability = np.random.normal(0.4, 0.1)
            acousticness = np.random.normal(0.8, 0.1)
            valence = np.random.normal(0.5, 0.1)
            lyrics_sentiment = np.random.normal(0.5, 0.1)
            
        else:  # energetic
            tempo = np.random.normal(140, 10)
            energy = np.random.normal(0.9, 0.1)
            danceability = np.random.normal(0.8, 0.1)
            acousticness = np.random.normal(0.2, 0.1)
            valence = np.random.normal(0.7, 0.1)
            lyrics_sentiment = np.random.normal(0.6, 0.1)
        
        # Clip values to valid ranges
        tempo = np.clip(tempo, 60, 200)
        energy = np.clip(energy, 0, 1)
        danceability = np.clip(danceability, 0, 1)
        acousticness = np.clip(acousticness, 0, 1)
        valence = np.clip(valence, 0, 1)
        lyrics_sentiment = np.clip(lyrics_sentiment, 0, 1)
        
        # Add some random genre information
        genres = ['pop', 'rock', 'classical', 'electronic', 'jazz']
        genre = np.random.choice(genres)
        
        data.append({
            'tempo': tempo,
            'energy': energy,
            'danceability': danceability,
            'acousticness': acousticness,
            'valence': valence,
            'lyrics_sentiment': lyrics_sentiment,
            'genre': genre,
            'mood': mood
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    os.makedirs('../data', exist_ok=True)
    df.to_csv('../data/generated_dataset.csv', index=False)
    
    return df

if __name__ == "__main__":
    print("Generating music dataset...")
    df = generate_music_dataset()
    print(f"Dataset generated with {len(df)} samples")
    print("\nSample of the dataset:")
    print(df.head())
    print("\nDataset statistics:")
    print(df.describe()) 