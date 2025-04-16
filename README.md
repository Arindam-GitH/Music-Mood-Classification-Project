# Music Mood Classification

This project implements a machine learning system for classifying music moods based on audio features. The system analyzes various musical characteristics to predict the mood of songs into categories like happy, sad, calm, and energetic.

## Project Structure

```
music_mood_classification/
├── data/
│   └── generated_dataset.csv
├── models/
│   └── saved_models/
├── notebooks/
│   └── model_exploration.ipynb
├── src/
│   ├── data_generation.py
│   ├── preprocessing.py
│   ├── visualization.py
│   ├── model_training.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Features

The dataset includes the following audio features:
- tempo: Speed of the music (BPM)
- energy: Intensity and activity level
- danceability: How suitable for dancing
- acousticness: How acoustic the track is
- valence: Musical positiveness
- lyrics_sentiment: Sentiment analysis of lyrics

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Generate the dataset:
   ```bash
   python src/data_generation.py
   ```

2. Run the complete analysis:
   ```bash
   python src/model_training.py
   ```

## Models Implemented

1. K-Nearest Neighbors (KNN)
2. Support Vector Machine (SVM)
3. Random Forest Classifier
4. Neural Network (TensorFlow)

## Results

The models are evaluated using:
- Accuracy
- Classification Report
- Confusion Matrix

Detailed results and visualizations are available in the notebooks directory.

## License

MIT License 