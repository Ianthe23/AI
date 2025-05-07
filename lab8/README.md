# Facebook Sentiment Analysis Project

This project implements sentiment analysis capabilities for Facebook's content platform to detect the emotional tone of user messages. The project addresses the need to filter feeds by sentiment, providing users with more positive content.

## Features

- **Azure Sentiment Analysis**: Uses Azure Text Analytics API to detect sentiment in messages
- **Custom Neural Network Implementation**:
  - Uses TensorFlow/Keras for a powerful, scalable solution
  - Includes a custom ANN implementation from scratch (without using a neural network library)
- **Feature Extraction Methods**:
  - Bag of Words (BoW)
  - Term Frequency-Inverse Document Frequency (TF-IDF)
  - Word2Vec embeddings

## Requirements

Install all required dependencies with:

```
pip install -r requirements.txt
```

## Usage

### Main Sentiment Analysis Tool

Run the main script for a complete analysis using Azure and TensorFlow:

```
python sentiment_analysis.py
```

### Custom Neural Network Implementation

Run the custom neural network implementation:

```
python custom_ann.py
```

## Dataset

The system is designed to work with labeled emotion datasets, such as:

- The provided review_mixed.csv (expected to be in the data/ directory)
- Datasets from [GitHub: unify-emotion-datasets](https://github.com/sarnthil/unify-emotion-datasets/tree/master/datasets)

## Example Message

The project analyzes the following message:

> "By choosing a bike over a car, I'm reducing my environmental footprint. Cycling promotes eco-friendly transportation, and I'm proud to be part of that movement."

## Comparing Performance

After running both implementations, you can compare the performance of:

1. Azure Text Analytics service
2. TensorFlow-based neural network
3. Custom-implemented neural network

The scripts will show the predicted sentiment for each approach along with confidence scores.
