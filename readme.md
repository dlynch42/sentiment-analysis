# Sentiment Analysis RNN Project

## Project Overview

This project implements a Recurrent Neural Network (RNN) model for sentiment analysis using TensorFlow and Keras. The goal is to classify text data into positive, neutral, or negative sentiments. We use GridSearchCV to fine-tune hyperparameters and improve the model's accuracy.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Hyperparameters](#hyperparameters)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)

## Installation

To get started, clone this repository and install the necessary packages:

```bash
git clone https://github.com/dlynch42/sentiment-analysis.git
cd sentiment-analysis
pip install -r requirements.txt
```

### Requirements

- pandas
- numpy
- tensorflow
- sklearn
- pyprind
- re

Install these packages using pip:

```bash
pip install pandas numpy tensorflow scikit-learn pyprind matplotlib seaborn
```

## Usage

To train the model and make predictions, follow these steps:

1. **EDA & Preprocessing**: Load and preprocess the text data to remove HTML tags, URLs, special characters, and stopwords, then tokenize and stem the text. Create sequence mappings for model. 
2. **Model Architecutre**: Initialize and build the RNN model. Train model on `X_train`. Tune and adjust hyperparameters using `GridSearchCV` and `Pipe`. 
3. **Test Model**: Use the trained model to make predictions on new data.
4. **Conclusion**: Analyze results.

## Model Architecture

The model architecture consists of the following layers:

1. **Embedding Layer**: Converts input sequences into dense vectors of fixed size.
2. **LSTM Layers**: Captures dependencies in both forward and backward directions using bidirectional LSTM. Multiple layers can be stacked for deeper representations.
3. **Dropout Layers**: Helps prevent overfitting.
4. **Dense Output Layer**: Outputs the final sentiment prediction.
5. **Optimizer & Loss Function**: Used Adam to optimize and BCE to calculate loss

## Hyperparameters

We focused on the following hyperparameters to optimize the model:

- `seq_len`: Sequence length of the input data.
- `lstm_size`: Number of units in the LSTM layers.
- `num_layers`: Number of LSTM layers.
- `batch_size`: Size of the batches during training.
- `learning_rate`: Learning rate for the optimizer.

## Results

The best model achieved a test **accuracy of 73.45%**. The results were lower than expected, likely due to over-processing the data. The original text sequences were short, and excessive preprocessing reduced the amount of useful data.