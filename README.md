# Twitter Sentiment Analysis Using Logistic Regression

This project aims to perform sentiment analysis on tweets using a dataset from Kaggle. The goal is to classify tweets as either positive or negative based on their textual content.

## Requirements

Before running the project, you'll need to install the required dependencies. Make sure you have the following libraries installed:

- `kaggle` - For downloading the dataset from Kaggle.
- `numpy` - For numerical operations.
- `pandas` - For data manipulation and analysis.
- `nltk` - For natural language processing tasks (such as stopwords and stemming).
- `scikit-learn` - For machine learning models and data preprocessing.

## Dataset

The dataset used in this project is the **Sentiment140** dataset, which contains 1.6 million tweets labeled as positive or negative. You can find it on Kaggle [here](https://www.kaggle.com/datasets/kazanova/sentiment140).

The dataset consists of the following columns:
- `target`: Sentiment of the tweet (0 for negative, 1 for positive).
- `id`: Unique tweet identifier.
- `date`: Date of tweet.
- `flag`: Query flag (not used here).
- `user`: Twitter handle of the user.
- `text`: The tweet content.

## Model Overview

We use a **Logistic Regression** model to classify tweets into two categories: positive and negative. The steps involved in the project include:

1. **Data Preprocessing**: 
   - Clean the text data by removing non-alphabetic characters.
   - Convert all text to lowercase.
   - Perform stemming on words to reduce them to their root form.
   - Remove stopwords (common words like "the", "is", etc., that don't contribute much to sentiment analysis).

2. **Feature Extraction**: 
   - Convert the textual data into numerical vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)**, which helps to reflect the importance of words in the context of the dataset.

3. **Model Training**: 
   - Split the dataset into training and testing sets.
   - Train the Logistic Regression model on the training data.

4. **Evaluation**:
   - Evaluate the performance of the model using accuracy scores on both the training and test sets.

5. **Model Saving**:
   - Save the trained model for future predictions.

## How to Use

1. **Download the Dataset**:
   - Use Kaggle’s API to download the Sentiment140 dataset.

2. **Preprocessing the Data**:
   - Clean the tweet texts by removing irrelevant characters and stopwords.
   - Apply stemming to reduce words to their base form.

3. **Model Training**:
   - Train the Logistic Regression model using the processed data.

4. **Evaluate the Model**:
   - Check the model's accuracy on both the training and testing datasets.

5. **Make Predictions**:
   - Load the saved model and use it to classify new tweets as either positive or negative.

## Performance

The model achieved an accuracy score of around 79.87% on the training data and 77.67% on the test data. This indicates that the model generalizes well and is not overfitting.

## Future Improvements

- **Hyperparameter Tuning**: Explore different values for Logistic Regression’s parameters to improve performance.
- **Deep Learning Models**: Investigate the use of neural networks for potentially better accuracy.
- **Data Augmentation**: Use techniques like data augmentation to balance the dataset or improve diversity.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The dataset used in this project is provided by [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140).
- Thanks to Kaggle for making the dataset available for research.
