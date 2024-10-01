Sentiment Analysis using IMDB Dataset
Project Overview
Perform sentiment analysis on movie reviews from the IMDB dataset.
Classify reviews as either positive or negative using a machine learning model.
Utilize techniques like:
Text preprocessing
Feature extraction using TF-IDF
Model training with Logistic Regression
Dataset
Name: IMDB Movie Reviews Dataset
Source: IMDB Dataset
Description: Contains 50,000 highly polar movie reviews, labeled as either "positive" or "negative."
Columns:
review: Text of the movie review
sentiment: Sentiment associated with the review (positive or negative)
Project Structure
Libraries:
pandas
numpy
sklearn (scikit-learn)
matplotlib
seaborn
re (for regular expressions)
nltk (for natural language processing)
google.colab (for file uploading in Google Colab)
Project Workflow
Data Loading:

Load the dataset using pandas.
Display sample reviews and sentiments to understand the data.
Data Preprocessing:

Remove HTML tags, punctuation, and special characters from reviews.
Convert all text to lowercase.
Remove stopwords using nltk to reduce noise in the data.
Feature Extraction:

Transform cleaned reviews into numerical features using the TF-IDF method.
Use this representation as input for the machine learning model.
Model Building:

Train a Logistic Regression model to predict sentiment (positive or negative).
Split the dataset into training (80%) and testing sets (20%).
Model Evaluation:

Evaluate the model using various metrics:
Confusion Matrix: Visualize correct and incorrect predictions.
Precision, Recall, F1-Score: Assess model performance.
Learning Curve: Analyze how model performance improves with more training data.
ROC Curve: Visualize trade-offs between true positive and false positive rates.
Visualizations
Confusion Matrix: Displays correct vs. incorrect predictions.
Classification Report Bar Plot: Compares precision, recall, and F1-scores for positive and negative classes.
Learning Curve: Shows change in accuracy as training size increases.
ROC Curve: Illustrates trade-offs between sensitivity and specificity.
Results
Logistic Regression model achieved high accuracy:
Validation Accuracy: Approximately 88% on the test set.
Potential for further improvements by:
Tuning hyperparameters
Exploring different models (e.g., SVM, Random Forest)
Conclusion
Demonstrates how to perform sentiment analysis on text data using machine learning techniques.
Highlights the process of:
Text preprocessing
Feature extraction
Training a classification model for sentiment prediction.
Suggests that further optimizations can enhance model performance.
