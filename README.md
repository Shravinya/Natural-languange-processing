Sentiment Analysis using IMDB Dataset
Project Overview
This project performs sentiment analysis on movie reviews from the IMDB dataset. The goal is to classify movie reviews as either positive or negative using a machine learning model. Techniques like text preprocessing, feature extraction using TF-IDF, and model training with Logistic Regression are used.

Dataset
Name: IMDB Movie Reviews Dataset
Source: IMDB Dataset
Description: The dataset contains 50,000 highly polar movie reviews, where each review is labeled as either "positive" or "negative."
Columns:
review: The text of the movie review.
sentiment: Sentiment associated with the review (positive or negative).
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
You can install the required packages using the following command:

Project Workflow
1. Data Loading
The dataset is loaded using pandas.
Sample reviews and sentiments are displayed to get an understanding of the data.
2. Data Preprocessing
HTML tags, punctuation, and special characters are removed from the reviews.
All text is converted to lowercase.
Stopwords are removed using nltk to reduce noise in the data.
3. Feature Extraction
The cleaned reviews are transformed into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) method.
This representation is used as input to the machine learning model.
4. Model Building
A Logistic Regression model is trained to predict the sentiment (positive or negative) of a movie review.
The dataset is split into training and testing sets (80% training, 20% testing).
5. Model Evaluation
After training, the model is evaluated using various metrics:
Confusion Matrix: To visualize the correct and incorrect predictions.
Precision, Recall, F1-Score: Metrics for model performance.
Learning Curve: To assess how the model's performance improves with more training data.
ROC Curve: To visualize the trade-off between true positive and false positive rates.
Visualizations
Several visualizations are provided in the notebook to better understand the model's performance:

Confusion Matrix: Shows correct vs. incorrect predictions.
Classification Report Bar Plot: Compares precision, recall, and F1-scores for positive and negative classes.
Learning Curve: Plots the change in accuracy as the training size increases.
ROC Curve: Shows the trade-off between sensitivity and specificity.
Results
The Logistic Regression model achieved a high level of accuracy and performed well on both the training and test sets.
Validation Accuracy: Around ~88% on the test set.
Further improvements can be achieved by tuning hyperparameters or trying different models (e.g., SVM, Random Forest).

Conclusion
This project demonstrates how to perform sentiment analysis on text data using machine learning techniques. By preprocessing text, extracting features, and training a classification model, we can accurately predict the sentiment of movie reviews. Further optimization can be done to improve the model's performance.
