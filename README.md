# Sentiment Analysis of Amazon Cell Phone Reviews

## Introduction
This project focuses on building a sentiment analysis model to classify Amazon product reviews into positive, negative, and neutral categories. The dataset consists of reviews for cell phones and accessories collected from Amazon for the years 2017-2018. Given the vast amount of consumer feedback available, this model aims to automatically categorize the sentiment expressed in user reviews, which can be beneficial for both potential buyers and product vendors in assessing consumer satisfaction.

## Approach Used
The project employs a Machine Learning pipeline that includes preprocessing of text data, feature extraction using TF-IDF vectorization, and classification using logistic regression. The data handling involves reading large volumes of reviews efficiently, cleaning text data, and preparing it for modeling. We chose logistic regression due to its efficiency and effectiveness in binary and multiclass classification problems.

## Review Star Rating
Reviews on Amazon are often accompanied by a star rating system, where:

- **1 star** is the lowest rating.
- **5 stars** is the highest rating.

These star ratings serve as a proxy for the sentiment expressed in the review.

## Labeling Logic
Based on the star ratings, reviews are categorized into three sentiment labels:

- **Positive**: Reviews with a star rating of **4 or 5**. These ratings indicate that the reviewer was mostly satisfied with the product.
- **Neutral**: Reviews with a star rating of **3**. This rating suggests that the reviewer had a mixed or average opinion about the product.
- **Negative**: Reviews with a star rating of **1 or 2**. These ratings indicate dissatisfaction or significant issues with the product.


### Data Preprocessing
- **Parsing Data:** Data was obtained in a compressed JSON format.Was extracted and formated correctly .
- **Handling Missing Values:** Reviews missing summary text were excluded.
- **Labeling:** Reviews are labeled as positive, neutral, or negative based on their star ratings.



### Feature Extraction
- **TF-IDF Vectorization:** Converts text data into numerical format, capturing the importance of terms relative to their frequency across all documents.

## Model Architecture
The model uses a simple yet effective logistic regression classifier that is trained on features extracted from the review text via TF-IDF vectorization. The choice of logistic regression was driven by its capability to provide good baseline performance with the ability to scale with more complex feature sets and regularization strategies.

## Model Accuracy
The model achieves an overall accuracy of 89%, with the following detailed performance metrics:

- **Precision**:
  - Positive: 91%
  - Neutral: 73%
  - Negative: 76%
- **Recall**:
  - Positive: 98%
  - Neutral: 37%
  - Negative: 69%
- **F1-Score**:
  - Positive: 94%
  - Neutral: 49%
  - Negative: 72%

### Confusion Matrix
The confusion matrix for the model is as follows:
![image](https://github.com/Rahulkrishna-M/Sentiment_analysis_of_reviews/assets/102946334/429ed46d-f3db-4997-aa92-358c548c96a1)

### Running a few test cases
-**For Positive reviews**:
  -![image](https://github.com/Rahulkrishna-M/Sentiment_analysis_of_reviews/assets/102946334/14276a9a-8942-4e22-be9e-1a05fff65859)
  
-**For Negative reviews**:
  -![image](https://github.com/Rahulkrishna-M/Sentiment_analysis_of_reviews/assets/102946334/33348cd2-935e-4fca-bf94-d7fca117d12f)
  
-**For Neutral reviews**:
  -![image](https://github.com/Rahulkrishna-M/Sentiment_analysis_of_reviews/assets/102946334/409fa3f0-ac53-4f78-b4da-f43f611955ba)

### Reasons for Lower Accuracy in Some Categories
- **Data Imbalance**: The dataset might have an uneven distribution of reviews across the categories, particularly fewer neutral reviews, which can make it difficult for the model to learn effective patterns for this class.
- **Complexity of Natural Language**: Sentiments are inherently subjective and can be influenced by context, sarcasm, and subtlety in language, which logistic regression may not capture effectively.

### Strategies for Improvement
- **Model Complexity**: Incorporate more complex models like Support Vector Machines, or deep learning models such as LSTM or BERT, which can better understand the context and nuances of language.
- **Feature Engineering**: Enhance feature extraction by including n-grams, part-of-speech tagging, or sentiment lexicons.
- **Handling Imbalanced Data**: Apply techniques like SMOTE, class weight adjustments, or undersampling to better handle data imbalance.

## Usage
The model is deployed as a Flask application, making it easy to use via a simple API. Users can input a review text and receive a sentiment prediction.

### Predict Example
Due to limitations during deployment, the complete Heroku deployment could not be finalized. However, the model can still be used locally or with future deployment as shown below:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"review":"The product was fantastic!"}' https://<app-name>.herokuapp.com/predict
