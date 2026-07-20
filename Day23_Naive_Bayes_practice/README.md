# Day 23 - Gaussian Naive Bayes

## Objective
Learn and implement the Gaussian Naive Bayes classification algorithm using the Breast Cancer and Wine datasets from Scikit-learn.

## Datasets Used
- Breast Cancer Dataset
- Wine Dataset

## Tasks Performed
- Loaded and explored both datasets.
- Split the data into training and testing sets.
- Applied StandardScaler for comparison.
- Trained a Gaussian Naive Bayes classifier.
- Evaluated the model using:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report
- Compared Gaussian Naive Bayes with:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)

## Mini Challenge
Implemented Gaussian Naive Bayes and KNN on the Wine dataset and compared their performance.

### Accuracy Comparison

| Model | Accuracy |
|--------|---------:|
| Gaussian Naive Bayes | **1.00 (100%)** |
| KNN | **0.94 (94%)** |

### Observation
Gaussian Naive Bayes outperformed KNN on the Wine dataset. It achieved perfect classification accuracy, while KNN achieved 94% accuracy. This suggests that the Wine dataset fits the probabilistic assumptions of Gaussian Naive Bayes very well.

## Files
- `breast_cancer_naive_bayes.py`
- `wine_naive_bayes.py`

## Libraries Used
- pandas
- scikit-learn

## Concepts Learned
- Gaussian Naive Bayes
- Probabilistic Classification
- Feature Scaling
- Model Evaluation
- Accuracy Comparison

## Result
Successfully implemented Gaussian Naive Bayes on both the Breast Cancer and Wine datasets. The algorithm achieved strong performance, including **100% accuracy on the Wine dataset**, and outperformed KNN in the mini challenge.