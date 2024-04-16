
# k-Nearest Neighbors for Car Evaluation

Welcome to the k-Nearest Neighbors for Car Evaluation repository. This project implements the k-Nearest Neighbors (k-NN) algorithm to classify cars based on their attributes using a dataset sourced from the UCI Machine Learning Repository.

## Dataset

The dataset used for this project is available [here](https://archive.ics.uci.edu/dataset/19/car+evaluation). It contains data on various characteristics of cars, such as buying price, maintenance cost, number of doors, capacity, size of luggage boot, and safety. The target variable is the car's acceptability (unacc, acc, good, vgood).


## Code Overview

The Python script `knn_car_evaluation.py` encapsulates the following functionalities:

- **Data Loading**: Loads the car evaluation dataset from a CSV file.
- **Data Preprocessing**: Utilizes `LabelEncoder` from scikit-learn to encode categorical features.
- **Data Splitting**: Segregates the dataset into training and test sets using `train_test_split` from scikit-learn.
- **Model Creation**: Constructs a k-Nearest Neighbors classifier with `n_neighbors=5` using `KNeighborsClassifier` from scikit-learn.
- **Model Training**: Trains the classifier on the training data.
- **Model Evaluation**: Evaluates the classifier's performance on the test data and computes the accuracy metric.


## Requirements

This project requires the following dependencies:

- **Python 3.x**
- **pandas**
- **numpy**
- **scikit-learn**


## License

This project is licensed under the MIT License. Feel free to use and modify the code according to your requirements.

---



