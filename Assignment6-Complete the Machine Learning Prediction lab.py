# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Task 1: Load and Prepare the Data
print("Task 1: Load and Prepare the Data")
data_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
X_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv"

data = pd.read_csv(data_url)
X = pd.read_csv(X_url)
Y = data['Class'].to_numpy()

print("First 5 rows of data:")
print(data.head())
print("\nFirst 5 rows of X:")
print(X.head())
print("\nY shape:", Y.shape)

# Task 2: Standardize the Features
print("\nTask 2: Standardize the Features")
transform = preprocessing.StandardScaler()
X = transform.fit_transform(X)

print("Standardized features (first 5 rows):")
print(X[:5])

# Task 3: Split the Data into Training and Testing Sets
print("\nTask 3: Split the Data into Training and Testing Sets")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(f"Training data shape: X_train={X_train.shape}, Y_train={Y_train.shape}")
print(f"Testing data shape: X_test={X_test.shape}, Y_test={Y_test.shape}")

# Task 4: Create a Logistic Regression model and tune hyperparameters
print("\nTask 4: Hyperparameter Tuning with GridSearchCV")
parameters = {
    'C': [0.01, 0.1, 1],
    'penalty': ['l2'],  # Only 'l2' is supported by 'lbfgs'
    'solver': ['lbfgs']
}

lr = LogisticRegression()
logreg_cv = GridSearchCV(estimator=lr, param_grid=parameters, cv=10, scoring='accuracy')
logreg_cv.fit(X_train, Y_train)

# Output the best parameters and the best accuracy
print("Tuned hyperparameters (best parameters):", logreg_cv.best_params_)
print("Accuracy on validation data:", logreg_cv.best_score_)

# Task 5: Test Accuracy and Confusion Matrix
print("\nTask 5: Test Accuracy and Confusion Matrix")
test_accuracy = logreg_cv.score(X_test, Y_test)
print("Test Accuracy:", test_accuracy)

# Define confusion matrix plotting function
def plot_confusion_matrix(y, y_predict):
    """This function plots the confusion matrix."""
    cm = confusion_matrix(y, y_predict)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['Did Not Land', 'Landed'])
    ax.yaxis.set_ticklabels(['Did Not Land', 'Landed'])
    plt.show()

# Predict on the test data
yhat = logreg_cv.predict(X_test)

# Plot the confusion matrix
plot_confusion_matrix(Y_test, yhat)

# Import necessary libraries for Task 6
from sklearn.svm import SVC
import numpy as np

# Task 6: Support Vector Machine with GridSearchCV
print("\nTask 6: Support Vector Machine with GridSearchCV")

# Define the parameters to tune
parameters = {
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'C': np.logspace(-3, 3, 5),  # Regularization parameter, values in logarithmic space
    'gamma': np.logspace(-3, 3, 5)  # Kernel coefficient, values in logarithmic space
}

# Create the SVM model
svm = SVC()

# Create the GridSearchCV object with 10-fold cross-validation
svm_cv = GridSearchCV(estimator=svm, param_grid=parameters, cv=10, scoring='accuracy')

# Fit the GridSearchCV object with training data
svm_cv.fit(X_train, Y_train)

# Output the best parameters and the best accuracy
print("Tuned hyperparameters (best parameters):", svm_cv.best_params_)
print("Accuracy on validation data:", svm_cv.best_score_)

# Task 7: Calculate Accuracy on Test Data and Plot the Confusion Matrix

print("\nTask 7: Accuracy on Test Data and Confusion Matrix")

# Calculate the accuracy on the test data using the score method
test_accuracy = svm_cv.score(X_test, Y_test)
print(f"Accuracy on Test Data: {test_accuracy * 100:.2f}%")

# Use the trained model to predict the test data
yhat = svm_cv.predict(X_test)

# Plot the confusion matrix for the test data
plot_confusion_matrix(Y_test, yhat)

# Task 8: Decision Tree Classifier with GridSearchCV

# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Define parameters for GridSearchCV
parameters = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [2 * n for n in range(1, 10)],
    'max_features': ['sqrt', 'log2', None],  # Updated to valid options
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10]
}

# Create DecisionTreeClassifier object
tree = DecisionTreeClassifier()

# Create GridSearchCV object with cross-validation
tree_cv = GridSearchCV(estimator=tree, param_grid=parameters, cv=10)

# Fit the GridSearchCV object to the data
tree_cv.fit(X_train, Y_train)

# Output the best parameters and best score
print("Tuned Hyperparameters (Best Parameters):", tree_cv.best_params_)
print("Accuracy:", tree_cv.best_score_)


# Task 9: Calculate accuracy and plot confusion matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("\nTask 9: Calculate accuracy and plot confusion matrix")

# Calculate the accuracy of the Decision Tree classifier on the test data
test_accuracy = tree_cv.score(X_test, Y_test)
print(f"Accuracy on the test data: {test_accuracy * 100:.2f}%")

# Predict the test data labels using the best model from GridSearchCV
yhat_tree = tree_cv.predict(X_test)

# Plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt='g')  # annot=True to annotate cells with numeric values
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['Did Not Land', 'Landed'])
    ax.yaxis.set_ticklabels(['Did Not Land', 'Landed'])
    plt.show()

# Plot the confusion matrix for the Decision Tree model on the test data
plot_confusion_matrix(Y_test, yhat_tree)

# Task 10: K-Nearest Neighbors (KNN) Model with GridSearchCV

print("\nTask 10: K-Nearest Neighbors (KNN) Model with GridSearchCV")
# Import necessary libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Define the parameters for GridSearchCV
parameters = {
    'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1, 2]  # 'p' determines the distance metric: 1 for Manhattan, 2 for Euclidean
}

# Create the KNeighborsClassifier object
knn = KNeighborsClassifier()

# Create the GridSearchCV object with 10-fold cross-validation
knn_cv = GridSearchCV(estimator=knn, param_grid=parameters, cv=10)

# Fit the model to the training data
knn_cv.fit(X_train, Y_train)

# Output the best hyperparameters and the best score
print("Tuned Hyperparameters (Best Parameters) for KNN:", knn_cv.best_params_)
print("Accuracy for KNN:", knn_cv.best_score_)

# Calculate and print accuracy on the test data
test_accuracy = knn_cv.score(X_test, Y_test)
print(f"Accuracy on the test data: {test_accuracy}")

# Optionally, plot the confusion matrix for KNN
yhat_knn = knn_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat_knn)

# Task 11: Evaluate KNN Model on Test Data and Plot Confusion Matrix
print("\nTask 11: Evaluate KNN Model on Test Data and Plot Confusion Matrix")
# Calculate the accuracy of KNN on the test data
test_accuracy_knn = knn_cv.score(X_test, Y_test)
print(f"Test Accuracy for KNN: {test_accuracy_knn}")

# Plot the confusion matrix for KNN
yhat_knn = knn_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat_knn)

print("\nTASK 12: Compare Performance of Models")
# Assuming you have already fitted the models with GridSearchCV
# You should have the following variables for each model's test accuracy:

# Logistic Regression Test Accuracy
logreg_accuracy = logreg_cv.score(X_test, Y_test)

# Support Vector Machine Test Accuracy
svm_accuracy = svm_cv.score(X_test, Y_test)

# Decision Tree Test Accuracy
tree_accuracy = tree_cv.score(X_test, Y_test)

# K-Nearest Neighbors Test Accuracy
knn_accuracy = knn_cv.score(X_test, Y_test)

# Create a dictionary to store the model names and their respective test accuracies
model_performance = {
    'Logistic Regression': logreg_accuracy,
    'Support Vector Machine': svm_accuracy,
    'Decision Tree': tree_accuracy,
    'K-Nearest Neighbors': knn_accuracy
}

# Print the accuracy for each model
for model, accuracy in model_performance.items():
    print(f"{model} Test Accuracy: {accuracy:.4f}")

# Find the model with the best accuracy
best_model = max(model_performance, key=model_performance.get)
best_accuracy = model_performance[best_model]

print(f"\nThe best performing model is {best_model} with an accuracy of {best_accuracy:.4f}")



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
num_test_samples = X_test.shape[0]
print(f"Number of records in the test sample: {num_test_samples}")

# Print the best kernel that gave the best result on the validation dataset
print("Best kernel:", svm_cv.best_params_['kernel'])

# Get the accuracy of the best model on the test data
accuracy = tree_cv.score(X_test, Y_test)
print(f"Accuracy on the test data: {accuracy}")
