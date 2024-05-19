# CIFAR-10 Classification

This project explores the classification of images from the CIFAR-10 dataset using various machine learning algorithms.

## Dependencies

- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Scikit-learn](https://scikit-learn.org/)

## Data

The CIFAR-10 dataset is a collection of 60,000 32x32 color images in 10 classes:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## Machine Learning Algorithms

### SVM (Support Vector Machine)

- A non-probabilistic, linear classifier that finds a hyperplane that maximizes the margin between the classes.
- Applied to the flattened image data (32x32x3 pixels converted to a 1D vector of 3072 features).

### Logistic Regression

- A probabilistic classifier that models the relationship between features and a binary class label using a logistic function.
- Similarly applied to the flattened image data.

### Naive Bayes

- A probabilistic classifier based on Bayes' theorem that assumes independence between features.
- Used with the flattened image data as well.

### KNN (K-Nearest Neighbors)

- A non-parametric, lazy learning algorithm that classifies an instance based on the majority vote of its k nearest neighbors in the training data.
- Employed with the flattened image data.

### CNN (Convolutional Neural Network)

- A deep learning architecture specifically designed for image classification.
- Utilizes convolutional layers to extract spatial features from the images.
- Employs the full image data format (32x32x3).

## Code Structure

### Data Loading and Preprocessing

- Imports the CIFAR-10 dataset using `tensorflow.keras.datasets.cifar10`.
- Normalizes the pixel values by dividing by 255.
- One-hot encodes the class labels using `to_categorical`.
- Visualizes sample images with their corresponding class labels using Matplotlib.

### Machine Learning Models

- Implements SVM, Logistic Regression, Naive Bayes, and KNN models from Scikit-learn.
- Trains each model on the flattened image data (`X_train.reshape(-1, 32*32*3)`).
- Evaluates each model's accuracy on the test data using `accuracy_score`.

### CNN Model

- Defines a CNN architecture using TensorFlow's Keras API.
- Employs convolutional layers with ReLU activation and Batch Normalization for efficient training.
- Incorporates MaxPooling and Dropout layers for dimensionality reduction and regularization.
- Uses a Flatten layer to convert the feature maps into a 1D vector before feeding it into dense layers.
- Compiles the model with categorical cross-entropy loss, Adam optimizer, and accuracy metric.
- Trains the CNN model using early stopping to prevent overfitting.
- Visualizes the model architecture using `plot_model`.
- Evaluates the trained CNN model on the test data using `model.evaluate`.
- Plots the training and validation loss and accuracy curves using Matplotlib.

## Comparison of Model Accuracies

- Plots a bar chart to compare the accuracies of all the models using Matplotlib.

## Results

- The experiment evaluates the performance of various machine learning algorithms on the CIFAR-10 dataset.
- The CNN model is expected to achieve the highest accuracy due to its ability to learn spatial features from images.
- The comparison chart provides insights into the relative strengths and weaknesses of each model.
