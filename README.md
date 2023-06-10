# Machine-Learning-Algorithms
The goal of this project is to implement and compare three different classification algorithms: 
Perceptron, Support Vector Machine (SVM), and Passive-Aggressive (PA). 
The task is to classify examples into one of three classes using a provided training set of 3,286 examples.

# Data set
The dataset used in this project is related to Abalone age classification. 
Each instance in the dataset has eight features, including both numerical and categorical attributes. 
The target variable consists of three labels representing the age of Abalones. 

Here is a summarized description of the features in the Abalone age classification dataset:
1. Sex (nominal): Represents the sex of the Abalone. It has three possible values: M (male), F (female), and I (infant).
2. Length (continuous): The longest measurement of the Abalone's shell in millimeters.
3. Diameter (continuous): The measurement of the Abalone's shell diameter in millimeters, perpendicular to the length.
4. Height (continuous): The height of the Abalone's shell in millimeters, with the meat inside.
5. Whole weight (continuous): The weight of the whole Abalone in grams.
6. Shucked weight (continuous): The weight of the meat only, in grams.
7. Viscera weight (continuous): The weight of the gut after bleeding, in grams.
8. Shell weight (continuous): The weight of the shell after being dried, in grams.

These features provide measurements and attributes related to the physical characteristics of the Abalone. 
They can be used to predict the age of the Abalone based on the provided labels.

The categorical attribute, "Sex," needs to be converted into a numerical representation. 
Additionally, different normalization techniques and feature selection methods should be explored.

# usage
The main file should be executed with three input arguments: 
The first argument is the file path to the training examples (train_x.txt),
the second argument is the file path to the training labels (train_y.txt), 
and the third argument is the file path to the testing examples (test_x.txt).

The implementation should train the Perceptron, SVM, and PA algorithms in the given order. 
The hyperparameters for these models are hardcoded, meaning no external arguments will be provided during runtime.

# output
The final output should be the predictions for the testing examples (test_x.txt). 
The predictions should be displayed on the screen in the following format:

perceptron: 0, svm: 0, pa: 1
perceptron: 2, svm: 2, pa: 2
perceptron: 1, svm: 1, pa: 1
...

Each line in the output corresponds to a line in the test_x.txt file, 
and the numbers represent the class predictions for the classes 0, 1, and 2.

The project explore different methods for converting the categorical attribute,
apply various normalization techniques, and experiment with feature selection. 
By comparing the performance of Perceptron, SVM, and PA on the Abalone age classification task, 
you can gain insights into the strengths and weaknesses of each algorithm.
