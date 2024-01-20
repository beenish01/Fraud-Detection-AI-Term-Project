Project Idea: 
We have created a fraud detection system that uses Machine Learning Algorithms to  detect credit or debit card fraud. The system prompts the user with a set of questions and utilizes a pre-existing dataset for training and testing purposes to arrive at a final decision.

Machine Learning Algorithms Used:
Different ML algorithms are used along with their accuracies, precision, recall, F1-score and Balanced Accuracy to make a prediction. The following are the algorithms used:
•	Logistic Regression
•	SVM
•	K-nearest-NeighborsClassifier
•	Decision Tree
•	Random Forest
•	GradientBoostingClassifier


Explanation of the code:
Here’s a brief explanation of the functions performed by the code :
1.	A csv file named 'heart.csv' is first read.
2.	'is_fraudulent' column from the data  is removed and  assigned to variable X so that it is not used to train the model and bias its predictions.
3.	'is_fraudulent' column is assigned to variable y  since it is the target variable of the dataset.This will allow us to train a machine learning model to predict whether a transaction is fraudulent or not, based on the values of the other columns in the dataset.
4.	The data is  split into training and testing sets using train_test_split from scikit-learn
5.	Data preprocessing is performed on the data, including encoding categorical variables and standard scaling continuous variables
6.	Fitting and evaluation of  the performance of three different classification models (Logistic Regression, SVM, and KNN) on the data, reporting accuracy, recall, precision, F1-score, and balanced accuracy for each model.
7.	A GUI application  is used that allows users to enter details of a financial transaction, and based on these details, the system will predict whether the transaction is fraudulent or not. The application uses the Tkinter library, which is a standard Python library for creating graphical user interfaces. It also uses joblib to load a pre-trained machine learning model, which is used to make the prediction.



8.	LIME (Local Interpretable Model-Agnostic Explanations) is then used  to explain the prediction made by a machine learning model. LIME is a method for explaining the predictions of any black box machine learning model.

Note : All graphs (except LIME) have been commented. 


Data Source
We have extracted a CSV file from Kaggle.com containing all the requisite information. Data from this will be inserted into the program.





