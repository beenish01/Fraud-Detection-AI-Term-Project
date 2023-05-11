from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
import pandas as pd


data = pd.read_csv('C:/Users/beeni/Downloads/heart.csv')
X = data.drop('is_fraudulent', axis=1)



y = data['is_fraudulent']



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data Information")
print(data.isnull().sum())

# print(data.info())
print()
data_dup = data.duplicated().any()
print("Are there any duplicates:", data_dup)

print("Are there any duplicates,now:", data_dup)
print()
categorical = []
continuous = []

num_cols = len(data.columns)
i = 0
while i < num_cols:
    if data.iloc[:, i].nunique() <= 10:
        categorical.append(data.columns[i])
    else:
        continuous.append(data.columns[i])
    i += 1

# categorical:variable that takes on a limited and usually fixed number of possible value
print("categorical variables")
print(categorical)
print()
# continuous:variable that can take on any value within a certain range
print("continuous variables")

print(continuous)
print()
# Encode Categorical Data
print(categorical)
data['card_type'].unique()

categorical.remove('gender')
categorical.remove('is_fraudulent')

data = pd.get_dummies(data, columns=categorical, drop_first=True)
print(data.head())

print()
# standard scaling
# scaling the continuous variables in the dataframe 'data' and assigning them back to the same col
# to ensure that they are on the same scale and have similar variances.
print('standard scaling')
st = StandardScaler()
data[continuous] = st.fit_transform(data[continuous])
print(data.head())
# splitting data:split the feature matrix X and the target vector y into training and testing sets

# x= input data.Select all rows and all columns except for isfraudulent
X = data.loc[:, data.columns != 'is_fraudulent']
# output data isfraud col
y = data['is_fraudulent']




#20% of the data will be reserved for testing and the remaining 80% will be used for training.
# ensuring that the same split is obtained each time, by setting random state=42

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(y_test)
print(data.head())
print()

print()
print('Logistic Regression')
log = LogisticRegression()
log.fit(X_train, y_train)
LogisticRegression()
y_pred1 = log.predict(X_test)

# proportion of correct predictions made by the model.
accuracy_lr = accuracy_score(y_test, y_pred1)
# ow many true positives were correctly predicted out of all actual positive samples.
recall_lr = recall_score(y_test, y_pred1)
# measures the proportion of true positives among all positive predictions
precision_lr = precision_score(y_test, y_pred1)
# a measure of how many true positives were correctly predicted out of all actual positive samples
f1_lr = f1_score(y_test, y_pred1)
#  average recall obtained on each class
balanced_accuracy_lr = balanced_accuracy_score(y_test, y_pred1)
print('recall', recall_lr)
print('acurracy', accuracy_lr)
print("Precision:", precision_lr)
print("F1-score:", f1_lr)
print("Balanced accuracy:", balanced_accuracy_lr)




print()
print('svc')
svm = svm.SVC()
svm.fit(X_train, y_train)
y_pred2 = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred2)
recall_svm = recall_score(y_test, y_pred2)
print("Recall score: {:.3f}".format(recall_svm))
precision_svm = precision_score(y_test, y_pred2)
f1_svm = f1_score(y_test, y_pred2)
balanced_accuracy_svm = balanced_accuracy_score(y_test, y_pred2)
print("accuracy", accuracy_svm)
print("Precision:", precision_svm)
print("F1-score:", f1_svm)
print("Balanced accuracy:", balanced_accuracy_svm)



print()
print('K-nearest-NeighborsClassifier')
from sklearn.neighbors import KNeighborsClassifier


knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred3 = knn.predict(X_test)
accuracy_nn = accuracy_score(y_test, y_pred3)
recall_nn = recall_score(y_test, y_pred3)
print("Recall score: {:.3f}".format(recall_nn))
precision_nn = precision_score(y_test, y_pred3)
f1_nn = f1_score(y_test, y_pred3)
balanced_accuracy_nn = balanced_accuracy_score(y_test, y_pred3)
print("accuracy", accuracy_nn)
print("Precision:", precision_nn)
print("F1-score:", f1_nn)
print("Balanced accuracy:", balanced_accuracy_nn)
print(' knn on different values of k to predict accuracy')

score = []
recall = []
precision = []
f1_scores = []
balanced_acc = []

k = 1
while k <= 10:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    score.append(accuracy_score(y_test, y_pred))
    recall.append(recall_score(y_test, y_pred))
    precision.append(precision_score(y_test, y_pred))

    f1_scores.append(f1_score(y_test, y_pred))
    balanced_acc.append(balanced_accuracy_score(y_test, y_pred))
    k += 1

print("Accuracy scores:", score)
print("Recall scores:", recall)
print("Precision scores:", precision)
print("F1 scores:", f1_scores)
print("Balanced accuracy scores:", balanced_acc)







knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)

scores = [accuracy, recall, precision, f1, balanced_acc]
labels = ['accuracy', 'Recall', 'Precision', 'F1 Score', 'Balanced Accuracy']

# PLOT THE GRAPH AT 10 NEIGHBORS
# plt.bar(labels, scores)
# plt.title('KNN Model Evaluation')
#
# for i, v in enumerate(scores):
#     plt.text(i, v, str(round(v, 2)), color='green', ha='center')
#
# plt.show()
#

data = pd.read_csv('C:/Users/beeni/Downloads/heart.csv')
data = data.drop_duplicates()

X = data.drop('is_fraudulent', axis=1)

y = data['is_fraudulent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





print('Decision Tree')

print()
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

y_pred4 = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred4)
recall_dt = recall_score(y_test, y_pred4)
print("Recall score: {:.3f}".format(recall))
precision_dt = precision_score(y_test, y_pred4)
f1_dt = f1_score(y_test, y_pred4)
balanced_accuracy_dt = balanced_accuracy_score(y_test, y_pred4)

print('acurracy', accuracy_dt)
print("Precision:", precision_dt)
print("F1-score:", f1_dt)
print("Balanced accuracy:", balanced_accuracy_dt)

print()
print('random forest')
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred5 = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred5)
recall_rf = recall_score(y_test, y_pred5)
print("Recall score: {:.3f}".format(recall_rf))
precision_rf = precision_score(y_test, y_pred5)
f1_rf = f1_score(y_test, y_pred5)
balanced_accuracy_rf = balanced_accuracy_score(y_test, y_pred5)
print('acurracy', accuracy_rf)
print("Precision:", precision_rf)
print("F1-score:", f1_rf)
print("Balanced accuracy:", balanced_accuracy_rf)

from sklearn.ensemble import GradientBoostingClassifier

print()
print('GradientBoostingClassifier')
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
y_pred6 = gbc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred6)
recall = recall_score(y_test, y_pred6)
print("Recall score: {:.3f}".format(recall))
precision = precision_score(y_test, y_pred6)
f1 = f1_score(y_test, y_pred6)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred6)

print('acurracy', accuracy)
print("Precision:", precision)
print("F1-score:", f1)
print("Balanced accuracy:", balanced_accuracy)
# GRAPH TO COMPARE ACCURACIES
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Define the metrics and models to plot
# metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
# models = ['Logistic Regression', 'Random Forest', 'SVM', 'Decision Tree']
#
# # Define the values for each metric and model
# accuracy_values = [accuracy_lr, accuracy_rf, accuracy_svm, accuracy_dt]
# precision_values = [precision_lr, precision_rf, precision_svm, precision_dt]
# recall_values = [recall_lr, recall_rf, recall_svm, recall_dt]
# f1_values = [f1_lr, f1_rf, f1_svm, f1_dt]
#
# # Set the width of the bars and the spacing between them
# bar_width = 0.2
# bar_spacing = 0.1
#
# # Create a list of x positions for each metric
# x_positions = np.arange(len(metrics))
#
# # Plot the bars for each model and metric
# plt.bar(x_positions - 1.5 * bar_width - bar_spacing, accuracy_values, width=bar_width, label='Accuracy')
# plt.bar(x_positions - 0.5 * bar_width, precision_values, width=bar_width, label='Precision')
# plt.bar(x_positions + 0.5 * bar_width + bar_spacing, recall_values, width=bar_width, label='Recall')
# plt.bar(x_positions + 1.5 * bar_width + 2 * bar_spacing, f1_values, width=bar_width, label='F1 Score')
#
# # Add labels, title, and legend
# plt.xticks(x_positions, metrics)
# plt.xlabel('Metrics')
# plt.ylabel('Score')
# plt.title('Comparison of Machine Learning Models')
# plt.legend(models)
#
# # Display the plot
# plt.show()


new_data = pd.DataFrame({
    'age': 52,
    'gender': 1,
    'card_type': 0,
    'balance_before_transaction': 125,
    'newbalance': 212,
    'transaction_type': 0,
    'device_type': 1,
    'Transaction_authorization_code': 168,
    'transaction_city': 0,
    'merchant_location': 1.0,
    'Channel_used_for_transaction': 2,
    'Country_transaction': 2,
    'transaction_amount': 3,
}, index=[0])




print(new_data)




p = rf.predict(new_data)
print(p)


if p[0] == 0:
    print("No fraud")
if p[0] == 1:
    print("fraud")



import joblib




joblib.dump(rf, 'model_joblib_fraud')




['model_joblib_fraud']




model = joblib.load('model_joblib_fraud')




model.predict(new_data)




data.tail()





from tkinter import *
import joblib
import numpy as np
from sklearn import *


def show_entry_fields():
    p1 = int(e1.get())
    p2 = int(e2.get())
    p3 = int(e3.get())
    p4 = int(e4.get())
    p5 = int(e5.get())
    p6 = int(e6.get())
    p7 = int(e7.get())
    p8 = int(e8.get())
    p9 = int(e9.get())
    p10 = float(e10.get())
    p11 = int(e11.get())
    p12 = int(e12.get())
    p13 = int(e13.get())
    model = joblib.load('model_joblib_fraud')
    result = model.predict([[p1, p2, p3, p4, p5, p6, p7, p8, p8, p10, p11, p12, p13]])

    if result == 0:
        Label(master, text="No fraud").grid(row=31)
    else:
        Label(master, text="Possibility of fraud").grid(row=31)


master = Tk()
master.title("fraud detection")

label = Label(master, text="fraud detection Prediction System"
              , bg="black", fg="white"). grid(row=0, columnspan=2)

Label(master, text="Enter Your Age").grid(row=1)
Label(master, text="Male Or Female [1/0]").grid(row=2)
Label(master, text="Enter card_type ").grid(row=3)
Label(master, text="Enter balance_before_transaction").grid(row=4)
Label(master, text="Enter newbalance").grid(row=5)
Label(master, text="Enter transaction_type (Withdrawal or Purchase)").grid(row=6)
Label(master, text="Enter device_type (Point-Of-Sale (POS) terminal/ smartphone").grid(row=7)
Label(master, text="Enter Transaction_authorization_code").grid(row=8)
Label(master, text="Enter transaction_city (Inside [0] or Outside[1] your city)").grid(row=9)
Label(master, text="Enter merchant_location").grid(row=10)
Label(master, text="Enter Channel_used_for_transaction").grid(row=11)
Label(master, text="Enter Country_transaction").grid(row=12)
Label(master, text="Enter transaction_amount").grid(row=13)

e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e8 = Entry(master)
e9 = Entry(master)
e10 = Entry(master)
e11 = Entry(master)
e12 = Entry(master)
e13 = Entry(master)

e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)
e8.grid(row=8, column=1)
e9.grid(row=9, column=1)
e10.grid(row=10, column=1)
e11.grid(row=11, column=1)
e12.grid(row=12, column=1)
e13.grid(row=13, column=1)

Button(master, text='Predict', command=show_entry_fields).grid()
mainloop()


from lime import lime_tabular

explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=['0', '1'],
    mode='classification'
)
exp = explainer.explain_instance(
    data_row=X_test.iloc[1],
    predict_fn=rf.predict_proba
)
# Explain instance for each model
models = [log, svm, knn, dt, gbc]
for model in models:
    print(type(model).__name__)
    exp = explainer.explain_instance(
        data_row=X_test[X_train.columns].iloc[1],
        predict_fn=rf.predict_proba

    )




fig = exp.as_pyplot_figure()
plt.show()

mainloop()

