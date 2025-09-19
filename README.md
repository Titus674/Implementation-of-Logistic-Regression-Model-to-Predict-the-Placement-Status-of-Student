# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3.Import LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset.
5.Predict the values of array.
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7.Apply new unknown values.
## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Titus Ratna Kumar Karivella 
RegisterNumber: 212224230292
```
```py
import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()

data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:,:-1]
x

y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```
## Output:

Placement data

<img width="1420" height="262" alt="231648649-33f7653d-1911-4ba3-91da-3e9356a138f2" src="https://github.com/user-attachments/assets/c375e69d-4dd8-46ae-8961-5baed147e863" />

Salary data
<img width="1242" height="242" alt="2" src="https://github.com/user-attachments/assets/112a423a-4265-4e71-b82b-99562f522be7" />

Checking null function


<img width="350" height="315" alt="3" src="https://github.com/user-attachments/assets/2711f920-686f-480c-ac01-f836df348944" />

Data duplicate

<img width="185" height="40" alt="4" src="https://github.com/user-attachments/assets/09ce9e48-2977-4224-9663-5693836ed9b2" />

Data status

<img width="1210" height="522" alt="5" src="https://github.com/user-attachments/assets/15b2a31a-f219-4b5c-91e2-b0edfa5f83d3" />

Y-prediction array

<img width="552" height="287" alt="6" src="https://github.com/user-attachments/assets/28b307b6-ad64-44b4-8e85-66f0224a8c39" />

Accuracy value

<img width="798" height="68" alt="7" src="https://github.com/user-attachments/assets/e9488540-17e1-410e-814b-7ed5dfd3b51b" />

Confusion array

<img width="342" height="67" alt="8" src="https://github.com/user-attachments/assets/38f0b1f0-427c-4035-8ade-e8bbc69bd95c" />

Classification report

<img width="1778" height="71" alt="9" src="https://github.com/user-attachments/assets/2c435949-523d-4c3a-88ea-d1c125835d46" />

Prediction of LR

<img width="1603" height="88" alt="10" src="https://github.com/user-attachments/assets/f4802af5-f906-4e3b-87b6-3cd0d9d02aad" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
