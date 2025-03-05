# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
####  1. Import the standard Libraries.
####  2.Set variables for assigning dataset values.
####  3.Assign the points for representing in the graph.
####  4.Predict the regression for marks by using the representation of the graph.
#### 5.Compare the graphs and hence we obtained the linear regression for the given data.


## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
## Developed by:Kannan N 
## RegisterNumber:212223230097
``` 

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/SMARTLINK/Downloads/student_scores.csv")
df.head()

df.tail()

X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

#spilitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
Y_pred

Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
![image](https://github.com/user-attachments/assets/607e47ba-8574-4d7d-af38-2b5eac69951e)

![image](https://github.com/user-attachments/assets/93db7dc0-8280-4c25-98ce-ccd3465f7923)

![image](https://github.com/user-attachments/assets/61c2a464-6002-4840-84e5-497f3f51772b)

![image](https://github.com/user-attachments/assets/9c1b46f4-6755-4eea-ab64-131cbcc1630c)

![image](https://github.com/user-attachments/assets/03f8854b-c19b-4a72-a133-70a0cfd479d0)

![image](https://github.com/user-attachments/assets/eb5eeb29-2d70-4bed-9bf5-ae0ae3d0ffbc)

![image](https://github.com/user-attachments/assets/79c30490-9ea6-45e2-9fef-f3975ac53c25)

![image](https://github.com/user-attachments/assets/0f80dbb8-ce97-4eca-8c6c-ddb000070f1c)

![image](https://github.com/user-attachments/assets/b92629ef-26f3-44be-912d-84a75c2aa823)

![image](https://github.com/user-attachments/assets/937d0a00-eaa0-4740-aee3-5125a6d05bac)


![image](https://github.com/user-attachments/assets/1297bf2c-2bc0-4436-9d85-660bd7ac1ba3)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
