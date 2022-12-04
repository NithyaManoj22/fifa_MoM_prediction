import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

path = "C:/Users/User/Desktop/FIFA 2018 Statistics.csv"
fifa = pd.read_csv(path)
print(fifa.head(10))

fifa["Man_of_the_Match"].hist()
plt.show()


plt.scatter(fifa.Man_of_the_Match,fifa.Pass_Accuracy_percent)
plt.xlabel("Man of the Match")
plt.ylabel("Pass Accuracy")
plt.show()

plt.scatter(fifa.Man_of_the_Match,fifa.Ball_Possession_p)
plt.xlabel("Man of the Match")
plt.ylabel("Ball_Possession_p")
plt.show()

plt.scatter(fifa.Man_of_the_Match,fifa.Goal_Scored)
plt.xlabel("Man of the Match")
plt.ylabel("Goal Scored")
plt.show()

plt.scatter(fifa.Man_of_the_Match,fifa.Team)
plt.xlabel("Man of the Match")
plt.ylabel("Team")
plt.show()

train1 = fifa.drop(['Man_of_the_Match'],axis=1)
labels = fifa['Man_of_the_Match']

x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.10,random_state =2)

reg = LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
acc1=reg.score(x_test,y_test)
print("Accuracy due to linear regression test is :"+str(acc1*100)+"%")

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
acc2=decisiontree.score(x_test,y_test)
print("Accuracy due to decision tree classifier is :"+str(acc2*100)+"%")

Lreg=LogisticRegression()
Lreg.fit(x_train, y_train)
y_pred=Lreg.predict(x_test)
acc3=Lreg.score(x_test,y_test)
print("Accuracy due to Logistic Regression is :"+str(acc3*100)+"%")
print("The predicted values using Logistic Regression are: "+str(y_pred))
print("Now matching the predicted values with the team vs opponent")
print(y_pred)
array = fifa.values
elmt = array[116:129,0:2]
print(elmt)


