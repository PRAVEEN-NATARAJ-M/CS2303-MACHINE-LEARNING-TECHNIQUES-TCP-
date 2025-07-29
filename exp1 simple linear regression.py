#importing pre build modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#load the dataset
data=pd.read_csv("E:\CS2303-MACHINE LEARNING TECHNIQUES(TCP)\exp1 car dealership dataset.csv")

data.head()

data.tail()

x=data.iloc[:,:-1].values
print(x)

y=data.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)

ot=model.predict(x_test)
print(ot)

ot1=model.predict([[1000]])
print(ot1)

plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,model.predict(x_train),color="blue")
plt.title("Milleage vs Selling price (training set) ")
plt.xlabel("Milleage(in miles)")
plt.ylabel("Selling price (in dollars)")
plt.show()

plt.scatter(x_test,y_test,color='blue')
plt.plot(x_test,model.predict(x_test),color='red')
plt.title('Mileage vs selling price (teating set)')
plt.xlabel('Mileage (in miles)')
plt.ylabel('Selling price (in dollars)')
plt.show()
