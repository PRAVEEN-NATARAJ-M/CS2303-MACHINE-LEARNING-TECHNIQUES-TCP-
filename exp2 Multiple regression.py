import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import seaborn as sns
import matplotlib.pyplot as plt
import os
print(os.path.exists(r"E:\CS2303-MACHINE LEARNING TECHNIQUES(TCP)\exp2\exp2 house dealership dataset.csv"))

folder = r"E:\CS2303-MACHINE LEARNING TECHNIQUES(TCP)\exp2"
print("Files in folder:")
print(os.listdir(folder))



# Sample dataframe
df = pd.read_csv(r"E:/CS2303-MACHINE LEARNING TECHNIQUES(TCP)/exp2/exp2 house dealership dataset.csv")

# Features and target
x = df[['Bedrooms', 'Size', 'Age', 'ZipCode']]
y = df['SellingPrice']

# Handle categorical data (Zip Code)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['ZipCode'])], remainder='passthrough')
xen = ct.fit_transform(x)

# Split data
xtr, xte, ytr, yte = train_test_split(xen, y, test_size=0.2, random_state=42)
# Train the model
model = LinearRegression()
model.fit(xtr, ytr)
# Predicting the new value
ypr = model.predict(xte) 
ypr

# Evaluate
coefficients = model.coef_
intercept = model.intercept_
print("Coefficients:", coefficients)
print("Intercept:", intercept)
#visualize
plt.figure(figsize=(8,6))
sns.scatterplot(x=yte, y=ypr, color='blue', s=100)
plt.plot([min(yte), max(yte)], [min(yte), max(yte)], 'r--')
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.tight_layout( )
plt.show( )
sns.heatmap(x.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
