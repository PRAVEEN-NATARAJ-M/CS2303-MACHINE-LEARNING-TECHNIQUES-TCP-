import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree
import os

print(os.path.exists(r"E:\CS2303-MACHINE LEARNING TECHNIQUES(TCP)\exp4/exp4 student result dataset.csv"))#windows supports both /forward and \backward slash
folder=r"E:\CS2303-MACHINE LEARNING TECHNIQUES(TCP)\exp4"
print("FILES IN FOLDE",os.listdir(folder))

data=pd.read_csv(r"E:\CS2303-MACHINE LEARNING TECHNIQUES(TCP)\exp4/exp4 student result dataset.csv")

df=pd.DataFrame(data)
'''
df.head(5)
df.tail(5)
These lines return DataFrames, but do not display them because you're not printing them or running them in an
interactive notebook (like Jupyter).

In Python scripts (like in IDLE or .py files), you must explicitly print the results to see them.

✅ Solution: Add print() to display them
'''
print(df.head())
print(df.tail())


x=df[['study_hours','attendance',]]
y=df['result']

clf=DecisionTreeClassifier(criterion='entropy',random_state=42)

clf.fit(x,y)
plt.figure(figsize=(8,6))
plot_tree(clf,feature_names=['studyhours','atendance'],class_names=['0','1'],filled=True)
plt.show()

'''
This warning comes from the line:

new = [[5,85]]
pred = clf.predict(new)

When you trained your model:

clf.fit(x, y)

you used a pandas DataFrame (x) that includes column names ('study_hours' and 'attendance').
So scikit-learn tracks those feature names.

But when predicting:

new = [[5,85]]

you passed a plain Python list, which has no feature names,
so scikit-learn warns that the feature names are missing and may not match the training data structure.

✅ Solution: Use a DataFrame for prediction

Instead of passing a list, pass a pandas DataFrame with the same column names as used during training:

new = pd.DataFrame([[5, 85]], columns=['study_hours', 'attendance'])
pred = clf.predict(new)

✅ This will remove the warning, and ensure the input format is consistent with training.
'''

new = pd.DataFrame([[5, 85]], columns=['study_hours', 'attendance'])
pred=clf.predict(new)
print("PREDICTION OF NEW STUDENT:","1" if pred[0]==1 else "0")
