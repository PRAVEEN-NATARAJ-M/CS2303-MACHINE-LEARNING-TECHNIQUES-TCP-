import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay,classification_report
import matplotlib.pyplot as plt
import os
print(os.path.exists(r"E:\CS2303-MACHINE LEARNING TECHNIQUES(TCP)\exp3\exp3 human health dataset.csv"))
folder = r"E:\CS2303-MACHINE LEARNING TECHNIQUES(TCP)\exp2"
print("Files in folder:")
print(os.listdir(folder))

df=pd.read_csv("exp3 human health dataset.csv")
df['gender']=LabelEncoder().fit_transform(df['gender'])
x=df[['age','gender','bmi','bp','cholesterol']]
y=df['condition']
scaler=StandardScaler()
xscale=scaler.fit_transform(x)
xtr,xte,ytr,yte=train_test_split(xscale,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(xtr,ytr)
ypr=model.predict(xte)
yprob=model.predict_proba(xte)[:,1]
print("ACCURAY SCORE",accuracy_score(yte,ypr))
print("CLASSIFICATIO_REPORT:\n",classification_report(yte,ypr,zero_division=1))
cm=confusion_matrix(yte,ypr)
disp=ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
disp.plot(cmap='Blues')
plt.title("CONFUSION MATRIX")
plt.show()

new=pd.DataFrame([[60,1,27,130,200]],columns=['age','gender','bmi','bp','cholesterol'])
newscale=scaler.transform(new)
newcondition = model.predict_proba(newscale)[0][1]
print(f"PROBABILTY OF DEVELOPING THE CONDITION{newcondition:.2f}")
