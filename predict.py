import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
def diabetes(num):
    d = pd.read_csv('diabetes.csv')
    df = d.copy()
    y=df['Outcome']
    x=df.drop(columns=['Outcome'],axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    classifier = svm.SVC(kernel='linear')
    classifier.fit(x_train,y_train)
    from sklearn.metrics import accuracy_score
    x_train_acc = classifier.predict(x_train)
    t_data_train = accuracy_score(x_train_acc,y_train)
    print("accuracy of train",t_data_train)
    x_test_acc = classifier.predict(x_test)
    t_data_test = accuracy_score(x_test_acc,y_test)
    print(t_data_test)