# Marketing
My marketing research
import numpy as np
import pandas as pd
file = 'Mark2.xlsx'
xl = pd.ExcelFile(file)
df = xl.parse('Sheet')
df=df.loc[:,'Gender':'Diff']
num_cols = [
    'Num_ed',
    'Num_ew',
    'Middle',
    'Newtech',
    'New_inn',
    'New_all',
    'Buy_smart',
    'Causes_tech'
]
cat_cols = [
    'Causes_fact',
    'Age'
]
target_col ='Diff'
hist=df.hist(column=num_cols+cat_cols+[target_col], figsize=(16, 10))
data = df.copy(deep=True)
del data['Causes_fact']
del data['Age']
data['Diff'] = data['Diff'] * 100
file = 'Mark4.xlsx'
xll = pd.ExcelFile(file)
df4 = xll.parse('Sheet')
df4=df4.loc[:,'Gender':'Diff']
del df4['Causes_fact']
del df4['Age']
del df4['Newtech']
del df4['Buy_smart']
labels = df4[df4.columns[7]]
feature_matrix = df4[df4.columns[:7]]
from sklearn.model_selection import train_test_split
train_feature_matrix, test_feature_matrix, train_labels, test_labels = train_test_split(
    feature_matrix, labels, test_size=0.1, random_state=42)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1)
# обучение модели
clf.fit(train_feature_matrix, train_labels)
# предсказание на тестовой выборке
y_pred = clf.predict(test_feature_matrix)

from sklearn.metrics import accuracy_score

print(accuracy_score(test_labels, y_pred))
print(y_pred)
