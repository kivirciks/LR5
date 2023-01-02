import pandas as pd

from sklearn.tree import DecisionTreeClassifier

# формирование файла с данными для обучения.
train = pd.read_csv('./data/train/iris_train.csv')
X_train = train.drop(columns=['class']).to_numpy()
y_train = train['class'].to_numpy()

# формирование файла с данными для теста.
test = pd.read_csv('./data/test/iris_test.csv')
X_test = test.drop(columns=['class']).to_numpy()

tree = DecisionTreeClassifier().fit(X_train, y_train)
y_pred = tree.predict(X_test).tolist()

with open('dtree/predict.txt', 'w') as file:
    for el in y_pred:
        file.write(f"{el} \n")