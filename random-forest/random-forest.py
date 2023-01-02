import pandas as pd

from sklearn.ensemble import RandomForestClassifier

# формирование файла с данными для обучения.
train = pd.read_csv('./data/train/iris_train.csv')
X_train = train.drop(columns=['class']).to_numpy()
y_train = train['class'].to_numpy()

# формирование файла с данными для теста.
test = pd.read_csv('./data/test/iris_test.csv')
X_test = test.drop(columns=['class']).to_numpy()

rfc = RandomForestClassifier(max_depth=2, random_state=42).fit(X_train, y_train)
y_pred = rfc.predict(X_test).tolist()

with open('random-forest/predict.txt', 'w') as f:
    for el in y_pred:
        f.write(f"{el} \n")
