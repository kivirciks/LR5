import pandas as pd

from sklearn.cluster import KMeans

# формирование файла с данными для обучения.
train = pd.read_csv('./data/train/iris_train.csv')
X_train = train.drop(columns=['class']).to_numpy()
y_train = train['class'].to_numpy()

# формирование файла с данными для теста.
test = pd.read_csv('./data/test/iris_test.csv')
X_test = test.drop(columns=['class']).to_numpy()

# cluster = 3, так как 3 класса. Метод fit пытается найти такие коэффициенты,
# чтобы минимизировать различие между предсказанием модели по данным x_train и 
# реальным значением y_train.
kmeans = KMeans(n_clusters=3,random_state=42).fit(X_train)
# predict(X_test) предсказание на данных X_test во время теста.
y_pred = kmeans.predict(X_test).tolist()

# параметр mode = 'w' значит "обрежьте файл до нулевой длины или создайте текстовый файл 
# для записи2. Поток располагается в начале файла.
with open('k-means/predict.txt', 'w') as file:
    for el in y_pred:
        file.write(f"{el} \n")