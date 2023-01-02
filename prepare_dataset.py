import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/iris.csv')

# LabelEncoder - кодирование целевых меток, 
# fit_transform - установка кодировщика этикеток и возврат закодированных этикеток.
df['class'] = LabelEncoder().fit_transform(df['class'])

# деление массива данных на обучающую и тестовую выборки
df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

df_train.to_csv('data/train/iris_train.csv')
df_test.to_csv('data/test/iris_test.csv')