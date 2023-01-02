import pandas as pd

from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt

y_test = pd.read_csv('data/test/iris_test.csv')['class'].to_numpy().tolist()

kmeans_pred = list()
with open('k-means/predict.txt', 'r') as file:
    for line in file:
        kmeans_pred.append(int(line))

dtree_pred = list()
with open('dtree/predict.txt', 'r') as file:
    for line in file:
        dtree_pred.append(int(line))

rfc_pred = list()
with open('random-forest/predict.txt', 'r') as f:
    for line in f:
        rfc_pred.append(int(line))

dtree_report = accuracy_score(y_test, dtree_pred)
kmeans_report = accuracy_score(y_test, kmeans_pred)
rfc_report = accuracy_score(y_test, rfc_pred)

with open('metrics.txt', 'w') as f:
    f.write(f"dtree acc: {dtree_report}")
    f.write("\n")
    f.write(f"kmeans acc: {kmeans_report}")
    f.write("\n")
    f.write(f"rfc acc: {rfc_report}")

f, ax = plt.subplots()
ax.bar(['dtree', 'k-means', 'random-forest'], [dtree_report, kmeans_report, rfc_report])
f.savefig('plot.png')