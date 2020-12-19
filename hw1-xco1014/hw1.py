import sys
import pandas as pd
import numpy as np
import math
import csv


data = pd.read_csv('./train.csv', encoding = 'big5')

data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
    month_data[month] = sample

x = np.empty([12 * 471, 18 * 9], dtype = float)
y = np.empty([12 * 471, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9] #value


mean_x = np.mean(x, axis = 0) #18 * 9 
std_x = np.std(x, axis = 0) #18 * 9 

for i in range(len(x)): #12 * 471
    for j in range(len(x[i])): #18 * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
    



x_train_set = x[: math.floor(len(x) * 0.99), :]
y_train_set = y[: math.floor(len(y) * 0.99), :]
x_validation = x[math.floor(len(x) * 0.99): , :]
y_validation = y[math.floor(len(y) * 0.99): , :]






dim = (18 * 9) + 1
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([len(x_train_set), 1]), x_train_set), axis = 1).astype(float)
y = y_train_set
learning_rate = 10
iter_time = 10000
adagrad = np.zeros([dim, 1])
eps = 0.0000000001

for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/len(x_train_set))#rmse
    if(t%100==0):
        print(str(t) + ":" + str(loss))
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) #dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
np.save('weight.npy', w)
np.save("std_x.npy",std_x)
np.save("mean_x.npy",mean_x)



w = np.load('weight.npy')

testdata = pd.read_csv('./test.csv', header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18*9], dtype = float)

for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)


for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]

test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)


ans_y = np.dot(test_x, w)


with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)

x_validation = np.concatenate((np.ones([len(x_validation), 1]), x_validation), axis = 1).astype(float)
loss_valid = np.sqrt(np.sum(np.power(np.dot(x_validation, w) - y_validation , 2))/len(x_validation))
print("valid loss",loss_valid)