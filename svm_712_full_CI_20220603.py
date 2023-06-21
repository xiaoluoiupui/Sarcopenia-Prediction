import numpy as np
np.set_printoptions(threshold=100000)
import csv
import random
from sklearn import metrics
# from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import time
from sklearn.svm import SVC

start_time = time.time()

root_path = '/home/resadmin/haoran/BiLSTM/'
data_path = root_path+'data_20220603_1304/id_label_bi_20220603_digital_1304.csv'


def over_sampling(x_train, y_train):
    over_sampling_train = []
    train_0_id = []  # 0-records positions
    train_1_id = []  # 1-records positions
    for l, label in enumerate(y_train):
        if label == 0:
            train_0_id.append(l)
            over_sampling_train.append(np.append([label], x_train[l], axis=0))
        else:
            train_1_id.append(l)
            over_sampling_train.append(np.append([label], x_train[l], axis=0))
            over_sampling_train.append(np.append([label], x_train[l], axis=0))
            over_sampling_train.append(np.append([label], x_train[l], axis=0))
            over_sampling_train.append(np.append([label], x_train[l], axis=0))
    random.seed(555)
    random.shuffle(over_sampling_train)

    over_sampling_x_train = [item[1:] for item in over_sampling_train]
    over_sampling_y_train = [item[0] for item in over_sampling_train]
    return over_sampling_x_train, over_sampling_y_train


# check label-tf_feature record
x = []
y = []
with open(data_path, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(spamreader):
        if not row:
            continue
        row = [float(item) for item in row]
        y.append(row[0])
        x.append(row[1:])

x = np.array(x)
y = np.array(y)
overall_dict = {}
total_total_print = []

# split data into 70:10:20
x_train0, x_test, y_train0, y_test = train_test_split(x, y, random_state=99, train_size=(8/10), shuffle=True)  # 88

x_train, x_val, y_train, y_val = train_test_split(x_train0, y_train0, random_state=99, train_size=(7/8), shuffle=True)

# reduce some features; split above first, them reduce
x_train = x_train  # np.delete(x, [868, 1082], 1)
x_val = x_val  # np.delete(x, [868, 1082], 1)
x_test = x_test  # np.delete(x, [868, 1082], 1)
# print(np.shape(x_train), np.shape(x_val), np.shape(x_test))  # 1031=(1304, 378); 1105=(912, 369) (131, 369) (261, 369)

y1=[item for item in y_train if item==1]
y2=[item for item in y_val if item==1]
y3=[item for item in y_test if item==1]

# print((np.shape(x_train)[0]/np.shape(x_val)[0]), (np.shape(x_test)[0]/np.shape(x_val)[0]))  # (7,2)
# print(len(y1), len(y2), len(y3), (len(y1)/len(y2)), (len(y3)/len(y2)))  # random_state=88=(177 25 47)=(7.08, 1.88)

over_sampling_x_train, over_sampling_y_train = over_sampling(x_train, y_train)

# model train and validation
for left_ratio in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:   # 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8
    for lr in [0.001, 0.005, 0.01, 0.05]:  # 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1
        # fill parameter
        right_ratio = 1 - left_ratio
        model = SVC(kernel='linear', probability=True, C=lr, class_weight={0: left_ratio, 1: right_ratio},
                    random_state=6)

        # train model on the training set
        model.fit(over_sampling_x_train, over_sampling_y_train)
        # validate the trained model via validation set
        prob_val = model.predict_proba(x_val)
        prob_1_val = [item[1] for item in prob_val]  # prob_test = [value0, value1]
        # convert prediction to roc score
        fpr, tpr, threshold = metrics.roc_curve(y_val, prob_1_val)
        roc_auc = metrics.auc(fpr, tpr)

        # get training set score
        prob_train = model.predict_proba(x_train)
        prob_1_train = [item[1] for item in prob_train]  # prob_test = [value0, value1]
        # convert prediction to roc score
        fpr0, tpr0, threshold0 = metrics.roc_curve(y_train, prob_1_train)
        roc_auc0 = metrics.auc(fpr0, tpr0)

        a = np.round(roc_auc * 100, 2)
        b = np.round(roc_auc0 * 100, 2)

        print('left_ratio=', left_ratio, 'learning rate=', lr, '&', a, '&', b)

        k = 'left_ratio=' + str(left_ratio) + ', learning rate=' + str(lr)
        v = a, b
        if 0 < v[1] - v[0] <= 10:
            overall_dict[k] = v

print('\n', '-' * 50, '\n')

parameter_list = []
for top_parameter, roc_values in sorted(overall_dict.items(), key=lambda item: item[1][0], reverse=False):
    # print(top_parameter, '&', roc_values[0], '&', roc_values[1])
    parameter_list.append([top_parameter, roc_values[0], roc_values[1]])

parameter_list = parameter_list[-10:]

roc_test_list = []
for best_parameter, roc_val, roc_train in parameter_list:

    new_parameter = [float(item.split('=')[1]) for item in best_parameter.split(',')]
    print('Best parameters:', best_parameter)

    # testing
    # re-build the model
    left_ratio = new_parameter[0]
    lr = new_parameter[1]
    right_ratio = 1 - left_ratio
    model = SVC(kernel='linear', probability=True, C=lr, class_weight={0: left_ratio, 1: right_ratio},
                random_state=6)
    # model = RandomForestClassifier(random_state=6, n_jobs=-1, n_estimators=new_parameter[0],
                                   # max_depth=new_parameter[1])
    model.fit(over_sampling_x_train, over_sampling_y_train)
    # get testing set score
    prob_test = model.predict_proba(x_test)
    prob_1_test = [item[1] for item in prob_test]  # prob_test = [value0, value1]
    # convert prediction to roc score
    fpr1, tpr1, threshold1 = metrics.roc_curve(y_test, prob_1_test)
    roc_test = metrics.auc(fpr1, tpr1)
    roc_test = np.round(roc_test * 100, 2)
    roc_test_list.append(roc_test)
    print('test: ', roc_test, 'val: ', roc_val, 'train: ', roc_train)

print('\n', '-' * 50, '\n')
roc_val_list = [item[1] for item in parameter_list]
roc_train_list = [item[2] for item in parameter_list]
method = 'svm'
print('test_list_'+method+'=', roc_test_list, '#', np.mean(roc_test_list))
print('val_list_'+method+'=', roc_val_list, '#', np.mean(roc_val_list))
print('train_list_'+method+'=', roc_train_list, '#', np.mean(roc_train_list))

"""Best parameters: left_ratio=0.6, learning rate=0.01
test:  70.9 val:  72.86 train:  78.61
Best parameters: left_ratio=0.65, learning rate=0.001
test:  71.76 val:  74.0 train:  76.54
Best parameters: left_ratio=0.65, learning rate=0.005
test:  71.52 val:  74.18 train:  77.02
Best parameters: left_ratio=0.65, learning rate=0.01
test:  71.28 val:  74.43 train:  78.88
Best parameters: left_ratio=0.6, learning rate=0.05
test:  69.78 val:  75.11 train:  84.28
Best parameters: left_ratio=0.7, learning rate=0.005
test:  72.2 val:  75.61 train:  78.59
Best parameters: left_ratio=0.7, learning rate=0.01
test:  72.35 val:  75.61 train:  79.5
Best parameters: left_ratio=0.7, learning rate=0.001
test:  72.17 val:  75.64 train:  78.61
Best parameters: left_ratio=0.7, learning rate=0.05
test:  70.03 val:  77.64 train:  83.48
Best parameters: left_ratio=0.65, learning rate=0.05
test:  69.74 val:  77.67 train:  83.7

 -------------------------------------------------- 

test_list_svm= [70.9, 71.76, 71.52, 71.28, 69.78, 72.2, 72.35, 72.17, 70.03, 69.74] 71.173
val_list_svm= [72.86, 74.0, 74.18, 74.43, 75.11, 75.61, 75.61, 75.64, 77.64, 77.67] 75.275
train_list_svm= [78.61, 76.54, 77.02, 78.88, 84.28, 78.59, 79.5, 78.61, 83.48, 83.7] 79.921

Process finished with exit code 0

"""