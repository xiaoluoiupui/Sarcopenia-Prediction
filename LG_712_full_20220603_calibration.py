import numpy as np
np.set_printoptions(threshold=100000)
import csv
import random
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import time

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
I = list(range(len(y)))

# split data into 70:10:20
I_train, I_test, x_train0, x_test, y_train0, y_test = train_test_split(I, x, y, random_state=99, train_size=(8/10), shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train0, y_train0, random_state=99, train_size=(7/8), shuffle=True)
over_sampling_x_train, over_sampling_y_train = over_sampling(x_train, y_train)

print(len(y_train), sum(y_train))
print(len(y_val), sum(y_val))
print(len(y_test), sum(y_test))
print(I_test)
stop
# rebuild model
ks = """Best parameters: allow_iter=10, left_ratio=0.5, learning rate=0.005
test:  71.6 val:  69.16 train:  75.13
Best parameters: allow_iter=10, left_ratio=0.55, learning rate=0.005
test:  71.51 val:  69.16 train:  75.03
Best parameters: allow_iter=10, left_ratio=0.6, learning rate=0.005
test:  71.58 val:  69.2 train:  74.86
Best parameters: allow_iter=10, left_ratio=0.65, learning rate=0.01
test:  71.6 val:  69.84 train:  76.87
Best parameters: allow_iter=10, left_ratio=0.7, learning rate=0.01
test:  71.65 val:  69.84 train:  76.43
Best parameters: allow_iter=10, left_ratio=0.6, learning rate=0.01
test:  71.64 val:  70.01 train:  77.23
Best parameters: allow_iter=10, left_ratio=0.45, learning rate=0.01
test:  71.54 val:  70.16 train:  77.7
Best parameters: allow_iter=10, left_ratio=0.4, learning rate=0.01
test:  71.56 val:  70.19 train:  77.73
Best parameters: allow_iter=10, left_ratio=0.5, learning rate=0.01
test:  71.59 val:  70.19 train:  77.63
Best parameters: allow_iter=10, left_ratio=0.55, learning rate=0.01
test:  71.59 val:  70.19 train:  77.47"""
for l, local_k in enumerate(ks.split('\n')):

    if local_k.find('Best parameters') == -1:
        continue

    k = local_k[local_k.find(':')+2:]
    print(k)

    new_parameter0 = [item.split('=')[1] for item in k.split(',')]
    new_parameter = [int(new_parameter0[0]), float(new_parameter0[1]), float(new_parameter0[2])]

    allow_iter = new_parameter[0]
    left_ratio = new_parameter[1]
    lr = new_parameter[2]
    right_ratio = 1 - left_ratio

    model = LogisticRegression(random_state=6, n_jobs=-1,
                               class_weight={0: left_ratio, 1: right_ratio}, max_iter=allow_iter, C=lr)

    model.fit(over_sampling_x_train, over_sampling_y_train)


    # calibration curve
    prob_train = model.predict_proba(x_train)
    prob_test = model.predict_proba(x_test)

    total_full_training = [item[1] for item in prob_train]  # pred_train
    total_full_testing = [item[1] for item in prob_test]  # pred_test #  prob_test = [value0, value1]
    total_train_y = list(y_train)
    total_test_y = list(y_test)

    # write
    local_model = 'lg'
    w20 = csv.writer(open(root_path + 'data_20220603_1304/AUC_Calibration_712_full/'+local_model+'_top'+str(int(10-l/2))+'_20220603.csv', "a"))
    w20.writerow(['name', 'list'])

    w20.writerow(['y_test1=']+total_train_y)
    w20.writerow(['y_test2=']+total_test_y)
    w20.writerow(['auc_'+local_model+'_over1=']+total_full_training)
    w20.writerow(['auc_'+local_model+'_over2=']+total_full_testing)