import numpy as np
np.set_printoptions(threshold=100000)
import csv
import random
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import time

start_time = time.time()

root_path = '/home/resadmin/haoran/BiLSTM/'
data_path = root_path+'data_20220603_1304/id_label_bi_20220603_digital_1304.csv'
save_path = root_path + 'data_20220603_1304/AUC_Calibration_712_full/'
date = '20220603'
ks = """Best parameters: 
 estors=80, learn-rate=0.05, max_depth=1, child_weight=3, gamma=0.7, subsample=0.3, colsample_bytree=0.1, reg_alph
test:  71.07 val:  69.96 train:  75.2
Best parameters: 
 estors=80, learn-rate=0.05, max_depth=1, child_weight=3, gamma=0.8, subsample=0.3, colsample_bytree=0.1, reg_alpha=0
 test:  71.07 val:  69.96 train:  75.2
Best parameters: 
 estors=80, learn-rate=0.01, max_depth=1, child_weight=3, gamma=0.5, subsample=0.3, colsample_bytree=0.1, reg_alpha=0
test:  69.67 val:  70.05 train:  73.43
Best parameters: 
 estors=80, learn-rate=0.01, max_depth=1, child_weight=3, gamma=0.6, subsample=0.3, colsample_bytree=0.1, reg_alpha=0
 test:  69.67 val:  70.05 train:  73.43
Best parameters: 
 estors=80, learn-rate=0.01, max_depth=1, child_weight=3, gamma=0.7, subsample=0.3, colsample_bytree=0.1, reg_alph
test:  69.67 val:  70.05 train:  73.43
Best parameters: 
 estors=80, learn-rate=0.01, max_depth=1, child_weight=3, gamma=0.8, subsample=0.3, colsample_bytree=0.1, reg_alph
test:  69.67 val:  70.05 train:  73.43
Best parameters: 
 estors=80, learn-rate=0.01, max_depth=1, child_weight=5, gamma=0.5, subsample=0.3, colsample_bytree=0.1, reg_alpha=0
 test:  70.26 val:  70.76 train:  73.18
Best parameters: 
 estors=80, learn-rate=0.01, max_depth=1, child_weight=5, gamma=0.6, subsample=0.3, colsample_bytree=0.1, reg_alph
test:  70.26 val:  70.76 train:  73.18
Best parameters: 
 estors=80, learn-rate=0.01, max_depth=1, child_weight=5, gamma=0.7, subsample=0.3, colsample_bytree=0.1, reg_alph
test:  70.26 val:  70.76 train:  73.18
Best parameters: 
 estors=80, learn-rate=0.01, max_depth=1, child_weight=5, gamma=0.8, subsample=0.3, colsample_bytree=0.1, reg_alpha=0
 test:  70.26 val:  70.76 train:  73.18"""

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
with open(data_path , newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(spamreader):
        if not row:
            continue
        row = [float(item) for item in row]
        y.append(row[0])
        x.append(row[1:])

x = np.array(x)
y = np.array(y)

# split data into 70:10:20
x_train0, x_test, y_train0, y_test = train_test_split(x, y, random_state=99, train_size=(8/10), shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train0, y_train0, random_state=99, train_size=(7/8), shuffle=True)
over_sampling_x_train, over_sampling_y_train = over_sampling(x_train, y_train)


# rebuild model
for l, local_k in enumerate(ks.split('\n')):

    if local_k.find('Best parameters') == -1:
        continue

    k = local_k[local_k.find(':')+2:]
    print(k)

    new_parameter = [int(item.split('=')[1]) for item in k.split(',')]

    model = MLPClassifier(random_state=6, max_iter=new_parameter[0], alpha=new_parameter[1])

    model.fit(over_sampling_x_train, over_sampling_y_train)

    # calibration curve
    prob_train = model.predict_proba(x_train)
    prob_test = model.predict_proba(x_test)

    total_full_training = [item[1] for item in prob_train]  # pred_train
    total_full_testing = [item[1] for item in prob_test]  # pred_test #  prob_test = [value0, value1]
    total_train_y = list(y_train)
    total_test_y = list(y_test)

    # write
    local_model = 'mlp'
    w20 = csv.writer(open(save_path+local_model+'_top'+str(int(10-l/2))+'_'+date+'.csv', "a"))
    w20.writerow(['name', 'list'])

    w20.writerow(['y_test1=']+total_train_y)
    w20.writerow(['y_test2=']+total_test_y)
    w20.writerow(['auc_'+local_model+'_over1=']+total_full_training)
    w20.writerow(['auc_'+local_model+'_over2=']+total_full_testing)