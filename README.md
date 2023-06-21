# Sarcopenia-Prediction

# -------------------- models --------------------
gb = gradient boost
lg = logistic regression
mlp = multilayer perceptron
rf = random forest
svm = multilayer perceptron
xgb = xgboost

for example:
LG_712_full_20220603_calibration.py
    # LG = logistic regression model, 712 = data split 70-10-20, full = full features, no feature selection, calibration = generate calibration information for later use
LG_712_full_CI_20220603.py
    # CI = calculate AUC confidence interval
