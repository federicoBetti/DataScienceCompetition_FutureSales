import os

import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from src.utils.submission import createSubmissionFile

DATA_FOLDER = '../../data/'
CUSTOM_DATA_FOLDER = '../../data_custom/'
SUBMISSION_FOLDER = "../../submissions/"

train_test_df = pd.read_feather(
    os.path.join(os.getcwd(), CUSTOM_DATA_FOLDER, 'all_data_preprocessed.feather')).set_index("index")


def trainXGBoost(train_x, train_y, valid_x=None, valid_y=None, n_estimators=50):
    model = XGBRegressor(
        max_depth=10,
        n_estimators=n_estimators,
        min_child_weight=0.5,
        colsample_bytree=0.8,
        subsample=0.8,
        eta=0.1,
        #     tree_method='gpu_hist',
        seed=42)
    if valid_x is None:
        eval_set = None
        early_stopping = None
    else:
        eval_set = [(train_x, train_y), (valid_x, valid_y)]
        early_stopping = 10
    print("XBGoost model created. Training started..")
    model.fit(
        train_x,
        train_y,
        eval_metric="rmse",
        eval_set=eval_set,
        verbose=True,
        early_stopping_rounds=early_stopping)

    return model


import lightgbm as lgb


def trainLGBM(train_x, train_y, valid_x=None, valid_y=None, n_estimators=100):
    lgb_params = {
        'num_iterations': n_estimators,
        'feature_fraction': 0.75,
        'metric': 'rmse',
        'nthread': 4,
        'min_data_in_leaf': 2 ** 7,
        'bagging_fraction': 0.75,
        'learning_rate': 0.03,
        'objective': 'rmse',
        'bagging_seed': 2 ** 7,
        'num_leaves': 2 ** 7,
        'bagging_freq': 1,
        'verbose': 1
    }
    eval_dict = {}
    if valid_x is None:
        model = lgb.train(lgb_params, lgb.Dataset(train_x, label=train_y))
    else:
        model = lgb.train(lgb_params, lgb.Dataset(train_x, label=train_y),
                          valid_sets=lgb.Dataset(valid_x, label=valid_y), early_stopping_rounds=10,
                          evals_result=eval_dict)

    return model


def trainLR(train_x, train_y):
    lr = LinearRegression()
    lr.fit(train_x.fillna(0).values, train_y.fillna(0))
    return lr


from sklearn import svm


def trainSVM(train_x, train_y):
    regr = svm.LinearSVR()
    regr.fit(train_x.values, train_y)
    return regr


from sklearn.neural_network import MLPRegressor


def trainNN(train_x, train_y):
    regr = MLPRegressor(hidden_layer_sizes=(16, 8), learning_rate="adaptive", verbose=True, max_iter=8)
    regr.fit(train_x.values, train_y)
    return regr


from sklearn.metrics import mean_squared_error


def getRMSE(y_actual, y_predicted):
    rms = mean_squared_error(y_actual.clip(upper=20), y_predicted.clip(max=20), squared=True)
    return rms


test_x = train_test_df[train_test_df.date_block_num == 34].drop(['item_cnt_month'], axis=1)
"""
train_x = train_test_df[train_test_df.date_block_num < 33].drop(['item_cnt_month'], axis=1)
train_y = train_test_df[train_test_df.date_block_num < 33]['item_cnt_month'].clip(lower=0, upper=20)
valid_x = train_test_df[train_test_df.date_block_num == 33].drop(['item_cnt_month'], axis=1)
valid_y = train_test_df[train_test_df.date_block_num == 33]['item_cnt_month'].clip(lower=0, upper=20)

print("Length of dfs: train = {}, valid = {}, test = {}".format(len(train_x), len(valid_x), len(test_x)))

xgboost_model = trainLGBM(train_x, train_y, valid_x, valid_y, 100)

# lr_model = trainLR(train_x.fillna(0), train_y.fillna(0))

# svm_model = trainSVM(train_x.fillna(0), train_y.fillna(0))


# nn_model = trainNN(train_x.fillna(0), train_y.fillna(0))


xgboost_predictions = xgboost_model.predict(valid_x)
# lr_predictions = lr_model.predict(valid_x.fillna(0))
# sv_predictions = svm_model.predict(valid_x.fillna(0))
# nn_predictions = nn_model.predict(valid_x.fillna(0))

xgboost_score = getRMSE(valid_y, xgboost_predictions)
# lr_score = getRMSE(valid_y, lr_predictions)
# svm_score = getRMSE(valid_y, sv_predictions)
# nn_score = getRMSE(valid_y, nn_predictions)
lr_score = 0
svm_score = 0
nn_score = 0
print("XGB score: {}, LRScore: {}, SVMScore: {}, NNScore: {}".format(xgboost_score, lr_score, svm_score, nn_score))

# xgb_complete = trainXGBoost(all_train_x, all_train_y, n_estimators=7)
test_predict_xgb = xgboost_model.predict(test_x)

xgb_result = test_x.copy()
xgb_result["ID"] = range(len(xgb_result))
xgb_result["item_cnt_month"] = test_predict_xgb
createSubmissionFile(xgb_result, "xgb_new_data_100_no_all_data.csv", submission_folder=SUBMISSION_FOLDER)
del xgboost_model
"""
all_train_x = train_test_df[train_test_df.date_block_num < 34].drop(['item_cnt_month'], axis=1)
all_train_y = train_test_df[train_test_df.date_block_num < 34]['item_cnt_month'].clip(lower=0, upper=20)
del train_test_df

xgboost_model = trainLGBM(all_train_x, all_train_y, n_estimators=100)

test_predict_xgb = xgboost_model.predict(test_x)

xgb_result = test_x.copy()
xgb_result["ID"] = range(len(xgb_result))
xgb_result["item_cnt_month"] = test_predict_xgb
createSubmissionFile(xgb_result, "xgb_new_data_100_all_data.csv", submission_folder=SUBMISSION_FOLDER)
