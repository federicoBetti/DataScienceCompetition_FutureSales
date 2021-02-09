import os

import hyperopt
import pandas as pd
from hyperopt import fmin, tpe, STATUS_OK, Trials
from hyperopt import hp
from hyperopt.pyll import scope
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from src.utils.memory_managment import save_object


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

    model.fit(
        train_x,
        train_y,
        eval_metric="rmse",
        eval_set=eval_set,
        verbose=True,
        early_stopping_rounds=early_stopping)

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


# train_test_df.dropna(inplace=True)
# all_train_x = train_test_df[train_test_df.date_block_num < 34].drop(['item_cnt_month'], axis=1)
# all_train_y = train_test_df[train_test_df.date_block_num < 34]['item_cnt_month'].clip(lower=0, upper=20)

def get_data():
    CUSTOM_DATA_FOLDER = '../../data_custom/'

    train_test_df = pd.read_feather(
        os.path.join(os.getcwd(), CUSTOM_DATA_FOLDER, 'all_data_preprocessed.feather')).set_index("index")

    train_x = train_test_df[train_test_df.date_block_num < 33].drop(['item_cnt_month'], axis=1)
    train_y = train_test_df[train_test_df.date_block_num < 33]['item_cnt_month'].clip(lower=0, upper=20)
    valid_x = train_test_df[train_test_df.date_block_num == 33].drop(['item_cnt_month'], axis=1)
    valid_y = train_test_df[train_test_df.date_block_num == 33]['item_cnt_month'].clip(lower=0, upper=20)
    del train_test_df
    # test_x = train_test_df[train_test_df.date_block_num == 34].drop(['item_cnt_month'], axis=1)
    return train_x, train_y, valid_x, valid_y


def get_validation_score(args):
    max_depth = args["max_depth"]
    min_child_weight = args["min_child_weight"]
    eta = args["eta"]
    subsample = args["subsample"]
    colsample_bytree = args["colsample_bytree"]

    train_x, train_y, valid_x, valid_y = get_data()

    model = XGBRegressor(
        max_depth=max_depth,
        n_estimators=100,
        min_child_weight=min_child_weight,
        colsample_bytree=colsample_bytree,
        subsample=subsample,
        eta=eta,
        #     tree_method='gpu_hist',
        seed=42)

    eval_set = [(train_x, train_y), (valid_x, valid_y)]
    early_stopping = 15

    model.fit(
        train_x,
        train_y,
        eval_metric="rmse",
        eval_set=eval_set,
        verbose=False,
        early_stopping_rounds=early_stopping)

    rmse = getRMSE(valid_y, model.predict(valid_x, ntree_limit=model.best_ntree_limit))

    dict_to_ret = {
        "loss": -rmse,
        "status": STATUS_OK,
        "best_tree_number": model.best_ntree_limit
    }
    return dict_to_ret


space = {
    "max_depth": scope.int(hp.quniform("max_depth", 5, 40, 2)),
    "min_child_weight": hp.uniform("min_child_weight", 0.3, 1),
    "eta": hp.choice("eta", [0.1, 0.01, 0.001]),
    "subsample": hp.uniform("subsample", 0.6, 1),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1),
}

trials = Trials()
best = fmin(get_validation_score, space, algo=tpe.suggest, max_evals=10, trials=trials)

print(best)
# -> {'a': 1, 'c2': 0.01420615366247227}
print(hyperopt.space_eval(space, best))

print(trials)

best_path = os.path.join(os.getcwd(), CUSTOM_DATA_FOLDER, 'best_opt.pkl')
trials_path = os.path.join(os.getcwd(), CUSTOM_DATA_FOLDER, 'trials.pkl')
space_path = os.path.join(os.getcwd(), CUSTOM_DATA_FOLDER, 'space.pkl')
save_object(best, best_path)
save_object(trials, trials_path)
save_object(space, space_path)
