import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import xgboost as xgb
import time
import lightgbm as lgb
start_time = time.time()

train_xy = pd.read_pickle("../data/train/train.pkl")

val = pd.read_pickle("../data/train/validation.pkl")
test = pd.read_pickle("../data/train/test.pkl")
#train_xy.to_csv("test.csv",index=False)
#exit()

test['Date_received_int'] = test.Date_received.dt.year*10000 + test.Date_received.dt.month *100 + test.Date_received.dt.day

df = test[['User_id','Coupon_id','Date_received_int']]

train_xy.drop(['ID','User_id','Merchant_id','Coupon_id','day_diff','Date_received','Date_received_int','user_label_coupon_min_date','user_label_coupon_max_date'],axis=1, inplace=True)
val.drop(['ID','User_id','Merchant_id','Coupon_id','day_diff','Date_received','Date_received_int','user_label_coupon_min_date','user_label_coupon_max_date'],axis=1, inplace=True)
test.drop(['Date','ID','User_id','Coupon_id','Date_received','Date_received','Date_received_int','user_label_coupon_min_date','user_label_coupon_max_date','Merchant_id'],axis=1, inplace=True)



############
train_xy.drop(['off_mu_total_coupon','ol_user_get_coupon_num_via_mean','ol_user_buy_via_mean','off_m_diff_median','off_m_diff_mean'],axis=1, inplace=True)
val.drop(['off_mu_total_coupon','ol_user_get_coupon_num_via_mean','ol_user_buy_via_mean','off_m_diff_median','off_m_diff_mean'],axis=1, inplace=True)
test.drop(['off_mu_total_coupon','ol_user_get_coupon_num_via_mean','ol_user_buy_via_mean','off_m_diff_median','off_m_diff_mean'],axis=1, inplace=True)
############

# print(train_xy.info())
# print(test.info())
# exit()

if (train_xy.shape[1]-1!= test.shape[1]):
    print("Train")
    print("-"*20)
    print(train_xy.info())
    print("Test")
    print("-" * 20)
    print(test.info())
    exit()

train_xy = pd.concat([train_xy,val],axis=0)

y = train_xy.coupon_in_15_day
X = train_xy.drop(['coupon_in_15_day'],axis=1)
val_y = val.coupon_in_15_day
val_X = val.drop(['coupon_in_15_day'],axis=1)

print(X.info())
#exit()
if 1:
    #xgb矩阵赋值
    xgb_train = xgb.DMatrix(X, label=y)

    xgb_val = xgb.DMatrix(val_X,label=val_y)
    xgb_test = xgb.DMatrix(test)
    #watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
    watchlist = [(xgb_train, 'train')]
    num_rounds = 5000  # 迭代次数
    def calSearch(xgb_train, watchlist,val1, label1, val2='', label2=''):
        print(label1,':',val1,' ',label2,':',val2)
        params = {
            'booster': 'gbtree',
            'objective': 'reg:logistic',  # 多分类的问题
            'gamma': 0.3,  # init 0.1 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
            'max_depth': 7,  # 6 init 构建树的深度，越大越容易过拟合
            'lambda': 10,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
            'subsample': 0.76,  # init 0.76 随机采样训练样本
            'colsample_bytree': 0.95,  # 0.95 生成树时进行的列采样
            'min_child_weight': 1, #init 3
            # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
            # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
            # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
            'silent': 1,  # 设置成1则没有运行信息输出，最好是设置为0.
            'eta': 0.09,  # 0.1如同学习率
            'seed': 1000,
            # 'nthread':7,# cpu 线程数
            'eval_metric': 'auc'
        }
        plst = list(params.items())



        # 训练模型并保存
        # early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
        model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds=100)
        print(label1, ':', val1, ' ', label2, ':', val2)
        print("best best_ntree_limit", model.best_ntree_limit)

    # calSearch(xgb_train, watchlist, 3, 'max_depth', 1, 'min_child_weight')  #train-auc:0.878184	val-auc:0.825246
    # calSearch(xgb_train, watchlist, 3, 'max_depth', 3, 'min_child_weight')  #train-auc:0.885195	val-auc:0.829318
    # calSearch(xgb_train, watchlist, 3, 'max_depth', 5, 'min_child_weight')  #train-auc:0.878015	val-auc:0.826032
    # calSearch(xgb_train, watchlist, 5, 'max_depth', 1, 'min_child_weight')  #train-auc:0.892164	val-auc:0.831793
    # calSearch(xgb_train, watchlist, 5, 'max_depth', 3, 'min_child_weight')  #train-auc:0.891914	val-auc:0.833314 ###
    # calSearch(xgb_train, watchlist, 5, 'max_depth', 5, 'min_child_weight')  #train-auc:0.893316	val-auc:0.831922
    # calSearch(xgb_train, watchlist, 7, 'max_depth', 1, 'min_child_weight')  #train-auc:0.89603	val-auc:0.835448 ###
    # calSearch(xgb_train, watchlist, 7, 'max_depth', 3, 'min_child_weight')  #train-auc:0.898705	val-auc:0.831567
    # calSearch(xgb_train, watchlist, 7, 'max_depth', 5, 'min_child_weight')  #train-auc:0.897978	val-auc:0.832926
    # calSearch(xgb_train, watchlist, 7, 'max_depth', 6, 'min_child_weight')  #train-auc:0.90112	val-auc:0.834043
    # calSearch(xgb_train, watchlist, 9, 'max_depth', 1, 'min_child_weight')  #train-auc:0.905741	val-auc:0.826717
    # calSearch(xgb_train, watchlist, 9, 'max_depth', 3, 'min_child_weight')  #train-auc:0.912337	val-auc:0.830032
    # calSearch(xgb_train, watchlist, 9, 'max_depth', 5, 'min_child_weight')  #train-auc:0.916119	val-auc:0.831023

    #calSearch(xgb_train, watchlist, 6, 'max_depth', 1, 'min_child_weight') #train-auc:0.898984	val-auc:0.834105
    #calSearch(xgb_train, watchlist, 6, 'max_depth', 3, 'min_child_weight') #train-auc:0.901465	val-auc:0.833344
    #calSearch(xgb_train, watchlist, 6, 'max_depth', 5, 'min_child_weight') #train-auc:0.903919	val-auc:0.831085
    #calSearch(xgb_train, watchlist, 6, 'max_depth', 6, 'min_child_weight') #train-auc:0.889621	val-auc:0.834457

    ### best is max_depth : 7   min_child_weight : 1   #train-auc:0.89603	val-auc:0.835448

    #calSearch(xgb_train, watchlist, 0.0, 'gamma')  #train-auc:0.908531	val-auc:0.836503  ##
    #calSearch(xgb_train, watchlist, 0.1, 'gamma')  #train-auc:0.89603	val-auc:0.835448
    #calSearch(xgb_train, watchlist, 0.2, 'gamma')  #train-auc:0.903157	val-auc:0.834757
    #calSearch(xgb_train, watchlist, 0.3, 'gamma')  #train-auc:0.907241	val-auc:0.836517  ###
    #calSearch(xgb_train, watchlist, 0.4, 'gamma')  #train-auc:0.901165	val-auc:0.833931
    ### best is gamma : 0.3: train-auc:0.907241	val-auc:0.836517

    # calSearch(xgb_train, watchlist, 0.6, 'subsample', 0.6, 'colsample_bytree')  #train-auc:0.900168	val-auc:0.828672
    # calSearch(xgb_train, watchlist, 0.6, 'subsample', 0.7, 'colsample_bytree')  #train-auc:0.902451	val-auc:0.826574
    # calSearch(xgb_train, watchlist, 0.6, 'subsample', 0.8, 'colsample_bytree')  #train-auc:0.902787	val-auc:0.830603
    # calSearch(xgb_train, watchlist, 0.6, 'subsample', 0.9, 'colsample_bytree')  #train-auc:0.902488	val-auc:0.82819
    # calSearch(xgb_train, watchlist, 0.7, 'subsample', 0.6, 'colsample_bytree')  #train-auc:0.902802	val-auc:0.826171
    # calSearch(xgb_train, watchlist, 0.7, 'subsample', 0.7, 'colsample_bytree')  #train-auc:0.892464	val-auc:0.824861
    # calSearch(xgb_train, watchlist, 0.7, 'subsample', 0.8, 'colsample_bytree')  #train-auc:0.896486	val-auc:0.828042
    # calSearch(xgb_train, watchlist, 0.7, 'subsample', 0.9, 'colsample_bytree')  #train-auc:0.900522	val-auc:0.827351
    # calSearch(xgb_train, watchlist, 0.8, 'subsample', 0.6, 'colsample_bytree')  #train-auc:0.896031	val-auc:0.827899
    # calSearch(xgb_train, watchlist, 0.8, 'subsample', 0.7, 'colsample_bytree')  #train-auc:0.905134	val-auc:0.82694
    # calSearch(xgb_train, watchlist, 0.8, 'subsample', 0.8, 'colsample_bytree')  #train-auc:0.902582	val-auc:0.831854 #
    # calSearch(xgb_train, watchlist, 0.8, 'subsample', 0.9, 'colsample_bytree')  #train-auc:0.895718	val-auc:0.831957 ##
    # calSearch(xgb_train, watchlist, 0.9, 'subsample', 0.6, 'colsample_bytree')  #train-auc:0.901681	val-auc:0.827874
    # calSearch(xgb_train, watchlist, 0.9, 'subsample', 0.7, 'colsample_bytree')  #train-auc:0.902839	val-auc:0.825303
    # calSearch(xgb_train, watchlist, 0.9, 'subsample', 0.8, 'colsample_bytree')  #train-auc:0.89319	val-auc:0.824661
    # calSearch(xgb_train, watchlist, 0.9, 'subsample', 0.9, 'colsample_bytree')  #train-auc:0.907749	val-auc:0.833412 ###

    # calSearch(xgb_train, watchlist, 0.85, 'subsample', 0.8, 'colsample_bytree') #train-auc:0.895154	val-auc:0.831403
    # calSearch(xgb_train, watchlist, 0.85, 'subsample', 0.85, 'colsample_bytree') #train-auc:0.902685	val-auc:0.831448
    # calSearch(xgb_train, watchlist, 0.85, 'subsample', 0.9, 'colsample_bytree') #train-auc:0.894427	val-auc:0.826799
    # calSearch(xgb_train, watchlist, 0.85, 'subsample', 0.95, 'colsample_bytree') #train-auc:0.895367	val-auc:0.828048
    # #calSearch(xgb_train, watchlist, 0.9, 'subsample', 0.8, 'colsample_bytree') #train-auc:0.89319	val-auc:0.824661
    # calSearch(xgb_train, watchlist, 0.9, 'subsample', 0.85, 'colsample_bytree') #train-auc:0.897904	val-auc:0.829227
    # #calSearch(xgb_train, watchlist, 0.9, 'subsample', 0.9, 'colsample_bytree') #train-auc:0.907749	val-auc:0.833412 ###
    # calSearch(xgb_train, watchlist, 0.9, 'subsample', 0.95, 'colsample_bytree') #train-auc:0.895701	val-auc:0.829992
    # calSearch(xgb_train, watchlist, 0.95, 'subsample', 0.8, 'colsample_bytree') #train-auc:0.89674	val-auc:0.831971
    # calSearch(xgb_train, watchlist, 0.95, 'subsample', 0.85, 'colsample_bytree') #train-auc:0.901636	val-auc:0.826608
    # calSearch(xgb_train, watchlist, 0.95, 'subsample', 0.9, 'colsample_bytree') #train-auc:0.896554	val-auc:0.830373
    # calSearch(xgb_train, watchlist, 0.95, 'subsample', 0.95, 'colsample_bytree') #train-auc:0.90218	val-auc:0.829304

    #calSearch(xgb_train, watchlist, 0.76, 'subsample', 0.9, 'colsample_bytree') #train-auc:0.903585	val-auc:0.831434
    #calSearch(xgb_train, watchlist, 0.78, 'subsample', 0.9, 'colsample_bytree') #train-auc:0.903324	val-auc:0.83175
    #calSearch(xgb_train, watchlist, 0.76, 'subsample', 0.95, 'colsample_bytree')  #
    #calSearch(xgb_train, watchlist, 0.78, 'subsample', 0.95, 'colsample_bytree')  #train-auc:0.903573	val-auc:0.835369
    #calSearch(xgb_train, watchlist, 0.8, 'subsample', 0.95, 'colsample_bytree')  #train-auc:0.894786	val-auc:0.833167
    #calSearch(xgb_train, watchlist, 0.75, 'subsample', 0.95, 'colsample_bytree') #train-auc:0.903522	val-auc:0.829385
    #calSearch(xgb_train, watchlist, 0.77, 'subsample', 0.95, 'colsample_bytree') #train-auc:0.895669	val-auc:0.830632

    ### best is subsample : 0.76   colsample_bytree : 0.95   train-auc:0.907241	val-auc:0.836517
    #calSearch(xgb_train, watchlist, 0.01, 'eta') #train-auc:0.899131	val-auc:0.832178
    #calSearch(xgb_train, watchlist, 0.02, 'eta') #train-auc:0.901333	val-auc:0.830064
    #calSearch(xgb_train, watchlist, 0.08, 'eta') #train-auc:0.901738	val-auc:0.833808
    #calSearch(xgb_train, watchlist, 0.09, 'eta') #train-auc:0.901744	val-auc:0.836587 ###
    #calSearch(xgb_train, watchlist, 0.11, 'eta') #train-auc:0.901996	val-auc:0.833593
    #calSearch(xgb_train, watchlist, 0.05, 'eta')  #train-auc:0.896571	val-auc:0.830521
    #calSearch(xgb_train, watchlist, 0.1, 'eta')  # train-auc:0.907241	val-auc:0.836517 ##
    #calSearch(xgb_train, watchlist, 0.2, 'eta')   #train-auc:0.898929	val-auc:0.829466



    # params = {
    #         'booster': 'gbtree',
    #         'objective': 'reg:logistic',  # 多分类的问题
    #         'gamma': 0.1,  # init 0.1 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。 0.3
    #         'max_depth': 6,  # 6 init 构建树的深度，越大越容易过拟合 7
    #         'lambda': 10,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    #         'subsample': 0.76,  # init 0.76 随机采样训练样本
    #         'colsample_bytree': 0.95,  # 0.95 生成树时进行的列采样
    #         'min_child_weight': 3, #init 3    1
    #         # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    #         # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    #         # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    #         'silent': 1,  # 设置成1则没有运行信息输出，最好是设置为0.
    #         'eta': 0.1,  # 0.1如同学习率
    #         'seed': 1000,
    #         # 'nthread':7,# cpu 线程数
    #         'eval_metric': 'auc'
    #     }
    params = {'booster': 'gbtree',
              'objective': 'rank:pairwise',
              'eval_metric': 'auc',
              'gamma': 0.1,
              'min_child_weight': 1.1,
              'max_depth': 5,
              'lambda': 10,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'colsample_bylevel': 0.7,
              'eta': 0.01,
              'tree_method': 'exact',
              'seed': 0,
              'nthread': 12
              }
    plst = list(params.items())

    # 训练模型并保存
    # early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
    model = xgb.train(plst, xgb_train, num_rounds, watchlist)
    print("best best_ntree_limit", model.best_ntree_limit)

    xgb.plot_importance(model)
    model.save_model('xgb.model')  # 用于存储训练出的模型
    preds = model.predict(xgb_test,ntree_limit=model.best_ntree_limit)
    df['Probability'] = preds
    df.to_csv("result-xgb-20170905-2.csv",index=False,header=False)
    #输出运行时长
    cost_time = time.time()-start_time
    print ("xgboost success!",'\n',"cost time:",cost_time,"(s)......")
    plt.show()

    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
    fs = []

    for (key, value) in feature_score:
        print(key)
        fs.append("{0},{1}\n".format(key, value))

    with open('xgb_feature_score.csv', 'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)

else:
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss', 'auc'},
        'num_leaves': 256,
        'min_sum_hessian_in_leaf': 20,
        'max_depth': 12,
        'learning_rate': 0.05,
        'feature_fraction': 0.6,
        # 'bagging_fraction': 0.9,
        # 'bagging_freq': 3,
        'verbose': 1
    }
    lgb_train = lgb.Dataset(X, y)
    lgb_val = lgb.Dataset(val_X, val_y)
    gbm = lgb.train(params,
        lgb_train,
        num_boost_round=380,
        valid_sets= [lgb_val]
                    )

    prediction = gbm.predict(test)
    df['Probability'] = prediction
    df.to_csv("result-light-20170827.csv", index=False)
    # 输出运行时长
    cost_time = time.time() - start_time
    print("lightbgm success!", '\n', "cost time:", cost_time, "(s)......")

    #print('Feature names:', gbm.feature_name())

    print('Calculate feature importances...')
    # feature importances
    #print('Feature importances:', list(gbm.feature_importance()))

    result = pd.DataFrame({'feature': gbm.feature_name(), 'importances': gbm.feature_importance()})
    print(result.sort_values('importances'))