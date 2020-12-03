import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler

dataset1 = pd.read_csv('data/dataset1.csv')
dataset1.label.replace(-1,0,inplace=True)
dataset2 = pd.read_csv('data/dataset2.csv')
dataset2.label.replace(-1,0,inplace=True)
dataset3 = pd.read_csv('data/dataset3.csv')

dataset1.drop_duplicates(inplace=True)
dataset2.drop_duplicates(inplace=True)
dataset3.drop_duplicates(inplace=True)

dataset12 = pd.concat([dataset1,dataset2],axis=0)

# dataset1_y = dataset1.label
# dataset1_x = dataset1.drop(['user_id','label','day_gap_before','day_gap_after','is_man_jian','user_median_distance'],axis=1)  # 'day_gap_before','day_gap_after' cause overfitting, 0.77
# dataset2_y = dataset2.label
# dataset2_x = dataset2.drop(['user_id','label','day_gap_before','day_gap_after','is_man_jian','user_median_distance'],axis=1)
dataset12_y = dataset12.label
dataset12_x = dataset12.drop(['user_id','label','day_gap_before','day_gap_after','is_man_jian','user_median_distance'],axis=1)
dataset3_preds = dataset3[['user_id','coupon_id','date_received']]
dataset3_x = dataset3.drop(['user_id','coupon_id','date_received','day_gap_before','day_gap_after','is_man_jian','user_median_distance'],axis=1)

dataset12 = xgb.DMatrix(dataset12_x,label=dataset12_y)
dataset3 = xgb.DMatrix(dataset3_x)

data = dataset12_x
labels = dataset12_y

data_val = dataset3_x

lgb_train = lgb.Dataset(data, labels)
lgb_eval = lgb.Dataset(data, labels, reference=lgb_train)


# specify your configurations as a dict
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

print('Start training...')
# train
gbm = lgb.train(params,
				lgb_train,
				num_boost_round=380
				)

prediction = gbm.predict(data_val)

print('Feature names:', gbm.feature_name())

print('Calculate feature importances...')
# feature importances
print('Feature importances:', list(gbm.feature_importance()))

df = pd.DataFrame({'feature':gbm.feature_name(), 'importances': gbm.feature_importance()})
df.sort_values('importances',inplace=True)
df.to_csv("light_feature_score.csv")

dataset3_preds['label'] = prediction
dataset3_preds.label = MinMaxScaler().fit_transform(dataset3_preds.label)
dataset3_preds.sort_values(by=['coupon_id','label'],inplace=True)
dataset3_preds.to_csv("light_preds.csv",index=None,header=None)
print (dataset3_preds.describe())


