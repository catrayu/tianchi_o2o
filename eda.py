import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#test = pd.read_csv("input/ccf_offline_stage1_test_revised.csv")
#df = test[['User_id','Coupon_id','Date_received']]
#df.drop_duplicates().to_pickle("../data/test.pkl")
train_online = pd.read_csv("input/ccf_online_stage1_train.csv")
print(train_online[train_online.Action==2][train_online.Date!='null'].head(10))
exit()
train_offline = pd.read_csv("input/ccf_offline_stage1_train.csv")

t = train_offline[['Merchant_id']]
t.is_copy = False
t.drop_duplicates(inplace=True)

df = pd.merge(train_online, t, how="inner",on="Merchant_id")
print(df.shape)
#print(df.User_id.unique().shape)
exit()
# train_online_user = train_online[['Coupon_id']].drop_duplicates()
# train_offline_user = train_offline[['Coupon_id']].drop_duplicates()
#
# test_user = test[['Coupon_id']].drop_duplicates()
# print("Train Online User",train_online_user.shape)
# print("Train Offline User",train_offline_user.shape)
# print(test.shape,"Test User",test_user.shape)
# df = train_online_user.merge(train_offline_user, how="inner",on="Coupon_id")
# df1 = test_user.merge(train_online_user, how="inner",on="Coupon_id")
#
# print(df.shape)
# print(df1.shape)
#print(test[test.User_id==209])
print(train_online[train_online.User_id==10516465])
#print(train_online[train_online.User_id ==209])



# sns.countplot(x="Action", data=train_online)
# plt.show()