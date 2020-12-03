import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def split_online_data(start_date=20160101, end_date= 20160630, tag = 'train'):
    train_online = pd.read_csv("../input/ccf_online_stage1_train.csv")
    online_coupon_data = train_online.loc[train_online.Action == 2]
    online_coupon_data.Date_received =online_coupon_data.Date_received.astype(int)
    online_coupon_dataset = online_coupon_data.loc[
        (online_coupon_data.Date_received >= start_date) & (online_coupon_data.Date_received <= end_date)].reset_index()
    online_coupon_dataset.drop(['index'], axis=1, inplace=True)
    online_coupon_dataset.to_pickle("../data/ol_coupon_data_%s.pkl" % tag)
    online_click_buy_data = train_online.loc[(train_online.Action == 0)| (train_online.Action == 1)]
    online_click_buy_data.Date =online_click_buy_data.Date.astype(int)
    online_click_buy_dataset = online_click_buy_data.loc[
        (online_click_buy_data.Date >= start_date) & (online_click_buy_data.Date <=end_date)].reset_index()
    online_click_buy_dataset.drop(['index'], axis=1, inplace=True)
    online_click_buy_dataset.to_pickle("../data/ol_click_buy_data_%s.pkl" % tag)

def split_train_and_label_data(start_date, end_date, label_start_date,label_end_date, tag):
    train_offline = pd.read_csv("../input/ccf_offline_stage1_train.csv")

    offline_consumption_data = train_offline.loc[
        (train_offline.Date != 'null') & (train_offline.Coupon_id == 'null')].reset_index(drop=True)
    offline_consumption_data.Date = offline_consumption_data.Date.astype(int)
    offline_consumption_data = offline_consumption_data.loc[
        (offline_consumption_data.Date >= start_date) & (offline_consumption_data.Date < label_start_date)].reset_index(drop=True)
    offline_consumption_data.to_pickle("../data/off_consumption_data_%s.pkl" % tag)

    offline_sample_data = train_offline.loc[train_offline.Date_received != 'null'].reset_index()

    offline_sample_data.Date_received = offline_sample_data.Date_received.astype(int)
    offline_sample_data.drop(['index'], axis=1, inplace=True)
    offline_sample_data.loc[offline_sample_data.Coupon_id=='null','Coupon_id'] =-1
    offline_sample_data.Coupon_id = offline_sample_data.Coupon_id.astype(int)
    offline_sample_data.loc[offline_sample_data.Coupon_id == -1, 'Coupon_id'] = np.nan

    offline_sample_data = offline_sample_data.loc[
        (offline_sample_data.Date_received >= start_date) & (
        offline_sample_data.Date_received <= end_date)].reset_index()

    if tag == 'test':
        test = pd.read_csv("../input/ccf_offline_stage1_test_revised.csv")
        test.Coupon_id = test.Coupon_id.astype(int)
        test.Date_received = test.Date_received.astype(int)
        test['Date'] = np.nan
        offline_sample_data.drop(['index'], axis=1, inplace=True)
        offline_sample_data.Coupon_id = offline_sample_data.Coupon_id.astype(int)
        offline_sample_data = pd.concat([offline_sample_data, test])
        offline_sample_data.Date_received = offline_sample_data.Date_received.astype(int)
    else:
        offline_sample_data.drop(['index'], axis=1, inplace=True)
    offline_sample_data['Date_received_int'] = offline_sample_data['Date_received']
    offline_sample_data.sort_values(['User_id', 'Coupon_id', 'Date_received_int'], ascending=[True,True,True],inplace=True)
    offline_sample_data.loc[offline_sample_data.Distance == 'null', 'Distance'] = np.nan
    offline_sample_data.Distance = offline_sample_data.Distance.fillna(-1)
    offline_sample_data.Distance = offline_sample_data.Distance.astype(int)
    offline_sample_data.loc[offline_sample_data.Distance == -1, 'Distance'] = np.nan

    offline_sample_data['Date_received'] = pd.to_datetime(offline_sample_data.Date_received, format='%Y%m%d', errors='coerce')
    offline_sample_data['Coupon_received_diff'] = offline_sample_data[['User_id', 'Coupon_id', 'Date_received']].groupby(
        ['User_id', 'Coupon_id']).diff()

    offline_sample_data['Coupon_received_diff'] = offline_sample_data['Coupon_received_diff'].dt.days
    offline_sample_data.reset_index(inplace=True, drop=True)
    offline_sample_data['type'] = 'train'
    offline_sample_data.reset_index(inplace=True)
    offline_sample_data.rename(columns={'index': 'ID'}, inplace=True)
    offline_sample_data.to_pickle("../data/off_full_data_%s.pkl" % tag)


    offline_train_data = offline_sample_data.loc[
        (offline_sample_data.Date_received_int >= start_date) & (
            offline_sample_data.Date_received_int < label_start_date)].reset_index(drop=True)

    print(offline_train_data.shape)

    offline_train_data.to_pickle("../data/off_sample_data_%s.pkl" % tag)
    exit()
    offline_label_data = offline_sample_data.loc[
        (offline_sample_data.Date_received_int >= label_start_date) & (
        offline_sample_data.Date_received_int <= label_end_date)].reset_index(drop=True)
    offline_label_data['type']='label'
    if tag=='test':
        offline_label_data.drop(['Date'], axis=1, inplace=True)
    offline_label_data.to_pickle("../data/off_train_label_data_%s.pkl" % tag)


# split_online_data(20160101,20160414,'train')
#split_train_and_label_data(20160316,20160530,20160501,20160530, 'train')
split_train_and_label_data(20160101,20160514,20106414,20160514, 'train')

# split_online_data(20160201,20160514,'validation')
#split_train_and_label_data(20160401,20160615,20160516,20160615, 'validation')
split_train_and_label_data(20160201,20160615,20160515,20160615, 'validation')

# split_online_data(20160315,20160630,'test')
#split_train_and_label_data(20160516,20160630,20160701,20160731, 'test')
split_train_and_label_data(20160315,20160630,20160701,20160731, 'test')