import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import gc
import time
from joblib import Parallel, delayed
import multiprocessing

start_time = time.time()

def cal_online_user_habbit(tag):
    ol_click_buy_merchant = pd.read_pickle("../data/ol_click_buy_data_%s.pkl" % tag)
    ol_coupon_merchant = pd.read_pickle("../data/ol_coupon_data_%s.pkl" % tag)

    ol_click_buy_count_merchant = pd.crosstab(ol_click_buy_merchant.User_id, ol_click_buy_merchant.Action)
    ol_click_buy_count_merchant.columns = ['ol_user_click', 'ol_user_buy']
    ol_click_buy_count_merchant.reset_index(inplace=True)
    ol_click_buy_count_merchant['ol_user_click_via_mean'] = ol_click_buy_count_merchant.ol_user_click / ol_click_buy_count_merchant.ol_user_click.mean()
    ol_click_buy_count_merchant['ol_user_buy_via_mean'] = ol_click_buy_count_merchant.ol_user_buy / ol_click_buy_count_merchant.ol_user_buy.mean()

    ol_merchant = ol_click_buy_merchant.loc[
        (ol_click_buy_merchant.Action == 1) & (ol_click_buy_merchant.Date_received != 'null')].reset_index(drop=True)
    ol_merchant = ol_merchant[['User_id']].groupby('User_id').size().reset_index()
    ol_merchant.columns = ['User_id', 'ol_user_use_coupon_count']
    ol_merchant['ol_user_use_coupon_count_via_mean'] = ol_merchant.ol_user_use_coupon_count / ol_merchant.mean()

    ol_coupon_merchant = pd.crosstab(ol_coupon_merchant.User_id, ol_coupon_merchant.Action)
    ol_coupon_merchant.columns = ['ol_user_get_coupon_num']
    ol_coupon_merchant.reset_index(inplace=True)
    ol_coupon_merchant['ol_user_get_coupon_num_via_mean'] = ol_coupon_merchant.ol_user_get_coupon_num/ol_coupon_merchant.ol_user_get_coupon_num.mean()

    df = pd.merge(ol_click_buy_count_merchant, ol_coupon_merchant, how="outer", on="User_id")
    df = pd.merge(df, ol_merchant, how="outer", on="User_id")
    df['ol_user_coupon_rate'] = df['ol_user_use_coupon_count'] / df['ol_user_buy']
    df.to_pickle("../data/feature/ol_user_%s.pkl" % tag)

def cal_online_merchant_feature(tag):
    ol_click_buy_merchant = pd.read_pickle("../data/ol_click_buy_data_%s.pkl" % tag)
    ol_coupon_merchant = pd.read_pickle("../data/ol_coupon_data_%s.pkl" % tag)

    ol_click_buy_count_merchant = pd.crosstab(ol_click_buy_merchant.Merchant_id, ol_click_buy_merchant.Action)
    ol_click_buy_count_merchant.columns = ['ol_click', 'ol_buy']
    ol_click_buy_count_merchant.reset_index(inplace=True)

    ol_merchant = ol_click_buy_merchant.loc[(ol_click_buy_merchant.Action==1) & (ol_click_buy_merchant.Date_received!='null')].reset_index(drop = True)
    ol_merchant = ol_merchant[['Merchant_id']].groupby('Merchant_id').size().reset_index()
    ol_merchant.columns =['Merchant_id','ol_use_coupon_count']

    ol_coupon_merchant = pd.crosstab(ol_coupon_merchant.Merchant_id, ol_coupon_merchant.Action)
    ol_coupon_merchant.columns = ['ol_get_coupon_num']
    ol_coupon_merchant.reset_index(inplace=True)
    df = pd.merge(ol_click_buy_count_merchant, ol_coupon_merchant, how="outer",on="Merchant_id")
    df = pd.merge(df, ol_merchant, how="outer", on="Merchant_id")
    df['ol_m_coupon_rate'] = df['ol_use_coupon_count'] / df['ol_buy']
    df.to_pickle("../data/feature/ol_m_%s.pkl" % tag)

def cal_online_merchant_user_feature(tag):
    ol_click_buy_merchant = pd.read_pickle("../data/ol_click_buy_data_%s.pkl" % tag)
    ol_coupon_merchant = pd.read_pickle("../data/ol_coupon_data_%s.pkl" % tag)

    ol_click_count_merchant_user = ol_click_buy_merchant[['Merchant_id','User_id','Action']][ol_click_buy_merchant.Action==0].groupby(['Merchant_id','User_id']).size().reset_index()
    ol_click_count_merchant_user.columns = ['Merchant_id','User_id','ol_mu_click']

    ol_buy_count_merchant_user = ol_click_buy_merchant[['Merchant_id', 'User_id', 'Action']][
        ol_click_buy_merchant.Action == 1].groupby(['Merchant_id', 'User_id']).size().reset_index()
    ol_buy_count_merchant_user.columns = ['Merchant_id', 'User_id', 'ol_mu_buy']
    ol_merchant = ol_click_buy_merchant.loc[
        (ol_click_buy_merchant.Action == 1) & (ol_click_buy_merchant.Date_received != 'null')].reset_index(drop=True)
    ol_merchant = ol_merchant[['Merchant_id','User_id']].groupby(['Merchant_id','User_id']).size().reset_index()
    ol_merchant.columns = ['Merchant_id','User_id', 'ol_mu_use_coupon_count']

    ol_coupon_merchant = ol_coupon_merchant[['Merchant_id','User_id']].groupby(['Merchant_id','User_id']).size().reset_index()
    ol_coupon_merchant.columns = ['Merchant_id','User_id','ol_mu_get_coupon_num']

    #df = pd.merge(ol_click_count_merchant_user, ol_buy_count_merchant_user, how="outer", on="Merchant_id")
    df = pd.merge(ol_buy_count_merchant_user, ol_merchant, how="outer", on=["Merchant_id",'User_id'])
    df = pd.merge(df, ol_coupon_merchant, how="outer", on=["Merchant_id",'User_id'])
    df['ol_mu_coupon_rate'] = df['ol_mu_use_coupon_count'] / df['ol_mu_buy']
    df.to_pickle("../data/feature/ol_mu_%s.pkl" % tag)

def cal_offline_feature(tag):
    merchant_consumption = pd.read_pickle("../data/off_consumption_data_%s.pkl" % tag)
    merchant_sample = pd.read_pickle("../data/off_sample_data_%s.pkl" % tag)
    print(merchant_sample.shape)
    exit()
    merchant = pd.concat([merchant_consumption, merchant_sample])

    merchant.Distance = merchant.Distance.astype(float)
    merchant_distance = merchant.groupby(['Merchant_id']).agg({'Merchant_id':['size'], 'Distance':['mean','median','max']})
    merchant_distance.columns = merchant_distance.columns.droplevel(0)
    merchant_distance.columns = ['off_m_buy_total','off_m_distance_mean','off_m_distance_median','off_m_distance_max']
    merchant_distance.reset_index(inplace=True)

    gc.collect()

    m_consumption = merchant[merchant.Date!='null'][['Merchant_id']]
    m_consumption['total_sales'] = 1
    m_consumption = m_consumption.groupby('Merchant_id').agg('sum').reset_index()

    merchant_positive_data = merchant_sample[['Merchant_id','User_id','Discount_rate','Date','Date_received']].loc[merchant_sample.Date!='null'].reset_index(drop=True)


    merchant_positive_data['Discount_percent'] = merchant_positive_data[['Merchant_id', 'Discount_rate']].apply(formatRate, axis=1)
    merchant_positive_data['Discount_percent'] = merchant_positive_data['Discount_percent'].fillna(-1).astype(float)
    merchant_positive_data.loc[merchant_positive_data.Discount_percent == -1, 'Discount_percent'] = np.nan
    merchant_positive_data['Date_received']=pd.to_datetime(merchant_positive_data.Date_received, format= '%Y%m%d', errors ='coerce')
    merchant_positive_data['Date']=pd.to_datetime(merchant_positive_data.Date, format= '%Y%m%d', errors ='coerce')
    merchant_positive_data['diff'] = merchant_positive_data.Date - merchant_positive_data.Date_received
    merchant_positive_data.loc[:, 'diff'] = merchant_positive_data['diff'].dt.days
    merchant_positive_data['sales_use_coupon'] = 1

    merchant_data = merchant_positive_data.groupby(['Merchant_id']).agg({'sales_use_coupon':['sum'],'diff':['mean','median'],'Discount_percent':['mean','median']})
    merchant_data.columns = merchant_data.columns.droplevel(0)
    merchant_data.columns = ['sales_use_coupon','off_m_diff_mean', 'off_m_diff_median','off_m_use_discount_mean', 'off_m_use_discount_median']
    merchant_data.reset_index(inplace=True)

    merchant_sample.loc[merchant_sample.Date=='null','Date'] = np.nan
    off_m = merchant_sample.groupby(['Merchant_id']).agg({'Date_received':['count'],'Date':['count']})
    off_m.columns = off_m.columns.droplevel(0)
    off_m.columns = ['off_m_get_coupon','off_m_use_coupon']
    off_m.reset_index(inplace=True)
    off_m['off_m_use_coupon_rate'] = off_m['off_m_use_coupon']/ off_m['off_m_get_coupon']
    off_m = pd.merge(off_m, merchant_data, how="left",on = "Merchant_id")
    off_m = pd.merge(off_m, merchant_distance, how="left",on = "Merchant_id")
    off_m = pd.merge(off_m, m_consumption, how="left", on="Merchant_id")
    off_m['off_coupon_use_rate'] = off_m.sales_use_coupon / off_m.total_sales
    off_m.to_pickle("../data/feature/off_m_%s.pkl" % tag)

    ################################

    all_user_merchant = merchant_sample[['User_id', 'Merchant_id']]
    all_user_merchant.drop_duplicates(inplace=True)

    merchant_distance = merchant.groupby(['Merchant_id','User_id']).agg({'Distance':['mean','median']})
    merchant = ''
    gc.collect()
    merchant_distance.columns = merchant_distance.columns.droplevel(0)
    merchant_distance.columns = ['off_mu_distance_mean','off_mu_distance_median']
    merchant_distance.reset_index(inplace=True)

    mu_consumption = merchant_consumption.groupby(['Merchant_id','User_id']).size().reset_index()
    mu_consumption.columns = ['Merchant_id','User_id','off_mu_consumption_times']


    merchant_user_data = merchant_positive_data.groupby(['Merchant_id','User_id']).agg({'diff': ['mean', 'median'],'Date_received':['count'],'Date':['count']})
    merchant_user_data.columns = merchant_user_data.columns.droplevel(0)
    merchant_user_data.columns = ['off_mu_diff_mean', 'off_mu_diff_median','off_mu_total','off_mu_total_coupon']
    merchant_user_data.reset_index(inplace=True)
    merchant_user_data['off_mu_coupon_rate'] = merchant_user_data['off_mu_total_coupon'] / merchant_user_data['off_mu_total']

    merchant_user_sample = merchant_sample.groupby(['User_id']).agg({'Date':['count']})
    merchant_user_sample.columns = merchant_user_sample.columns.droplevel(0)
    merchant_user_sample.columns = ['off_u_total_coupon']
    merchant_user_sample.reset_index(inplace=True)

    all_user_merchant = pd.merge(all_user_merchant, merchant_user_data, how="left",on = ["Merchant_id",'User_id'])
    all_user_merchant = pd.merge(all_user_merchant, merchant_distance, how="left",on =  ["Merchant_id",'User_id'])
    all_user_merchant = pd.merge(all_user_merchant, mu_consumption, how="left", on= ["Merchant_id",'User_id'])
    all_user_merchant = pd.merge(all_user_merchant, merchant_user_sample, how="left", on=['User_id'])
    all_user_merchant['off_mu_coupon_via_all'] = all_user_merchant.off_mu_total_coupon / all_user_merchant.off_u_total_coupon
    all_user_merchant.to_pickle("../data/feature/off_mu_%s.pkl" % tag)


def cal_coupon_feature(tag):
    # train feature
    merchant_sample = pd.read_pickle("../data/off_sample_data_%s.pkl" % tag)

    merchant_positive_data = merchant_sample[['Merchant_id', 'Date', 'Date_received','Coupon_received_diff']].loc[
        merchant_sample.Date != 'null'].reset_index(drop=True)

    merchant_positive_data['Date_received'] = pd.to_datetime(merchant_positive_data.Date_received, format='%Y%m%d',
                                                             errors='coerce')
    merchant_positive_data['Date'] = pd.to_datetime(merchant_positive_data.Date, format='%Y%m%d', errors='coerce')
    merchant_positive_data['coupon_diff'] = merchant_positive_data.Date - merchant_positive_data.Date_received
    merchant_positive_data.loc[:, 'coupon_diff'] = merchant_positive_data['coupon_diff'].dt.days
    merchant_positive_data.drop(['Date', 'Date_received'], axis=1, inplace=True)
    coupon_data = merchant_positive_data.groupby(['Merchant_id']).agg(
        {'coupon_diff': ['size', 'mean','median'], 'Coupon_received_diff':['mean','median']})
    coupon_data.columns = coupon_data.columns.droplevel(0)
    coupon_data.columns = ['off_merchant_coupon_use_count','off_merchant_coupon_diff_mean', 'off_merchant_coupon_diff_median', 'off_merchant_coupon_r_diff_mean', 'off_merchant_coupon_r_diff_median']
    coupon_data.off_merchant_coupon_use_count = coupon_data.off_merchant_coupon_use_count + 1
    coupon_data.reset_index(inplace=True)

    merchant_sample = merchant_sample.groupby(['Merchant_id']).size().reset_index()
    merchant_sample.columns = ['Merchant_id','coupon_count']

    merchant_sample = merchant_sample.merge(coupon_data, how="left", on="Merchant_id")
    # merchant_sample = merchant_sample.merge(merchant_coupon_data, how="left", on="Merchant_id")
    merchant_sample['off_merchant_coupon_rate'] = merchant_sample['off_merchant_coupon_use_count'] / merchant_sample['coupon_count']
    #merchant_sample.drop(['coupon_count'],axis=1,inplace=True)
    merchant_sample.to_pickle("../data/feature/off_coupon_%s.pkl" % tag)

def formatRate(text):
    merchant, rate = text
    if rate.find(":")>0:
        t = rate.split(":")
        v = int(t[0])
        q = int(t[1])
        return round((v-q)/v,2)
    elif rate =='fixed':
        return 0
    elif rate!='null':
        return float(rate)
    else:
        return np.nan

def formatRateValue(text):
    merchant, rate = text
    if rate.find(":")>0:
        t = rate.split(":")
        return t[0]
    else:
        return np.nan

def cal_train_data(tag):
    train = pd.read_pickle("../data/off_train_label_data_%s.pkl" % tag)
    train.drop(['type'], axis=1, inplace=True)

    train['Discount_percent'] = train[['Merchant_id', 'Discount_rate']].apply(formatRate, axis=1)
    train['Discount_percent'] = train['Discount_percent'].fillna(-1).astype(float)
    train['Discount_value'] = train[['Merchant_id', 'Discount_rate']].apply(formatRateValue, axis=1)
    train['Discount_value'] = train['Discount_value'].fillna(-1).astype(int)
    train['Date']=pd.to_datetime(train.Date, format= '%Y%m%d', errors ='coerce')
    train['day_diff'] = train.Date - train.Date_received
    train.loc[:, 'day_diff'] = train['day_diff'].dt.days
    train['coupon_in_15_day'] = 0
    train.loc[train.day_diff<=15, 'coupon_in_15_day'] = 1
    train.drop(["Discount_rate","Date"], axis=1, inplace=True)

    train.to_pickle("../data/feature/train_%s.pkl" % tag)

def cal_test_data(tag):
    test = pd.read_pickle("../data/off_train_label_data_test.pkl")
    test.drop(['type'], axis=1, inplace=True)

    test['Discount_percent'] = test[['Merchant_id', 'Discount_rate']].apply(formatRate, axis=1)
    test['Discount_percent'] = test['Discount_percent'].fillna(-1).astype(float)
    test['Discount_value'] = test[['Merchant_id', 'Discount_rate']].apply(formatRateValue, axis=1)
    test['Discount_value'] = test['Discount_value'].fillna(-1).astype(int)
    test.drop(["Discount_rate"], axis=1, inplace=True)

    test.to_pickle("../data/feature/train_%s.pkl" % tag)

def cal_label_data(tag):
    train = pd.read_pickle("../data/feature/train_%s.pkl" % tag)

    train['coupon_sort'] = train[['User_id', 'Coupon_id','Date_received']].groupby(['User_id', 'Coupon_id']).rank()
    df = train[['User_id', 'Coupon_id','Date_received']].groupby(['User_id', 'Coupon_id']).agg({'Date_received': ['size','min','max']})
    df.columns = df.columns.droplevel(0)
    df.columns = ['user_label_total_spe_coupon','user_label_coupon_min_date','user_label_coupon_max_date']
    df.reset_index(inplace=True)
    train = train.merge(df, how="left", on=["User_id", "Coupon_id"])
    df = train[['User_id','Date_received']].groupby(['User_id','Date_received']).size().reset_index()
    df.columns = ['User_id','Date_received','user_today_total_coupon']
    train = train.merge(df, how="left", on=["User_id","Date_received"])

    df = train[['User_id', 'Coupon_id', 'Date_received']].groupby(['User_id', "Coupon_id", 'Date_received']).size().reset_index()
    df.columns = ['User_id', 'Coupon_id', 'Date_received', 'user_today_total_spe_coupon']

    train = train.merge(df, how="left", on=["User_id", "Coupon_id", "Date_received"])
    train['user_today_total_spe_coupon'] = train.user_today_total_spe_coupon/ train.user_today_total_coupon
    train['user_since_last_coupon'] = train.user_label_total_spe_coupon - train.coupon_sort
    train['user_since_last_coupon_rate']  = train.coupon_sort / train.user_label_total_spe_coupon
    train['user_since_date_last_once'] = train.user_label_coupon_max_date - train.Date_received
    train['user_since_date_first_once'] = train.Date_received - train.user_label_coupon_min_date
    train['user_since_date_last_once'] = train['user_since_date_last_once'].dt.days
    train['user_since_date_first_once'] = train['user_since_date_first_once'].dt.days

    train.drop(["coupon_sort"], axis=1, inplace=True)

    train.to_pickle("../data/feature/train_format_%s.pkl" % tag)




def cal_full_data(tag):
    full_data = pd.read_pickle("../data/off_full_data_format_%s.pkl" % tag)
    full_data.drop(['type','index'],axis=1,inplace=True)
    full_data['Coupon_received_diff_prev'] = full_data.Coupon_received_diff.shift(-1)


    #full_data['Coupon_received_diff_per_15'] = full_data.Coupon_received_diff_prev / 15
    # merchant_distance = full_data.groupby(['Merchant_id']).agg({'Distance':['mean','median']})
    # merchant_distance.columns = merchant_distance.columns.droplevel(0)
    # merchant_distance.columns = ['ofl_m_distance_mean_new','ofl_m_distance_median_new']
    # merchant_distance.reset_index(inplace=True)

    df = full_data.groupby(['Coupon_id']).agg({'Distance':['mean','median'],'Coupon_received_diff':['mean','median']})
    df.columns = df.columns.droplevel(0)
    df.columns = ['full_coupon_distance_mean', 'full_coupon_distance_median','full_coupon_diff_mean','full_coupon_diff_median']
    df.reset_index(inplace=True)

    df1 = full_data.groupby(['User_id','Coupon_id']).size().reset_index()
    df1.columns = ['User_id','Coupon_id','total_user_spec_coupon']
    #
    # df2 = full_data.groupby(['User_id']).size().reset_index()
    # df2.columns = ['User_id', 'total_user_coupon']
    # df1= df1.merge(df2, how="left",on="User_id")
    # df1['user_spec_coupon_rate'] = df1.total_user_spec_coupon/df1.total_user_coupon

    #full_data = full_data.merge(merchant_distance, how="left", on=["Merchant_id"])
    full_data = full_data.merge(df, how="left",on=["Coupon_id"])
    full_data = full_data.merge(df1, how="left", on=['User_id','Coupon_id'])
    #full_data = full_data.merge(df2, how="left", on=['User_id'])

    # full_data['coupon_sort'] = full_data[['User_id', 'Coupon_id', 'Date_received']].groupby(
    #     ['User_id', 'Coupon_id']).rank()
    # full_data['user_total_since_last_coupon'] = full_data.total_user_spec_coupon - full_data.coupon_sort
    #full_data['user_total_since_last_coupon_rate'] = full_data.coupon_sort / full_data.total_user_spec_coupon
    #full_data.drop(["coupon_sort"], axis=1, inplace=True)

    if tag=='test':
        full_data.drop(['Discount_rate', 'Distance', 'Date_received','Date_received_int','Coupon_received_diff_prev','Coupon_received_diff'], axis=1,
                       inplace=True)
    else:
        full_data.drop(['Discount_rate','Distance','Date_received','Date_received_int', 'Date','Coupon_received_diff_prev','Coupon_received_diff'],axis=1, inplace=True)

    full_data.drop_duplicates(inplace=True)
    full_data.to_pickle("../data/feature/full_%s.pkl" % tag)

def combine_train_data(tag, y,m,d):
    train = pd.read_pickle("../data/feature/train_format_%s.pkl" % tag)
    full = pd.read_pickle("../data/feature/full_%s.pkl" % tag)
    coupon = pd.read_pickle("../data/feature/off_coupon_%s.pkl" % tag)
    ol_user = pd.read_pickle("../data/feature/ol_user_%s.pkl" % tag)
    off_m = pd.read_pickle("../data/feature/off_m_%s.pkl" % tag)
    off_mu = pd.read_pickle("../data/feature/off_mu_%s.pkl" % tag)
    train = train.merge(coupon,how="left",on=["Merchant_id"])
    train = train.merge(ol_user, how="left", on=["User_id"])
    train = train.merge(off_m, how="left", on=["Merchant_id"])
    train = train.merge(off_mu, how="left", on=["Merchant_id","User_id"])

    full.drop(['User_id','Coupon_id','Merchant_id'],axis=1, inplace=True)
    train = train.merge(full, how="left", on=["ID"])

    train['days_distance'] = train.Date_received_int.astype('str').apply(
        lambda x: (date(int(x[0:4]), int(x[4:6]), int(x[6:8])) - date(y, m, d)).days)
    train['day_of_month'] = train.Date_received_int.astype('str').apply(lambda x: int(x[6:8]))
    train['day_of_week'] = train.Date_received_int.astype('str').apply(
        lambda x: date(int(x[0:4]), int(x[4:6]), int(x[6:8])).weekday() + 1)
    train['is_weekend'] = train.day_of_week.apply(lambda x: 1 if x in (6, 7) else 0)
    #train['user_label_area_coupon_rate'] = train.user_label_total_spe_coupon / train.total_user_spec_coupon
    #train['Coupon_received_diff_per_15'] = train.Coupon_received_diff / 15
    #train.drop(['Coupon_received_diff'],axis=1,inplace=True)
    train.drop(['day_of_week'],axis=1,inplace=True)
    train.drop_duplicates(inplace=True)

    train.to_pickle("../data/train/%s.pkl" % tag)

    if tag=='train':
        train.to_csv("test.csv",index=False)

def cal(tag,y,m,d):
    cal_full_data(tag)
    cal_online_user_habbit(tag)
    cal_online_merchant_user_feature(tag)

    cal_offline_feature(tag)
    cal_coupon_feature(tag)
    if tag =='test':
        cal_test_data(tag)
    else:
        cal_train_data(tag)
    cal_label_data(tag)
    combine_train_data(tag,y,m,d)

cal('train',2016,5,30)
cal('validation',2016,6,15)
cal('test',2016,7,31)

cost_time = time.time()-start_time
print ("Calculating Feature Success!",'\n',"Cost time:",cost_time,"(s)......")