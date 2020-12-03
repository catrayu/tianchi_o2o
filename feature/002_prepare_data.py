import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import time
from joblib import Parallel, delayed
import multiprocessing as mp

full_data = pd.read_pickle("../data/off_full_data_test.pkl")
full_data.drop(['type'], axis=1, inplace=True)
full_data['Coupon_received_diff_prev'] = full_data.Coupon_received_diff.shift(-1)
full_data.reset_index(inplace=True)
full_data['deadline'] = pd.to_datetime(full_data.Date_received+pd.DateOffset(days=15),format="%Y%m%d")

full_data = full_data[pd.notnull(full_data.Coupon_received_diff_prev)]
print(full_data.shape)

def process(df):
    df['deals_num_in_15_day'] = df.apply(calDealsIn15Day, axis=1)
    return df

def calDealsIn15Day(x):
    global full_data
    days = x['Coupon_received_diff_prev']
    deadline = x['deadline'].year * 10000 + x['deadline'].month * 100 + x['deadline'].day
    startDay = x['Date_received_int']
    index = x['index']
    user_id = x['User_id']
    coupon_id = x['Coupon_id']
    max_days = full_data[full_data.index > index][full_data.User_id == user_id][full_data.Coupon_id == coupon_id][
                    full_data.Date_received_int <= deadline][full_data.Date_received_int >= startDay]['Coupon_received_diff_prev'].count()
    print('User_id:', user_id, ' Coupon_id:', coupon_id)
    return max_days

def processData(tag):
    global full_data
    p = mp.Pool(8)
    pool_results = p.map(process, np.array_split(full_data, 8))
    p.close()
    p.join()
    parts = pd.concat(pool_results, axis=0)
    parts.to_pickle("../data/off_full_data_new_%s.pkl" % tag)
    parts.drop(['deadline'],axis=1,inplace=True)

if __name__ == '__main__':
    start_time = time.time()
    processData('train')
    processData('validation')
    processData('test')


    cost_time = time.time()-start_time
    print ("Calculating Feature Success!",'\n',"Cost time:",cost_time,"(s)......")