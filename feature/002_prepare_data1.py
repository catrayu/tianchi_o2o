import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import time
from joblib import Parallel, delayed
import multiprocessing as mp

def foramtNewFullData(tag):
    full_data = pd.read_pickle("../data/off_full_data_%s.pkl" % tag)
    full_data_format = pd.read_pickle("../data/off_full_data_new_%s.pkl" % tag)

    df = full_data_format[['index','deals_num_in_15_day']]
    full_data.reset_index(inplace=True)
    full_data = full_data.merge(df, how="left", on="index")
    full_data.to_pickle("../data/off_full_data_format_%s.pkl" % tag)


foramtNewFullData('train')
foramtNewFullData('validation')
foramtNewFullData('test')
