'''
Author: Nasrin Seifi
Input: CRSP-DAILY-BY-YEAR and F-F-DAILY-BY-YEAR dataset
Functionality: Merge Daily stock and FF3 with date,
then for each month group by each stock based on PERMNO and CUSIP
Output: Pickle files for each stock in a month
Execution Comment:
    - before running this code, run the following scripts to prepare the required pre-processed data for this script:
        preprocess/daily_ff3_data_split.py
'''
import os
import pandas as pd
import numpy as np
from pandas.tseries.offsets import *
from tqdm import tqdm

########################################################################################################
# Daily CRSP and Fama-French Merge
########################################################################################################
os.mkdir("CRSP_FFRENCH_DAILY_GROUPED/")
for year in tqdm(range(1963, 2020)):
    # crsp_d = pd.read_csv('CRSP-DAILY-BY-YEAR/{}.csv'.format(year),
    #                     usecols=['PERMNO', 'date', 'SHRCD', 'NCUSIP', 'TICKER', 'PERMCO',
    #                              'CUSIP', 'BIDLO', 'ASKHI', 'PRC', 'VOL', 'RET', 'BID', 'ASK',
    #                              'RETX', 'vwretd', 'vwretx', 'ewretd', 'ewretx'])
    crsp_d = pd.read_csv('CRSP-DAILY-BY-YEAR/{}.csv'.format(year), dtype=str)
    crsp_d['date'] = pd.to_datetime(crsp_d['date'], format='%d%b%Y')
    ffrench_d = pd.read_csv('F-F-DAILY-BY-YEAR/{}.csv'.format(year), usecols=['date', 'Mkt-RF', 'SMB', 'HML', 'RF'])
    ffrench_d['date'] = pd.to_datetime(ffrench_d['date'], format='%Y%m%d')

    crsp_ffrench_d = pd.merge(crsp_d, ffrench_d, how='inner', on=['date'])
    crsp_ffrench_d['year_month'] = crsp_ffrench_d['date'].dt.to_period('M')
    grouped_df = crsp_ffrench_d.groupby(['PERMNO', 'CUSIP', 'year_month'])
    for key, item in grouped_df:
        grouped_df.get_group(key).reset_index().to_pickle(
            "CRSP_FFRENCH_DAILY_GROUPED/{}.pkl".format(str(key[0]) + "_" + key[1] + "_" + key[2].strftime("%Y-%m")))
