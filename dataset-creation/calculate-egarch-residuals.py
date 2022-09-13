"""
Author: Nasrin Seifi
Input: Mereged CRSP_DAILY_FF3 dataset
Functionality: calculate residuals for EGARCH model using millions of linear regression models
Output: csv files for each stock ticker
Execution Comment:
    - before running this code, run the following scripts to prepare the required pre-processed data for this script:
        dataset-creation/merge-crsp-famafrench.py
"""
import os
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm
import numpy as np

########################################################################################################
# Daily CRSP and Fama-French Load pickles and calculate regression residuals
########################################################################################################
N100_data = pd.read_csv("nasdaq-100.csv", delimiter='\t')
N100_data['FIRST DATE'] = pd.to_datetime(N100_data['FIRST DATE'], format='%Y%m%d')
N100_data['LAST DATE'] = pd.to_datetime(N100_data['LAST DATE'], format='%Y%m%d')
# N100_data['begin_year'] = N100_data['FIRST DATE'].dt.to_period('Y')
# N100_data['end_year'] = N100_data['LAST DATE'].dt.to_period('Y')
n100_processed = dict()
for ticker, ticker_group in N100_data.groupby("TICKER"):
    ticker = ticker.strip()
    ticker_group = ticker_group.sort_values(['FIRST DATE'])
    for index, row in ticker_group.iterrows():
        cusip_full = row['CUSIP FULL']
        # entity_name = row['ENTITY NAME']
        first_date = row['FIRST DATE']
        last_date = row['LAST DATE']
        permco = row['PERMCO']
        permno = row['PERMNO']
        n100_processed[f"{ticker}{cusip_full}{permco}{permno}"] = [first_date, last_date, []]

opened_tickers = dict()

dir_name = "CRSP_FFRENCH_DAILY_GROUPED/"
if not os.path.exists("egarch_company_preprocessed_csv/"):
    os.mkdir("egarch_company_preprocessed_csv/")
with open("IVOL_complete.csv", "w") as ivol_file:
    ivol_file.write("PERMNO, CUSIP, year, month, IVOL,residuals\n")
    for file_name in tqdm(os.listdir(dir_name)):
        FFC = pd.read_pickle(dir_name + file_name)
        if len(FFC) < 15:
            continue
        try:
            Y = (FFC["RET"].astype(float) - FFC["RF"].astype(float)).values
        except ValueError:
            continue
        X = FFC[["Mkt-RF", "SMB", "HML"]]
        X = sm.add_constant(X)
        X.rename(columns={"const": "alpha"}, inplace=True)
        model = sm.OLS(Y.astype(float), X.astype(float))
        result = model.fit()  # Set up regression
        predict_Y = result.predict(X.astype(float))

        residuals = Y.astype(float) - predict_Y
        IVOL = np.std(np.array(residuals)) * np.sqrt(len(FFC))
        for day, residual in zip(FFC.iterrows(), residuals):
            day = day[1]
            TICKER = day['TICKER']
            PERMNO = day['PERMNO']
            PERMCO = day['PERMCO']
            CUSIP = day['CUSIP']
            date_year = day['date'].year
            date_month = day['date'].month
            date_day = day['date'].day
            full_date = day['date'].date()

            lookup_key = f"{TICKER}{CUSIP}{PERMCO}{PERMNO}"
            if lookup_key in n100_processed and \
                    n100_processed[lookup_key][0] <= day['date'] <= n100_processed[lookup_key][1]:
                if TICKER not in opened_tickers:
                    opened_tickers[TICKER] = open(f'egarch_company_preprocessed_csv/{TICKER}.csv', 'w')
                    opened_tickers[TICKER].write("TICKER, year, month, day, full_date, IVOL, residual\n")
                opened_tickers[TICKER].write(f"{TICKER}, {date_year}, {date_month}, {date_day}, {full_date}, {IVOL}, {residual}\n")

for k in opened_tickers:
    opened_tickers[k].close()
