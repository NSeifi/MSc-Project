"""
Author: Nasrin Seifi
Input: CRSP-DAILY-BY-YEAR, F-F-DAILY-BY-YEAR, and IVOL-BY-YEAR datasets
Functionality: Merge Daily stock and FF3 with date and calculated IVOL data,
then for each company in NASDAQ-100 index (based on CUSIP, PERMCO, and PERMNO) create a separate dataset.
Output: csv files for each stock ticker

Execution Comment:
    - before running this code, run the following scripts to prepare the required pre-processed data for this script:
        dataset-creation/split-ivol.py
"""
import os
import pandas as pd
from tqdm import tqdm


def read_and_merge(year):
    crsp_d = pd.read_csv('CRSP-DAILY-BY-YEAR/{}.csv'.format(year), low_memory=False)
    crsp_d['date'] = pd.to_datetime(crsp_d['date'], format='%d%b%Y')
    ffrench_d = pd.read_csv('F-F-DAILY-BY-YEAR/{}.csv'.format(year))
    ffrench_d['date'] = pd.to_datetime(ffrench_d['date'], format='%Y%m%d')

    crsp_ffrench_d = pd.merge(crsp_d, ffrench_d, how='inner', on=['date'])
    crsp_ffrench_d['year_month'] = crsp_ffrench_d['date'].dt.to_period('M').astype(str)

    ivol = pd.read_csv('IVOL-BY-YEAR/{}.csv'.format(year))
    merged_dataset = pd.merge(crsp_ffrench_d, ivol, how='inner', on=['PERMNO', 'CUSIP', 'year_month'])
    for x in ['Mkt-RF', 'SMB', 'HML', 'RF', 'vwretx', 'VOL', 'RET']:
        merged_dataset[x] = pd.to_numeric(merged_dataset[x], errors='coerce').fillna(0)
    return merged_dataset


if __name__ == '__main__':
    if not os.path.exists("company_preprocessed_csv/"):
        os.mkdir("company_preprocessed_csv/")

    result = dict()

    for year in tqdm(range(1968, 2020)):
        crsp_ffrench_ivol = read_and_merge(year)

        N100_data = pd.read_csv("nasdaq_100.csv", delimiter='\t')
        N100_data['FIRST DATE'] = pd.to_datetime(N100_data['FIRST DATE'], format='%Y%m%d')
        N100_data['LAST DATE'] = pd.to_datetime(N100_data['LAST DATE'], format='%Y%m%d')
        # N100_data['begin_year'] = N100_data['FIRST DATE'].dt.to_period('Y')
        # N100_data['end_year'] = N100_data['LAST DATE'].dt.to_period('Y')
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

                selected = crsp_ffrench_ivol[crsp_ffrench_ivol['CUSIP'] == cusip_full]
                mask = (selected['date'] >= first_date) & (selected['date'] <= last_date)
                selected = selected.loc[mask]
                selected = selected[selected['PERMCO'] == permco]
                selected = selected[selected['PERMNO'] == permno]
                if not selected.empty:
                    if ticker not in result:
                        result[ticker] = selected
                    else:
                        # Merge the data frames
                        previous_years_data = result[ticker]
                        result[ticker] = pd.concat([previous_years_data, selected])

    for k in result:
        result[k].to_csv(f'company_preprocessed_csv/{k}.csv')
