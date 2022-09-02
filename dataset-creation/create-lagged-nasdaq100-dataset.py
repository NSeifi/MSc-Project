"""
Author: Nasrin Seifi
Input: company_preprocessed_csv dataset
Functionality: Takes the company_preprocessed_csv dataset and reassigns the IVOL values for each month to its previous
 month trade records.
Output: csv files for each stock ticker with lagged IVOL values

Execution Comment:
    - before running this code, run the following scripts to prepare the required pre-processed data for this script:
        dataset-creation/create-nasdaq-100-dataset.py
"""
import os
from tqdm import tqdm
from glob import glob

if __name__ == '__main__':
    assert os.path.exists("company_preprocessed_csv/")
    if not os.path.exists("lagged_company_preprocessed_csv/"):
        os.mkdir("lagged_company_preprocessed_csv/")
    for ticker_file in tqdm(glob("company_preprocessed_csv/*")):
        ticker_data = open(ticker_file, "r").read().split("\n")
        header = ticker_data[0]
        previous_month = []
        current_year_month = None
        result_ticker_file = open("lagged_"+ticker_file, "w")
        result_ticker_file.write(header + "\n")
        for trade_record in ticker_data[1:]:
            if not trade_record:
                continue
            trade_r = trade_record.split(",")
            ivol = trade_r[-1]
            year_month = trade_r[-4]
            if year_month != current_year_month and previous_month:
                for r in previous_month:
                    r[-1] = ivol
                    result_ticker_file.write(",".join(r) + "\n")
                del previous_month[:]
            previous_month.append(trade_r)
            current_year_month = year_month
        result_ticker_file.close()

