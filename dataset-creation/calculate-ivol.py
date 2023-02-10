'''
Author: Nasrin Seifi
Input: Mereged CRSP_DAILY_FF3 dataset
Functionality: calculate IVOL using millions of linear regression models
Output: a csv file for monthly IVOL (IVOL_complete.csv)
Execution Comment:
    - before running this code, run the following scripts to prepare the required pre-processed data for this script:
        dataset-creation/merge-crsp-famafrench.py
'''
import os
import pandas  as pd
import statsmodels.api as sm
from tqdm import tqdm
import numpy as np
########################################################################################################
# Daily CRSP and Fama-French Load pickles and calculate regression results
########################################################################################################
dir_name = "CRSP_FFRENCH_DAILY_GROUPED/"
# all_reseduals = [0.0 for _ in range(1011871)]
# res_cnt = 0
# filter trading days>15 and compute the IVOL
with open("IVOL_complete.csv", "w") as ivol_file:
    ivol_file.write("PERMNO, CUSIP, year, month, IVOL\n")
    for file_name in tqdm(os.listdir(dir_name)):
        FFC = pd.read_pickle(dir_name+file_name)
        if len(FFC) < 15:
            continue
        try:
            Y = (FFC["RET"].astype(float)-FFC["RF"].astype(float)).values
        except ValueError:
            continue
        X = FFC[["Mkt-RF","SMB","HML"]]
        X = sm.add_constant(X)
        X.rename(columns = {"const":"alpha"}, inplace = True)
        model = sm.OLS(Y.astype(float), X.astype(float))
        result = model.fit() #Set up regression
        predict_Y = result.predict(X.astype(float))

        residuals = Y.astype(float)-predict_Y
        # reseduals.sum() should be close to zero!
        IVOL = np.std(np.array(residuals)) * np.sqrt(len(FFC))
        ikeys = file_name.split("_")
        #res = "|".join([str(x) for x in residuals.tolist()])
        ivol_file.write("{},{},{},{},{}\n".format(ikeys[0], ikeys[1], ikeys[2].split("-")[0], ikeys[2].split("-")[1], IVOL))
