'''
Author: Nasrin Seifi
Input: IVOL_complete dataset
Functionality: Split IVOL data to reduce the memory requirement
Output: IVOL for each year for each stock
Execution Comment:
    - before running this code, run the following scripts to prepare the required pre-processed data for this script:
        dataset-creation/calculate-ivol.py
'''
import datetime
from tqdm import tqdm
import os
########################################################################################################
# IVOL data split to reduce the memory requirement
########################################################################################################
f = open('IVOL_complete.csv', "r", encoding="utf-8")
year_files = {}
# TODO do not create if it already exists
os.mkdir("IVOL-BY-YEAR")
res_folder_name = 'IVOL-BY-YEAR/{}.csv'
hdr = next(f)
headers = [x.strip() for x in hdr.split(",")]  # should be of size 5
for line in tqdm(f):
    l =  {k.strip(): v for v, k in zip(line.split(","), headers)}
    year = l['year']
    month = l['month'].split(".")[0]
    PERMNO = l['PERMNO']
    CUSIP = l['CUSIP']
    IVOL = l['IVOL'].strip()
    residuals = l['residuals'].strip()
    year_month = year + "-" + month
    if year not in year_files:
        year_files[year] = open(res_folder_name.format(year), "w", encoding="utf-8")
        year_files[year].write("PERMNO,CUSIP,year,month,year_month,IVOL,residuals\n")
    year_files[year].write("{},{},{},{},{},{},{}\n".format(PERMNO, CUSIP, year, month, year_month, IVOL, residuals))
f.close()
for k in year_files:
    year_files[k].close()
