'''
Author: Nasrin Seifi
Input: CRSP_Daily_Stock and FamaFrench_3factor dataset
Functionality: data split to reduce the memory requirement
Output: 2 Directories which have splited data by year
Execution Comment:
    - For execution of this script you need the following files:
        - CRSP-DAILY STOCK COMPLETE.csv which should be downloaded from WRDS
        - F-F_Research_Data_Factors_daily.CSV which should be downloaded from the web
'''

import datetime
from tqdm import tqdm
import os
########################################################################################################
# Daily CRSP data split to reduce the memory requirement
########################################################################################################
f = open('CRSP-DAILY STOCK COMPLETE.csv', "r", encoding="utf-8")
year_files = {}
# TODO make sure the folder is not recreated!
os.mkdir("CRSP-DAILY-BY-YEAR/")
res_folder_name = 'CRSP-DAILY-BY-YEAR/{}.csv'
hdr = next(f)
headers = hdr.split(",") # should be of size 62
for line in tqdm(f):
    l =  {k: v for v, k in zip(line.split(","), headers)}
    year = datetime.datetime.strptime(l['date'], '%d%b%Y').year
    if year not in year_files:
        year_files[year] = open(res_folder_name.format(year), "w", encoding="utf-8")
        year_files[year].write(hdr.strip() + "\n")
    year_files[year].write(line.strip() + "\n")
f.close()
for k in year_files:
    year_files[k].close()
###############################################################################
# Fama-French data split to reduce the memory requirement
########################################################################################################
f = open('F-F_Research_Data_Factors_daily.CSV', "r", encoding="utf-8")
year_files = {}
# TODO make sure the folder is not recreated!
os.mkdir("F-F-DAILY-BY-YEAR/")
res_folder_name = 'F-F-DAILY-BY-YEAR/{}.csv'
hdr = next(f)
headers = hdr.split(",") # should be of size 4
for line in tqdm(f):
    l =  {k: v for v, k in zip(line.split(","), headers)}
    year = datetime.datetime.strptime(l['date'], '%Y%m%d').year
    if year not in year_files:
        year_files[year] = open(res_folder_name.format(year), "w", encoding="utf-8")
        year_files[year].write(hdr.strip() + "\n")
    year_files[year].write(line.strip() + "\n")
f.close()
for k in year_files:
    year_files[k].close()
