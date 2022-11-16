# Idiosyncratic_Volatility_Estimation

To downlaod the CRSP DAILY stocks:

https://wrds-web.wharton.upenn.edu/wrds/ds/crsp/stock_a/dsf.cfm?navId=128

To download the FF3 factors:

https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

Then follow the steps to process the data:

dataset-creation/datasplit-daily-crsp-famafrench.py

dataset-creation/merge-crsp-famafrench.py

dataset-creation/calculate-ivol.py

dataset-creation/split-ivol.py

dataset-creation/create-nasdaq100-dataset.py

dataset-creation/create-lagged-nasdaq100-dataset.py

feature-extractors.py

decision-tree-experiments.py

lstm-experiments.py

lstm-monthly-experiments.py
