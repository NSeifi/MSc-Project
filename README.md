# Idiosyncratic_Volatility_Estimation

To downlaod the CRSP DAILY stocks:

https://wrds-web.wharton.upenn.edu/wrds/ds/crsp/stock_a/dsf.cfm?navId=128

To download the FF3 factors:

https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

Then follow the steps to process the data:

datasplit-daily-crsp-famafrench.py

merge-crsp-famafrench.py

calculate-ivol.py

split-ivol.py

create-nasdaq100-dataset.py

create-lagged-nasdaq100-dataset.py
