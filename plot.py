import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dtype = {'SHRCD': float, 'EXCHCD': float, 'SICCD': float, 'ISSUNO': float, 'HEXCD': float, 'HSICCD': float,
         'DLAMT': float, 'DLSTCD': float, 'SHRFLG': float, 'HSICMG': float, 'HSICIG': float, 'DISTCD': float,
         'DIVAMT': float, 'FACPR': float, 'FACSHR': float, 'ACPERM': float, 'ACCOMP': float, 'NWPERM': float,
         'DLRETX': float, 'DLPRC': float, 'DLRET': float, 'TRTSCD': float, 'NMSIND': float, 'MMCNT': float,
         'NSDINX': float, 'BIDLO': float, 'ASKHI': float, 'PRC': float, 'VOL': float, 'RET': float, 'BID': float,
         'ASK': float, 'SHROUT': float, 'CFACPR': float, 'CFACSHR': float, 'OPENPRC': float, 'NUMTRD': float,
         'RETX': float, 'vwretd': float, 'vwretx': float, 'ewretd': float, 'ewretx': float, 'sprtrn': float,
         'Mkt-RF': float, 'SMB': float, 'HML': float, 'RF': float, 'IVOL': float}

#filename = "preprocess/company_preprocessed_csv/{}.csv"
filename = "preprocess/lagged_company_preprocessed_csv/{}.csv"

ticker = ['MSFT', 'SIRI', 'OKTA', 'GOOG', 'ALGN', 'REGN', 'CHTR', 'NVDA', 'NFLX', 'ILMN', 'IDXX', 'NTES', 'VRTX']
for ticker in ticker:
    data = pd.read_csv(filename.format(ticker))
    datas = data[['PRC', 'RET', 'IVOL']]
    data = data[data['IVOL'].notna()]
    data = data[data['PRC'] >= 0]
    data = data.fillna(0)
    traindata = data[['RET', 'IVOL', 'year_month', 'year', 'month']]
    x1 = np.zeros(1)
    x2 = np.zeros(1)
    y = np.zeros(1)
    for year_index, year in enumerate([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]):
        ret = traindata[traindata['year'] == year].groupby(['year', 'month']).mean()
        x1 = np.append(x1, ret.to_numpy()[:, 0].tolist())
        x2 = np.append(x2, ret.to_numpy()[:, 1].tolist())
        for element in data[data['year'] == year].groupby(['year', 'month']):
            month = int(list(element[1].iterrows())[0][1]['month'])
            month = str(year) + '_' + str(month)
            y = np.append(y, month)
            continue

    data = data[['PRC', 'RET', 'IVOL']]
    print(data.describe())
    y[0] = y[1]
    y = pd.to_datetime(y, format="%Y_%m")
    plt.plot(y, [t * 10 for t in x1], 'r-', y, x2, 'b-')
    plt.ylabel('')
    plt.grid(axis='x', color='0.95')
    plt.xlabel("Year")
    plt.ylabel("IVOL and (Return * 10)")
    plt.gca().legend(["return", "IVol"])

    fname = ticker + ".png"
    plt.savefig(fname)
    plt.show()
