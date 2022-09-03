import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

pd.options.mode.chained_assignment = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_feature_set_file = "preprocess/best_feature_set.csv"
ticker_file_name_format = "preprocess/company_preprocessed_csv/{}.csv"


class MonthlyPredictor(nn.Module):
    def __init__(self, n_experiment_features=7, n_hidden_layer=32, n_out_labels=1, encoder_layers=2):
        super(MonthlyPredictor, self).__init__()
        self.emb = nn.Linear(n_experiment_features, n_hidden_layer, bias=True)
        self.encoder = nn.LSTM(n_hidden_layer, n_hidden_layer, encoder_layers)
        self.generator = nn.Linear(n_hidden_layer, n_out_labels, bias=True)
        self.encoder_layers = encoder_layers
        self.encoder_bidirectional = False
        self.encoder_hidden = n_hidden_layer
        self.criterion = nn.MSELoss()

    def forward(self, input_tensor, target_tensor=None, previous_context=None):
        """
        :param input_tensor: number of trade days * number of features
        :param target_tensor: number of target classes for that specific month
        :return: predicted classes for that month and the prediction loss value
        """
        e = self.emb(input_tensor).unsqueeze(1)
        init_context = self.encoder_init(1) if previous_context is None else previous_context
        encoded, output_context = self.encoder(e, init_context)
        pred = self.generator(output_context[0]).view(-1)
        loss = 0
        if target_tensor is not None:
            loss = self.criterion(pred, target_tensor)
        return pred, loss, output_context

    def encoder_init(self, batch_size):
        # num_layers * num_directions, batch, hidden_size
        return torch.zeros(self.encoder_layers * (2 if self.encoder_bidirectional else 1),
                           batch_size, self.encoder_hidden, device=device, dtype=torch.float32), \
               torch.zeros(self.encoder_layers * (2 if self.encoder_bidirectional else 1),
                           batch_size, self.encoder_hidden, device=device, dtype=torch.float32)


def inline_normalize(dataset, experiment_features, n_params=None):
    _params = {} if n_params is None else n_params
    for x in experiment_features:
        if x not in _params:
            _params[x] = {'min': dataset[x].min(), 'max': dataset[x].max(), 'mean': dataset[x].mean()}
        if _params[x]['min'] == _params[x]['max']:
            dataset[x] = dataset[x] - _params[x]['min']
        else:
            dataset[x] = (dataset[x] - _params[x]['min']) / (_params[x]['max'] - _params[x]['min'])
            if 'mean' not in _params[x]:
                _params[x]['mean'] = dataset[x].mean()
            dataset[x] = (dataset[x] - _params[x]['mean'])
    return _params


def get_stock_data(dataset, experiment_features, target_label):
    for month_group in dataset.groupby(['year', 'month']):
        dataset_frame = month_group[1]
        dataset_frame.drop(columns=['year'])
        dataset_frame.drop(columns=['month'])
        pack = []
        for index, row in dataset_frame.iterrows():
            X = row[experiment_features].astype(float)
            X_tensor = torch.FloatTensor(X.to_numpy()).to(device)
            # Y is a tensor of target values (vector)
            Y = torch.FloatTensor([row[target_label].astype(float).iloc[0]]).to(device)
            pack.append((X_tensor, Y))
        yield pack, month_group[0]


def avg_eucleadan_distance(y_p, y_test):
    assert len(y_p) == len(y_test)
    return sum([abs(a - p) / a for p, a in zip(y_p, y_test)]) / len(y_p)


def evaluate(mdl, testdata, experiment_features, target_label, verbose):
    # Validation
    if verbose:
        dt = tqdm(get_stock_data(testdata, experiment_features, target_label))
    else:
        dt = get_stock_data(testdata, experiment_features, target_label)
    predictions = []
    actuals = []
    for pack, (year, month) in dt:
        # Prediction
        assert pack, "monthly data must not be empty"
        context = None
        Y = None
        pred = None
        for X, Y in pack:
            X = X.unsqueeze(0)
            pred, _, context = mdl(X, Y, context)
        actuals.append(Y.view(-1).item())
        predictions.append(pred.view(-1).item())
    if not predictions or not actuals:
        print(f"Warninig this should not happen! ticker: {ticker} null actuals and predictions in test set!")
        error = 0.0
    else:
        error = avg_eucleadan_distance(predictions, actuals)
    if verbose:
        print("Test validation error: {:.5f}".format(error))
    return error


def train(traindata, testdata, experiment_features, target_label, epochs=50, lr=0.003, learning_momentum=0.9,
          max_grad_norm=1.0, cumulate_loss=16, optimizer_name="adadelta", verbose=True):
    """
     This experiment tries to predict the values of IVOL from the data by directly looking at the raw variables.
    """
    # Instance from Predictor Class
    p = MonthlyPredictor(len(experiment_features), 32, len(target_label), 1).to(device)
    # Stochastic Gradient Descent optimizer
    if optimizer_name == "adadelta":
        optimizer = torch.optim.Adadelta(p.parameters(), lr=lr, rho=0.9, eps=1e-06, weight_decay=0.01)
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(p.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0,
                                        momentum=learning_momentum, centered=False)
    else:
        optimizer = torch.optim.SGD(p.parameters(), lr=lr, momentum=learning_momentum)
    # Temporary variables for computing the average loss
    all_loss = 0.0
    all_count = 0
    # range from 1968 to 2019
    optimizer.zero_grad()
    n_params = inline_normalize(traindata, experiment_features)
    inline_normalize(testdata, experiment_features, n_params)
    best_error = float("inf")
    best_epoch = -1
    for epoch in range(epochs):
        # Training
        context = None
        if verbose:
            dt = tqdm(get_stock_data(traindata, experiment_features, target_label))
        else:
            dt = get_stock_data(traindata, experiment_features, target_label)
        for pack, (year, month) in dt:
            # Prediction
            assert pack, "monthly data must not be empty"
            # context = None
            loss = 0
            for X, Y in pack:
                X = X.unsqueeze(0)
                _, loss, context = p(X, Y, context)
            all_loss += loss.item()
            all_count += 1
            loss.backward()
            context = (context[0].detach(), context[1].detach())
            if all_count and all_count % cumulate_loss == 0:
                nn.utils.clip_grad_norm_(p.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            if verbose:
                dt.set_description("E: {} M-Y: {}-{}\tAverage Loss: {:.9f}".format(epoch, int(year), int(month), all_loss / all_count))
        epoch_error = evaluate(p, testdata, experiment_features, target_label, verbose)
        if epoch_error < best_error:
            best_error = epoch_error
            best_epoch = epoch
    return best_error, best_epoch


def prepare_data(ticker='AAPL', number_of_history_years=0, verbose=True):
    """
    :param ticker: The ticker for which we look for the error rate
    :param number_of_history_years: can be the following numbers:
         0 : All the data
         1 : one year prior to the test year
         2 : two years prior to the test years
         3 : three years prior to the test years
         ...
    :param verbose: a flag stating whether it is required to print logs
    """
    dtype = {'SHRCD': float, 'EXCHCD': float, 'SICCD': float, 'ISSUNO': float, 'HEXCD': float, 'HSICCD': float,
             'DLAMT': float, 'DLSTCD': float, 'SHRFLG': float, 'HSICMG': float, 'HSICIG': float, 'DISTCD': float,
             'DIVAMT': float, 'FACPR': float, 'FACSHR': float, 'ACPERM': float, 'ACCOMP': float, 'NWPERM': float,
             'DLRETX': float, 'DLPRC': float, 'DLRET': float, 'TRTSCD': float, 'NMSIND': float, 'MMCNT': float,
             'NSDINX': float, 'BIDLO': float, 'ASKHI': float, 'PRC': float, 'VOL': float, 'RET': float, 'BID': float,
             'ASK': float, 'SHROUT': float, 'CFACPR': float, 'CFACSHR': float, 'OPENPRC': float, 'NUMTRD': float,
             'RETX': float, 'vwretd': float, 'vwretx': float, 'ewretd': float, 'ewretx': float, 'sprtrn': float,
             'Mkt-RF': float, 'SMB': float, 'HML': float, 'RF': float, 'IVOL': float}
    data1 = pd.read_csv(best_feature_set_file)

    features = None
    for index, row in data1.iterrows():
        if row['Ticker'] == ticker:
            features = row[' Features'].strip().split('_')
            break
    assert features is not None

    target_label = ['IVOL']
    selectedfeatures = features + target_label

    data = pd.read_csv(ticker_file_name_format.format(ticker), low_memory=False)
    data = data[data['IVOL'].notna()]
    data = data[data['PRC'] >= 0]
    data = data.fillna(0)
    for key in dtype:
        data.loc[pd.to_numeric(data[key], errors='coerce').isnull(), key] = 0
    data = data[selectedfeatures+['year', 'month']]
    data = data.astype(float)

    assert 0 <= number_of_history_years <=10

    if number_of_history_years == 0:
        traindata = data[data['year'] <= 2018]
    else:
        traindata = data[data['year'] <= 2018]
        traindata = traindata[traindata['year'] > (2018 - number_of_history_years)]

    testdata = data[data['year'] >= 2019]

    if not len(data[data['year'] < 2018]):
        traindata = data
        testdata = data

    traindata = traindata[selectedfeatures + ['year', 'month']]
    testdata = testdata[selectedfeatures + ['year', 'month']]

    return traindata, testdata, features, target_label


if __name__ == '__main__':
    all_tickers = [row['Ticker'] for index, row in pd.read_csv(best_feature_set_file).iterrows()]
    num_of_history_years = 1
    all_error_rates = []
    verbose = False
    for ticker in tqdm(all_tickers):
        if ticker == 'CEG':
            continue
        train_data, test_data, e_features, t_label = prepare_data(ticker, num_of_history_years, verbose=verbose)
        error_rate, best_epoch = train(train_data, test_data, e_features, t_label, epochs=50, lr=0.003,
                                       learning_momentum=0.9, max_grad_norm=1.0, cumulate_loss=16,
                                       optimizer_name="adadelta", verbose=verbose)
        all_error_rates.append((ticker, error_rate, best_epoch))
    print(f"Average error_rate: {sum([x[1] for x in all_error_rates])/len(all_error_rates):.3f}")
    print(f"Best error_rate: {[sorted(all_error_rates, key=lambda x: x[1])[0]][0]}")
    print(f"Worst error_rate: {[sorted(all_error_rates, key=lambda x: x[1], reverse=True)[0]][0]}")
