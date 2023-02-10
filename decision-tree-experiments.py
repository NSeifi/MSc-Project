from feature_extractors import *
from lstm_experiments import inline_normalize, prepare_data

functions = {
              'SelectFdrF': SelectFdrF,
              'SelectFprF2': SelectFprF2,
              'SelectFweF2': SelectFweF2,
              'SelectKBestF': SelectKBestF,
              'SelectKBestF2': SelectKBestF2,
              'SelectKBestF3': SelectKBestF3,
              'SelectKBestF4': SelectKBestF4,
              'SelectFromModelF2': SelectFromModelF2,
              'SelectFromModelF5': SelectFromModelF5,
              'SelectFromModelF6': SelectFromModelF6,
              'SequentialFeatureSelectorF': SequentialFeatureSelectorF,
              'SequentialFeatureSelectorF2': SequentialFeatureSelectorF2,
              'RFEF': RFEF,
              'VarianceThresholdF': VarianceThresholdF
}

models = {
    'decision_tree_regressor': train_decision_tree_regressor,
    'extra_tree_regressor': train_extra_tree_regressor,
    'gradient_boosting_regressor': train_gradient_boosting_regressor
}

best_feature_set_file = "best-feature-set.csv"
ticker_file_name_format = "company_preprocessed_csv/{}.csv"


def run_experiment(trainer, ticker='AAPL', number_of_history_years=0, verbose=True):
    train_data, test_data, e_features, t_label = prepare_data(ticker, number_of_history_years, verbose=verbose)

    n_params = inline_normalize(train_data, e_features)
    inline_normalize(test_data, e_features, n_params)

    y_train = train_data["IVOL"].astype(float)
    X_train = train_data.drop(columns=["IVOL"]).astype(float)
    y_test = test_data["IVOL"].astype(float)
    X_test = test_data.drop(columns=["IVOL"]).astype(float)
    try:
        _, error_rate, _, _ = trainer(e_features, X_train, X_test, y_train, y_test, "")
    except ValueError as e:
        print(f"Ticker: {ticker}, {e}")
        return 0.0
    return error_rate


if __name__ == '__main__':
    data = pd.read_csv(best_feature_set_file)
    num_of_history_years = 0
    all_error_rates = []
    # Possible Options:
    #           'decision_tree_regressor'
    #           'extra_tree_regressor'
    #           'gradient_boosting_regressor'
    #           None --> meaninig what is recorded as the best by the feature selector
    overwrite_model_name = None

    for index, row in tqdm(data.iterrows()):
        ticker = row['Ticker']
        if ticker in ['CEG', 'TEAM']:
            continue
        f_selector_name = row[' FeatureSelector'].strip()
        # best_tree = row[' BaselineModel'].strip()
        best_tree = 'decision_tree_regressor'
        assert best_tree in models, best_tree
        if overwrite_model_name is not None:
            best_tree = overwrite_model_name
        all_error_rates.append((ticker, run_experiment(models[best_tree], ticker, num_of_history_years, verbose=False),
                                f_selector_name))
    print(f"Average error_rate: {sum([x[1] for x in all_error_rates])/len(all_error_rates):.3f}")
    print(f"Best error_rate: {sorted(all_error_rates, key=lambda x: x[1])[0]}")
    print(f"Worst error_rate: {sorted(all_error_rates, key=lambda x: x[1], reverse=True)[0]}")
# for 2 years: ,'ABNB' , 'LCID'
# for 3 years: ,'DDOG'
# all time : 'CEG', 'TEAM'