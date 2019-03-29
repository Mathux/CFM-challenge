import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# Extract features from data
def add_features(data, eps=10**-10, n_cluster=50, embeddings=None, ewma = True, aggregate = True):
    #print("mem data:", id(data))
    # Get useful columns (data from different hours)
    return_cols = [col for col in data.columns if col.endswith(':00')]

    # Some stats features
    data['return_nan'] = data.isna().sum(axis=1)
    data['avg_return_date_eqt'] = data[return_cols].mean(axis=1)
    data['var_return_date_eqt'] = data[return_cols].var(axis=1)
    data['skew_return_date_eqt'] = data[return_cols].skew(axis=1)
    data['kurt_return_date_eqt'] = data[return_cols].kurt(axis=1)
    data['max_drawdown_date_eqt'] = data[return_cols].max(
        axis=1) - data[return_cols].min(axis=1)
    #data['avg_log_vol_date_eqt'] = np.log(
    #    np.abs(data[return_cols]).mean(axis=1))
    #data['var_log_vol_date_eqt'] = np.log(
    #    np.abs(data[return_cols]).var(axis=1))

    data = group_by_date_countd(data, return_cols)
    data = group_by_product_countd(data, return_cols)

    data['09:30:00'].fillna(0, inplace=True)
    data[return_cols] = data[return_cols].interpolate(axis=1)
    
    if aggregate : 
        returns = data[return_cols]
        df_train = pd.DataFrame(
            np.add.reduceat(
                returns.values, np.arange(len(returns.columns))[::7], axis=1))
        df_train.columns = returns.columns[::7]/7
        data = data.drop(return_cols, axis=1)
        new_returns_cols = return_cols[::7]
        data[new_returns_cols] = df_train
        
    return_cols = [col for col in data.columns if col.endswith(':00')]
    if ewma :
        data[return_cols] = data[return_cols].ewm(alpha=0.2, axis=1).mean()

    data['difference_to_market'] = data[return_cols[-1]] - data[
        'avg_market_return_date']
    data['return_trend'] = np.sum(data[return_cols],axis = 1)
    #data['log_vol_difference_to_market'] = np.log(
    #    np.abs(data[return_cols[-1]] + eps)) - data['avg_market_log_vol_date']
    #data['log_vol_trend'] = np.log(np.abs(data[return_cols[-1]] + eps)) - np.log(
    #    np.abs(data[return_cols[0]] + eps))

    if embeddings:
        sectors = get_sector(embeddings, n_clusters=n_cluster)
        add_sector(data, sectors)
        data = group_by_sector(data, return_cols, sectors)
    #print(data.keys())
    #print("mem data:", id(data))
    return data


def group_by_date_countd(all_data, return_cols):
    groupby_col = "date"
    unique_products = all_data.groupby([groupby_col])["eqt_code"].nunique()
    avg_market_return = all_data.groupby(
        [groupby_col])['avg_return_date_eqt'].mean()
    var_market_return = all_data.groupby(
        [groupby_col])['var_return_date_eqt'].mean()
#    avg_log_vol_market_return = all_data.groupby(
#        [groupby_col])['avg_log_vol_date_eqt'].mean()
#    var_log_vol_market_return = all_data.groupby(
#        [groupby_col])['var_log_vol_date_eqt'].mean()
    all_data.set_index([groupby_col], inplace=True)
    all_data["countd_product"] = unique_products.astype('float64')
    all_data["avg_market_return_date"] = avg_market_return.astype('float64')
    all_data["var_market_return_date"] = var_market_return.astype('float64')
#    all_data["avg_market_log_vol_date"] = avg_log_vol_market_return.astype(
#        'float64')
#    all_data["avg_market_log_vol_date"] = var_log_vol_market_return.astype(
#        'float64')
    all_data.reset_index(inplace=True)
    return all_data


def group_by_product_countd(all_data, return_cols):
    groupby_col = "eqt_code"
    unique_date = all_data.groupby([groupby_col])["date"].nunique()
    avg_market_return = all_data.groupby(
        [groupby_col])['avg_return_date_eqt'].mean()
    var_market_return = all_data.groupby(
        [groupby_col])['var_return_date_eqt'].mean()
#    avg_log_vol_market_return = all_data.groupby(
#        [groupby_col])['avg_log_vol_date_eqt'].mean()
#    var_log_vol_market_return = all_data.groupby(
#        [groupby_col])['var_log_vol_date_eqt'].mean()
    all_data.set_index([groupby_col], inplace=True)
    all_data["countd_date"] = unique_date.astype('float64')
    all_data["avg_market_return_eqt"] = avg_market_return.astype('float64')
    all_data["var_market_return_eqt"] = var_market_return.astype('float64')
#    all_data["avg_market_log_vol_eqt"] = avg_log_vol_market_return.astype(
#        'float64')
#    all_data["avg_market_log_vol_eqt"] = var_log_vol_market_return.astype(
#        'float64')
    all_data.reset_index(inplace=True)
    return all_data


def plot_corr(df, size=10):
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(df.corr(), annot=True)


def get_data_matrix(embeddings):
    n_samples, n_features = len(embeddings), len(list(embeddings.values())[0])
    X = np.zeros((n_samples, n_features))
    for i, eqt in enumerate(embeddings.keys()):
        X[i, :] = embeddings[eqt]
    return X


def get_sector(embeddings, n_clusters=8):
    X = get_data_matrix(embeddings)
    clf = KMeans(n_clusters=n_clusters, n_init=30)
    sectors_temp = clf.fit_predict(X)
    sectors = {}
    for j, eqt in enumerate(embeddings.keys()):
        sectors[eqt] = sectors_temp[j]
    return sectors


def add_sector(data, sectors):
    data['sector'] = data['eqt_code'].map(sectors)


def group_by_sector(all_data, return_cols, sector):
    groupby_col = "sector"
    unique_products = all_data.groupby([groupby_col])["eqt_code"].nunique()
    avg_market_return = all_data.groupby(
        [groupby_col])['avg_return_date_eqt'].mean()
    var_market_return = all_data.groupby(
        [groupby_col])['var_return_date_eqt'].mean()
    avg_log_vol_market_return = all_data.groupby(
        [groupby_col])['avg_log_vol_date_eqt'].mean()
    var_log_vol_market_return = all_data.groupby(
        [groupby_col])['var_log_vol_date_eqt'].mean()
    all_data.set_index([groupby_col], inplace=True)
    all_data["countd_product"] = unique_products.astype('float64')
    all_data["avg_market_return_date"] = avg_market_return.astype('float64')
    all_data["var_market_return_date"] = var_market_return.astype('float64')
    all_data["avg_market_log_vol_date"] = avg_log_vol_market_return.astype(
        'float64')
    all_data["avg_market_log_vol_date"] = var_log_vol_market_return.astype(
        'float64')
    all_data.reset_index(inplace=True)
    return all_data
