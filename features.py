import matplotlib.pyplot as plt
import pandas as pd
import utils


# Extract features from data
def add_features(data):
    # Get usefull columns (data from different hours)
    return_cols = [col for col in data.columns if col.endswith(':00')]

    # Some stats features
    data['return_nan'] = data.isna().sum(axis = 1)    
    data['avg_return_date_eqt'] = data[return_cols].mean(axis = 1)
    data['var_return_date_eqt'] = data[return_cols].var(axis = 1)
    data['skew_return_date_eqt'] = data[return_cols].skew(axis = 1)
    data['kurt_return_date_eqt'] = data[return_cols].kurt(axis = 1)
    
    data = group_by_date_countd(data,return_cols)
    data = group_by_product_countd(data,return_cols)
    data['tot_return_eqt_date'] = data[return_cols].sum(axis = 1)


    # Creation of the market feature
    stock_correlation_data = market_correlation(data)
    stock_correlation_data = stock_correlation_data.replace(to_replace = 1.0, value = 0)
    
    temp_data = (data.groupby('eqt_code')['tot_return_eqt_date'].mean()*stock_correlation_data/stock_correlation_data.sum(axis = 1)).sum()
    
    data.set_index(['eqt_code'],inplace = True)
    data['market_feature'] = temp_data
    data.reset_index(inplace = True)


def group_by_date_countd(all_data,return_cols):
    groupby_col = "date"
    unique_products = all_data.groupby([groupby_col])["eqt_code"].nunique()
    avg_market_return = all_data.groupby([groupby_col])['avg_return_date_eqt'].mean()
    var_market_return = all_data.groupby([groupby_col])['var_return_date_eqt'].mean()
    all_data.set_index([groupby_col], inplace=True)
    all_data["countd_product"] = unique_products.astype('uint16')
    all_data["avg_market_return_date"] = avg_market_return.astype('float64')
    all_data["var_marlet_return_date"] = var_market_return.astype('float64')
    all_data.reset_index(inplace=True)
    return all_data


def group_by_product_countd(all_data,return_cols):
    groupby_col = "eqt_code"
    unique_date = all_data.groupby([groupby_col])["date"].nunique()
    avg_market_return = all_data.groupby([groupby_col])['avg_return_date_eqt'].mean()
    var_market_return = all_data.groupby([groupby_col])['var_return_date_eqt'].mean()
    all_data.set_index([groupby_col], inplace=True)
    all_data["countd_date"] = unique_date.astype('uint16')
    all_data["avg_market_return_eqt"] = avg_market_return.astype('float64')
    all_data["var_market_return_eqt"] = var_market_return.astype('float64')
    all_data.reset_index(inplace=True)
    return all_data


def market_correlation(data):
    df = pd.pivot_table(data[['date','eqt_code','tot_return_eqt_date']], values='tot_return_eqt_date', index=['date'],columns=['eqt_code'])
    corr = df.corr()
    corr = corr.fillna(0)
    return corr


def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);


if __name__ == '__main__':
    # Load data
    X_train,X_test,y_train,y_train_labels = utils.load_data()

    # Add features (inplace)
    add_features(X_train)
    add_features(X_test)

