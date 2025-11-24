# %%
import os
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import plotly.graph_objects as go
from plotly.graph_objs.scatter.marker import Line
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import math

import warnings

warnings.filterwarnings('ignore')


# %%
marketDataNQ = '/Users/damensavvasavvi/Desktop/AlgoTrading/marketDataNasdaqFutures/NQ_continuous_backadjusted_1m_cleaned.csv'
prices = pd.read_csv(marketDataNQ)

prices = prices.set_index('timestamp')
prices = prices[prices.index < '2024-01-01']


# %%
# px.line(prices['close'], title='NQ 1 Min Close Prices').show()

# %%
def MA_log_returns(prices, window):
    """
    Calculate moving average and log returns of the prices.

    Parameters:
    prices (pd.Series): Series of prices.
    window (int): Window size for moving average.

    Returns:
    tuple: Moving average and log returns as pd.Series.
    """
    moving_avg = prices.rolling(window=window).mean()
    moving_avg[:window] = [moving_avg[window]]*window
    returns = prices/prices.shift(1)
    returns.where(abs(returns) != np.inf, 0, inplace=True)
    returns[0] = returns[1]
    log_returns = np.log(returns)
    print(f'number of NaN in log returns: {log_returns.isna().sum()}')
    print(f'number of inf in log returns: {np.isinf(abs(log_returns)).sum()}')
    print(f'number of NaN in moving average: {moving_avg.isna().sum()}')
    print(f'number of inf in moving average: {np.isinf(abs(moving_avg)).sum()}')
    return moving_avg, log_returns

moving_avg, log_returns = MA_log_returns(prices['close'], window=7)
prices['moving_avg'] = moving_avg
prices['log_returns'] = log_returns
prices = prices[['close', 'moving_avg', 'log_returns']]
prices



# %%
class RegimeDetection:
    def get_regimes_hmm(self, input_data, params):
        hmm_model = self.initialise_model(GaussianHMM(), params).fit(input_data)
        return hmm_model

    def get_regimes_clustering(self, params):
        clustering =  self.initialise_model(AgglomerativeClustering(), params)
        return clustering

    def get_regimes_gmm(self, input_data, params):
        gmm = self.initialise_model(GaussianMixture(), params).fit(input_data)
        return gmm

    def initialise_model(self, model, params):
        for parameter, value in params.items():
            setattr(model, parameter, value)
        return model

# %%
def plot_hidden_states(hidden_states, prices_df):
    
    '''
    Input:
    hidden_states(numpy.ndarray) - array of predicted hidden states
    prices_df(df) - dataframe of close prices
    
    Output:
    Graph showing hidden states and prices
    
    '''
    
    colors = ['blue', 'red']
    n_components = len(np.unique(hidden_states))
    
    fig = go.Figure()
 
    for i in range(n_components):
        mask = hidden_states == i
        print('Number of observations for State ', i,":", len(prices_df.index[mask]))

        fig.add_trace(go.Scatter(x=prices_df.index[mask], y=prices_df[mask],
                    mode='markers',  name='Hidden State ' + str(i), marker=dict(size=4,color=colors[i])))
        
    fig.update_layout(height=400, width=900, legend=dict(
            yanchor="top", y=0.99, xanchor="left",x=0.01), margin=dict(l=20, r=20, t=20, b=20)).show()

# %%
regime_detection = RegimeDetection()

# %%
# params = {'n_clusters': 2, 'linkage': 'complete', 'affinity': 'manhattan', 'metric': 'manhattan', 'random_state':100}

# clustering = regime_detection.get_regimes_clustering(params)

# clustering_states = clustering.fit_predict(prices['log_returns'].array.reshape(-1, 1))

# hidden_states = clustering_states
# n_components = len(np.unique(np.array(clustering_states)))

# for i in range(n_components):
#     mask = hidden_states == i
#     print('Number of observations for State ', i,":", len(prices['close'].index[mask]))


params = {'n_components':2, 'covariance_type': 'full', 'max_iter': 100000, 'n_init': 30,'init_params': 'kmeans', 'random_state':100}


gmm_model = regime_detection.get_regimes_gmm(prices['log_returns'].array.reshape(-1,1), params)

gmm_states = gmm_model.predict(prices['log_returns'].array.reshape(-1,1))

hidden_states = gmm_states
n_components = len(np.unique(np.array(hidden_states)))

 
for i in range(n_components):
    mask = hidden_states == i
    print('Number of observations for State ', i,":", len(prices['close'].index[mask]))





