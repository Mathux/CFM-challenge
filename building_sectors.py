#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 23:16:39 2019

@author: evrardgarcelon
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from processing_data import *
from sklearn import cluster, covariance, manifold
from sklearn.utils import resample
import config

def load_data(data, maxvalue=1511):

    return_cols = ['open close diff','ID']
    eqt_codes = data["eqt_code"].unique()
    n = len(eqt_codes)
    processed_vector = np.zeros((n, maxvalue, len(return_cols) - 1),dtype='float64')
    
    for i in range(n):
        
        vector_eqt = data[data["eqt_code"] == eqt_codes[i]][return_cols].values
        ndate = vector_eqt.shape[0]
        more_vector = resample(vector_eqt,n_samples=maxvalue - ndate,random_state=config.SEED)
        vector = np.concatenate((vector_eqt, more_vector))
        np.random.shuffle(vector)
        final_vector = vector[:, :-1]  
        processed_vector[i] = final_vector

    return processed_vector

data = Data(split = False,small = False, verbose = True, ewma = False)
return_cols = [c for c in data.x.columns if c.endswith(':00')]
data.x['open close diff'] = data.x[return_cols].sum(axis = 1)
X = load_data(data.x).squeeze().T

edge_model = covariance.GraphicalLassoCV(cv=5)
X /= X.std(axis=0)
edge_model.fit(X)

_, labels = cluster.affinity_propagation(edge_model.covariance_)
n_labels = labels.max()

#%%

symbol_dict = {}
for i,eqt in enumerate(data.x['eqt_code'].unique()) :
    symbol_dict[i] = str(eqt)

symbols, names = np.array(sorted(symbol_dict.items())).T
for i in range(n_labels + 1):
    print('Cluster %i: %s' % ((i + 1), ', '.join(names[labels == i])))

node_position_model = manifold.LocallyLinearEmbedding(
    n_components=2, eigen_solver='dense', n_neighbors=2)

embedding = node_position_model.fit_transform(X.T).T

plt.figure(1, facecolor='w', figsize=(10, 8))
plt.clf()
ax = plt.axes([0., 0., 1., 1.])
plt.axis('off')

# Display a graph of the partial correlations
partial_correlations = edge_model.precision_.copy()
d = 1 / np.sqrt(np.diag(partial_correlations))
partial_correlations *= d
partial_correlations *= d[:, np.newaxis]
non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

# Plot the nodes using the coordinates of our embedding
plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,
            cmap=plt.cm.nipy_spectral)

# Plot the edges
start_idx, end_idx = np.where(non_zero)
# a sequence of (*line0*, *line1*, *line2*), where::
#            linen = (x0, y0), (x1, y1), ... (xm, ym)
segments = [[embedding[:, start], embedding[:, stop]]
            for start, stop in zip(start_idx, end_idx)]
values = np.abs(partial_correlations[non_zero])
lc = LineCollection(segments,
                    zorder=0, cmap=plt.cm.hot_r,
                    norm=plt.Normalize(0, .7 * values.max()))
lc.set_array(values)
lc.set_linewidths(15 * values)
ax.add_collection(lc)

# Add a label to each node. The challenge here is that we want to
# position the labels to avoid overlap with other labels
for index, (name, label, (x, y)) in enumerate(
        zip(names, labels, embedding.T)):

    dx = x - embedding[0]
    dx[index] = 1
    dy = y - embedding[1]
    dy[index] = 1
    this_dx = dx[np.argmin(np.abs(dy))]
    this_dy = dy[np.argmin(np.abs(dx))]
    if this_dx > 0:
        horizontalalignment = 'left'
        x = x + .002
    else:
        horizontalalignment = 'right'
        x = x - .002
    if this_dy > 0:
        verticalalignment = 'bottom'
        y = y + .002
    else:
        verticalalignment = 'top'
        y = y - .002
    plt.text(x, y, name, size=10,
             horizontalalignment=horizontalalignment,
             verticalalignment=verticalalignment,
             bbox=dict(facecolor='w',
                       edgecolor=plt.cm.nipy_spectral(label / float(n_labels)),
                       alpha=.6))

plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
         embedding[0].max() + .10 * embedding[0].ptp(),)
plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
         embedding[1].max() + .03 * embedding[1].ptp())

plt.show()






