#import mapper as mp
#from sklearn import datasets

#data, labels = datasets.make_circles(n_samples=2000, noise=0.03, factor=0.5)
#X, Y  = data[:,0], data[:,1]
#data = [[x,y] for x,y in zip(X,Y)]
#out = mp.Mapper(lens = "PCA", clusterer = "DBSCAN", n_rcover = [50, 2], clusterer_params  = (0.1,5))
#out.write_to_json(data)

import mapper as mp
#from sklearn.datasets import fetch_mldata
from sklearn.datasets import fetch_openml
data = fetch_openml('diabetes').data.tolist()
out = mp.Mapper(lens = "PCA", clusterer = "DBSCAN", n_rcover = [100, 3], clusterer_params = (0.1,5))
out.write_to_json(data)
