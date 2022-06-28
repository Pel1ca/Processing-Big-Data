import numpy as np

class OutliersClusterModel:
    
    def __init__(self, centroids, tresholds):
        self.centroids = centroids
        self.tresholds = tresholds        
        
    def train(self, features, clusters):
        #get data for cluster
        self.centroids = []
        self.tresholds = []
        for i, cluster in enumerate(clusters):
            #find center
            center = np.average(features[cluster], axis=0)
            center = center / np.linalg.norm(center)
            #compute distance
            distances = np.linalg.norm(features[cluster] - center, axis = 1)
            #find outlier treshold
            quantiles = np.quantile(distances, [0.25, 0.75])
            iqr = quantiles[1] - quantiles[0]
            toRemove = quantiles[1] + 1.5*iqr
            self.centroids.append(center)
            self.tresholds.append(toRemove)
    
    def remove(self, features, clusters):
        data = []
        outliers = np.array([])
        for i, cluster in enumerate(clusters):
            distances = np.linalg.norm(features[cluster] - self.centroids[i], axis = 1)
            toKeep = distances < self.tresholds[i]
            data.append(cluster[toKeep])
            outliers =  np.append(outliers, cluster[np.logical_not(toKeep)])
        return data, outliers