import numpy as np
import pickle
import scipy.io
import scipy.stats
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from os.path import expanduser

home = expanduser("~")

import h2o
from h2o.estimators import H2OGeneralizedLowRankEstimator

import sys

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
            distances = np.linalg.norm(features[cluster] - center, axis = 1, ord = 1)
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
            distances = np.linalg.norm(features[cluster] - self.centroids[i], axis = 1, ord = 1)
            toKeep = distances < self.tresholds[i]
            data.append(cluster[toKeep])
            outliers =  np.append(outliers, cluster[np.logical_not(toKeep)])
        return data, outliers

def getFeaturesClusters(features):
    featuresClassifier = pickle.load(open(home + '/dataAnalysis/Models/featuresClassifier.md', 'rb'))
    outlierModel = pickle.load(open(home + '/dataAnalysis/Models/outlierModelFeatures.md', 'rb'))
    
    labels = featuresClassifier.predict(features)
    indexes = [np.array([i for i, y in enumerate(labels) if  y == cluster]) for cluster in range(len(set(labels)))]
    cleanIndexes, outliers = outlierModel.remove(features, indexes)
    return cleanIndexes, outliers

#To remove the probabilities and frame information from the skeletons
def extractSkelCoord(skeletons):
    probabilities = np.full(skeletons.shape[1], True)
    probabilities[::3] = False
    skelsCords = skeletons[:,probabilities]
    return skelsCords

def centerSkels(skels):
    noFirstPoint = np.where(np.logical_not(np.isnan(skels[:,2])))
    centerData = skels[noFirstPoint].copy()
    centerData[:,::2] -= centerData[:,2].reshape(-1, 1)
    centerData[:,1::2] -= centerData[:,3].reshape(-1, 1)
    return centerData, noFirstPoint

def completeMissingSkeletonData(skeletonMiss):
    #load models
    glrm_model = h2o.load_model(home + '/dataAnalysis/Models/GLRM_model_python_1655889296079_9')
    #get coordinates info
    skelsCords = extractSkelCoord(skeletonMiss)
    skelsCords[np.where(skelsCords == 0)] = np.nan
    #center data
    skelsCenter, removed = centerSkels(skelsCords)
    remainSkels = skeletonMiss[removed]
    #remove corrupt data
    usableIndexes = np.where(np.bincount(np.where(np.isnan(skelsCenter))[0]) < 20)
    remainSkels = remainSkels[usableIndexes]
    skelsCenter = skelsCenter[usableIndexes]
    #predict missing data
    skelsH2o = h2o.H2OFrame(skelsCenter, column_types = ['real' for i in range(36)])
    D=glrm_model.predict(skelsH2o).as_data_frame(use_pandas=True).to_numpy()
    D = np.hstack((D[:,:2], np.zeros((D.shape[0], 2)), D[:,2:]))
    skelsCenter[np.isnan(skelsCenter)] = D[np.isnan(skelsCenter)]
    remainSkels[:,1::3] = skelsCenter[:,::2] #x
    remainSkels[:,2::3] = skelsCenter[:,1::2] #y
    return remainSkels


def getPhiPolar(cords):
    return np.arctan2(cords[:,1::2], cords[:,::2])

def classifyImages(skels, frames):
    imagesClassifier =  pickle.load(open(home + '/dataAnalysis/Models/imagesClassifier.md', 'rb'))
    outlierSkelModel = pickle.load(open(home + '/dataAnalysis/Models/outlierModelSkel.md', 'rb'))
    
    skeletons = extractSkelCoord(skels)
    phi = getPhiPolar(skeletons)
    labels = imagesClassifier.predict(phi)
    indexes = [np.array([i for i, y in enumerate(labels) if  y == cluster]) for cluster in range(len(set(labels)))]
    cleanIndexes, outliers = outlierSkelModel.remove(skeletons, indexes)
    skelClusterIndexes = [cleanIndexes[1],cleanIndexes[3], np.append(cleanIndexes[0], cleanIndexes[2]) ]

    skelsClassification = np.full(skels.shape[0],-1)
    #Init frames array for the classification
    skelsClassification[skelClusterIndexes[0].astype(int)] = 0 #"front"
    skelsClassification[skelClusterIndexes[1].astype(int)] = 1 #"back"
    skelsClassification[skelClusterIndexes[2].astype(int)] = 2 #"lateral"

    for i in range(frames.shape[0]):#range(nbrImages):
        #Get all skeletons from the frame
        skelsInFrameIDs = np.where(skels[:,0] == i)
        skelis = skelsClassification[skelsInFrameIDs]
        valid = np.where(skelis!=-1)
        if valid[0].size != 0:
            frames[i,1] = np.bincount(skelis[valid[0]]).argmax()
            
    return frames

def getDataInCluster(skeletons, clusterIndex):
    cl = pd.DataFrame(clusterIndex)
    skel = pd.DataFrame(skeletons)
    dataInCluster = skel.merge(right=cl, left_on=0, right_on=0, how="inner")
    return dataInCluster.to_numpy()

def classifyVideo(features, skeletons, smoothWind = 7):
    clusters, outliers = getFeaturesClusters(normalize(features))
    imagesWithSkels = np.append(clusters[0], clusters[2])
    #Classify Base on Embedings
    frames = np.full((features.shape[0],2),-1)
    frames[clusters[1].astype(int),0] = 0 #"Aerial view"
    frames[clusters[0].astype(int),0] = 1 #"Individuals view"
    frames[clusters[2].astype(int),0] = 2 #"Peloton view"

    completeSkels = completeMissingSkeletonData(getDataInCluster(skeletons, imagesWithSkels))
    frames = classifyImages(completeSkels, frames)
    
    map1 = ["Aerial view", "Close view", "Peloton view", ""]
    map2 = ["Front ", "Back ", "Lateral ", ""]
    imagesClassification = [map2[frame[1]] + map1[frame[0]] for frame in frames]
    
    if smoothWind > 0:
        most_freq_val = lambda x: scipy.stats.mode(x)[0][0]
        imagesClassification = [most_freq_val(imagesClassification[i:i+smoothWind]) for i in range(0,len(imagesClassification)-smoothWind+1)]
    return imagesClassification
#features_path = '../datasets/EurosportCut/girosmallslow_cut.mp4_features.mat'
#skel_path = '../datasets/EurosportCut/esqueletosmallslow_cut.mat'
def main():
    
    if len(sys.argv) != 3:
        print("Invalid number of arguments. Expecting <features path> <skel path>")
        sys.exit(-1)
    h2o.init('http://localhost:54322')
    features_path = sys.argv[1]
    skel_path = sys.argv[2]
    
    mat1 = scipy.io.loadmat(skel_path)
    mat3 = scipy.io.loadmat(features_path)

    imgage_classification = classifyVideo(mat3['features'].T, mat1["skeldata"].T, 10)
    pickle.dump(imgage_classification, open(home + '/datasets/Classifications/videoClass.cl', 'wb'))

if __name__ == '__main__':
    main()