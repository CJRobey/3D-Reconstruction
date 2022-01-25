from __future__ import generators
#!/usr/bin/env python
# coding: utf-8
import laspy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import operator
from sklearn.cluster import SpectralClustering
from scipy.signal import argrelextrema
from scipy.spatial.distance import euclidean, pdist, squareform

        
def loadLASData(fpath):
    """fpath is the filepath to the .las file
    returns the file object"""
    inFile = laspy.file.File(fpath) 
    pointformat = inFile.point_format
    print('Attributes of the las file:')
    for spec in inFile.point_format:
        print(spec.name)
    return inFile


def extractXYZColors(inFile, colorSize=2**16):
    """
    Inputs:
    inFile              = .las file object rendered from laspy
    colorSize           = the size of the colors from the .las file - commonly 16 bit integer
    
    Returns:
    colors              = an Nx3 array of the colors (r,g,b format)
    dataset             = a normalized Nx3 array of the 3d points (x,y,z format)
    """
    colors = np.vstack([inFile.red/colorSize, inFile.green/colorSize,
                        inFile.blue/colorSize]).transpose()
    dataset = np.vstack([inFile.x, inFile.y, inFile.z]).transpose()
    dataset /= dataset.max()
    return colors, dataset



class plotParameters():
    
    def __init__(self):
        self.title = 'LAS Scatterplot'
        self.xlabel = 'X Axis'
        self.ylabel = 'Y Axis'
        self.zlabel = 'Z Axis'
        self.cmap = None
        self.colors = None
        self.axisView = None

def plotLASData(xyz, params):
    """
    Inputs:
    xyz                = Nx3 array with the x, y, and, z coordinates in their respective columns
                         recommended that you normalize the data. (xyz/xyz.max() should work fine)
    params             = The object that contains the following fields:
                            1. title : plot title
                            2. xlabel : label for x axis
                            3. ylabel : label for y axis
                            4. zlabel : label for z axis
                            5. colors : an Nx3 array of the colors (r,g,b format) normalized between 0 and 1.
                            6. axisView : choosethe axis that the plot will be viewed from

    """
    plt.rcParams.update({'font.size': 80})
    fig = plt.figure(figsize=[100, 50])
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(params.xlabel, labelpad=100)
    ax.set_ylabel(params.ylabel, labelpad=100)
    ax.set_zlabel(params.zlabel, labelpad=100)
    ax.set_title(params.title)
    # The final parameter is 1-xyz[:,2] because I have noticed that many .las formats
    # have the lowest z value as the closest to the camera. From arial photos, this
    # is the opposite of what we would wants
    
    # If no colors are provided, make all of the points black
    if params.colors is None:
        params.colors = np.zeros(xyz.shape)
    
    # Use the cmap if it is provided
    if params.cmap is None:
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=params.colors, marker=".")
    else:
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=params.colors, cmap=params.cmap, marker=".")
        
        
    # change the angle of the view
    # for reference -> ax.view_init(elevation_angle, rotation_angle)
    if params.axisView == 'x':
        ax.view_init(30, 90)
    if params.axisView == 'y':
        ax.view_init(30,0)
    if params.axisView == 'z':
        ax.view_init(65, 45)
    if params.axisView == None:
        ax.view_init(30, 45)
    plt.draw()
    return fig, ax



# Not the preferred method of background extraction - just cuts on a plane
def zThresholdBackgroundExtract(dataset, cols, zval):
    """
    Inputs:
    dataset             = an Nx3 array of the 3d points (x,y,z format)
    
    Returns:
    thrDataset          = an Mx3 array of 3d points that's been reduced by a z direction plane
    thrCols
    """
    idxes = np.where(dataset[:,2] > zval)[0]
    print('Indexes of found numbers:', idxes, 
          f'\nPercent reduction: {(1-(len(idxes)/len(dataset[:,0])))*100}%')
    thrDataset = dataset[idxes]
    thrCols = cols[idxes]
    return thrDataset, thrCols



class bgExtractParameters():
    
    def __init__(self):
        self.hueBounds = [60,160] # some fairly liberal bounds for the color green in the HSV space,
        # narrow the bounds to [75,145] if too many yellows and teals are being picked up
        self.satLow = 0.10 # low bound of saturation (you always want to go as high as possible)
        self.valLow = 0.13 # low bound of the value (brightness) for detecting green
        self.showOutput = True
        

def extractBGColor3D(dataset, colors, params=None):
    """extractBGColor
    Inputs:
    dataset             = an Nx3 array of the 3d points (x,y,z format)
    colors              = an Nx3 array of the colors (r,g,b format) 
                          normalized between 0 and 1. N must also be an
                          even number. Randomly removing 1 pixel should 
                          not be a big deal
    
    params              = The object that contains the following fields:
                            1. hueBounds : upper and lower bounds of the color to be extracted
                            2. satLow : low bound of saturation. You always want to go as high as possible,
                            but some low values are essentially just gray.
                            3. valLow : low bound of the value (brightness) for detecting the desired color
                            4. showOutput : allows brief print statement
    
    Returns:
    hsvBGColors         = Nx3 array of the new colors with same size as colors if colors has even number of 
                          pixels. Else, it N is one less than the original input size.
    hsvBGDataset        = Nx3 array of the new x,y, and z points with same size as colors if colors has even
                          number of pixels. Else, it N is one less than the original input size.
    finalIdx            = final indices of the nonbackground pixels relative to the original dataset input
    
    """

    if len(colors)%2 != 0:
        print(f"Deleting final point in colors because N must by even\nCurrent N value is {colors.shape[0]}")
        colors = colors[:-1]
    
    if params is None:
        params = bgExtractParameters()
    
    # reshape the colors so that they are able to be processed by opencv
    rs_cols = np.reshape(colors, (int(colors.shape[0]/2), 2, 3))
    
    # do a conversion from rgb to hsv
    hsvDataset = cv2.cvtColor(rs_cols.astype(np.float32), cv2.COLOR_RGB2HSV)
    # reshape the data back to the original shape
    hsvDataset = np.reshape(hsvDataset, colors.shape)
    hueBounds = params.hueBounds
    satLow = params.satLow
    valLow = params.valLow
    
    # grab the indices at each of the specified bounds
    bottomIdxHue = np.where(hsvDataset[:,0] > hueBounds[0])
    topIdxHue = np.where(hsvDataset[:,0] < hueBounds[1])
    idxSat = np.where(hsvDataset[:,1] > satLow)
    idxVal = np.where(hsvDataset[:,2] > valLow)
    
    # find the intersection of all of these indices
    finalIdx1 = np.intersect1d(bottomIdxHue, topIdxHue)
    finalIdx2 = np.intersect1d(idxSat, idxVal)
    finalIdx = np.intersect1d(finalIdx1, finalIdx2)
    
    if params.showOutput == True:
        print(f'There are {len(finalIdx)} points remaining after background subtraction by color' 
              f'\nReduced to {len(finalIdx)/len(dataset)*100}% of the original data')
        attrs = vars(params)
        print('\n***Parameter values***')
        print (', '.join("%s: %s" % item for item in attrs.items()))
    
    hsvBGColors = colors[finalIdx]
    hsvBGDataset = dataset[finalIdx]
    return hsvBGColors, hsvBGDataset, finalIdx



class trimLowDataParams():
    def __init__(self):
        self.thresh = 25 
        self.pctWindow = 5

        
def trimLowData(hsvData, hsvCols, params=None):
    """
    Inputs:
    hsvData            = Nx3 array with the x, y, and, z coordinates of the background-subtracted
                         data. This should be mostly green data, but some of the data may not be 
                         plants.
    colors             = Nx3 array of the background-subtracted colors (r,g,b format) normalized 
                         between 0 and 1. 
    params             = The object that contains the following fields:
                            1. thresh : the lower threshold percentage of z-values that serves as 
                                        the cutoff value for the method
                            2. pctWindow : determines how large the window used for this method is
                                           in terms of percentage of the y-axis
    
    Returns:
    newData            = Nx3 array in same format as hsvData, but without the trimmed points
    newCols            = Nx3 array of colors corresponding to newData
    
    """
    # get param values
    if params == None:
        params = trimLowDataParams()
    pctWindow = params.pctWindow
    thresh = params.thresh
    dataLen = len(hsvData[:, 0])
    
    # get the "cut point" at which to delete data
    cutPoint = np.percentile(hsvData[:,2], thresh)
    # just to keep format of arguments the same
    pctWindow = float(pctWindow/100)
    windowSize = int(dataLen*pctWindow)
    
    # create this super dope list that has the original index in the first item and the 
    # y-value is in the second col
    sortedValDict = np.array(sorted(enumerate(hsvData[:,1]), key=operator.itemgetter(1)))
    delIdx = []
    finIdx = windowSize
    
    # loop across windows of data (y-axis ref) and take the max. If it is smaller than the
    # z cutoff value, then delete the data
    while(finIdx < dataLen):
        indices = sortedValDict[(finIdx-windowSize):finIdx, 0].astype(int)
        if (hsvData[indices,2].max() < cutPoint):
            delIdx += [indices]
        finIdx += int(0.1*windowSize)
    
    newData = np.delete(hsvData, delIdx, 0)
    newCols = np.delete(hsvCols, delIdx, 0)
    return newData, newCols



class heatMapParams():

    def __init__(self):
        self.blockSize=50
        self.scaleFactor=10000

        
def makeHeatMap(data, params):
    """
    Inputs:
    data               = Nx3 array with the x, y, and, z coordinates of the background-subtracted
                         data. This should be mostly green data, but some of the data may not be 
                         plants.
    
    params             = The object that contains the following fields:
                            1. blockSize   : the size of the squares that will be used to create the
                                             height heat map. The smaller, the more granular the 
                                             detail, but the longer the processing time.
                            2. scaleFactor : Corresponds to the number of decimal points that you
                                             would like to recover before doing calculations
    Returns:
    None - the heatmap is plotted within the function
    
    """
    
    blockSize = params.blockSize
    scaleFactor = params.scaleFactor
    
    # define the heatmap data with integers for locations
    hmData = (data*scaleFactor).astype(int)
    # Make zero the min
    shiftMin = hmData.min()
    hmData -= shiftMin
    
    # establish 3 arrays that will be used for the visualization
    meanBlkHeatMap = np.zeros((hmData[:,0].max()+(blockSize - hmData[:,0].max()%blockSize), hmData[:,1].max()
                           +(blockSize-hmData[:,1].max()%blockSize)))
    maxBlkHeatMap = np.zeros((hmData[:,0].max()+(blockSize - hmData[:,0].max()%blockSize), hmData[:,1].max()
                           +(blockSize-hmData[:,1].max()%blockSize)))
    blkHeatMap = np.zeros((hmData[:,0].max()+(blockSize - hmData[:,0].max()%blockSize), hmData[:,1].max()
                           +(blockSize-hmData[:,1].max()%blockSize)))
    
    print(f'Making a heatmap of shape {blkHeatMap.shape}')
    blkHeatMap[hmData[:,0], hmData[:,1]] = hmData[:,2]#/scaleFactor
    for yBlock in range(0, blkHeatMap.shape[1], blockSize):
        for xBlock in range(0, blkHeatMap.shape[0], blockSize):
            meanBlkHeatMap[xBlock:xBlock+blockSize, yBlock:yBlock+blockSize] = blkHeatMap[xBlock:xBlock+blockSize, yBlock:yBlock+blockSize].mean()
            maxBlkHeatMap[xBlock:xBlock+blockSize, yBlock:yBlock+blockSize] = blkHeatMap[xBlock:xBlock+blockSize, yBlock:yBlock+blockSize].max()

    # change into numpy masked arrays for visualization purposes
    meanMasked = np.ma.masked_where(meanBlkHeatMap<=0.001, meanBlkHeatMap)
    maxMasked = np.ma.masked_where(maxBlkHeatMap<=0.001, maxBlkHeatMap)
    
    # get the images back to their original scales
    meanImg = meanMasked.T/scaleFactor
    maxImg = maxMasked.T/scaleFactor
    imgVmax = max(meanImg.max(), maxImg.max())
    imgVmax = None
    # make the figure and axes larger
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(50,50))
    axes[0].set_title('Mean Z-Axis Heat Map', pad=20)
    axes[1].set_title('Max Z-Axis Heat Map', pad=20)
    
    # change the font size
    for i in range(0,len(axes)):
        for item in ([axes[i].title, axes[i].xaxis.label, axes[i].yaxis.label] +
                 axes[i].get_xticklabels() + axes[i].get_yticklabels()):
            item.set_fontsize(45)
    
    # remove the x and y ticks
    axes[0].set_xticks([], [])
    axes[0].set_yticks([], [])
    axes[1].set_xticks([], [])
    axes[1].set_yticks([], [])
    
    # plot
    im0 = axes[0].imshow(meanImg, cmap='plasma')
    im1 = axes[1].imshow(maxImg, cmap='plasma')
    fig.colorbar(im0, ax=axes[0], shrink=0.5)
    fig.colorbar(im1, ax=axes[1], shrink=0.5)
    plt.show()


# Priority dictionary using binary heaps
# David Eppstein, UC Irvine, 8 Mar 2002
# http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/117228

# this class will just be used for the "Graph" class below

class priorityDictionary(dict):
    def __init__(self):
        '''Initialize priorityDictionary by creating binary heap
        of pairs (value,key).  Note that changing or removing a dict entry will
        not remove the old pair from the heap until it is found by smallest() or
        until the heap is rebuilt.'''
        self.__heap = []
        dict.__init__(self)

    def smallest(self):
        '''Find smallest item after removing deleted items from heap.'''
        if len(self) == 0:
            raise (IndexError, "smallest of empty priorityDictionary")
        heap = self.__heap
        while heap[0][1] not in self or self[heap[0][1]] != heap[0][0]:
            lastItem = heap.pop()
            insertionPoint = 0
            while 1:
                smallChild = 2*insertionPoint+1
                if smallChild+1 < len(heap) and \
                        heap[smallChild] > heap[smallChild+1]:
                    smallChild += 1
                if smallChild >= len(heap) or lastItem <= heap[smallChild]:
                    heap[insertionPoint] = lastItem
                    break
                heap[insertionPoint] = heap[smallChild]
                insertionPoint = smallChild
        return heap[0][1]

    def __iter__(self):
        '''Create destructive sorted iterator of priorityDictionary.'''
        def iterfn():
            while len(self) > 0:
                x = self.smallest()
                yield x
                del self[x]
        return iterfn()

    def __setitem__(self,key,val):
        '''Change value stored in dictionary and add corresponding
        pair to heap.  Rebuilds the heap if the number of deleted items grows
        too large, to avoid memory leakage.'''
        dict.__setitem__(self,key,val)
        heap = self.__heap
        if len(heap) > 2 * len(self):
            self.__heap = [(v,k) for k,v in self.items()]
            self.__heap.sort()  # builtin sort likely faster than O(n) heapify
        else:
            newPair = (val,key)
            insertionPoint = len(heap)
            heap.append(None)
            while insertionPoint > 0 and \
                    newPair < heap[(insertionPoint-1)//2]:
                heap[insertionPoint] = heap[(insertionPoint-1)//2]
                insertionPoint = (insertionPoint-1)//2
            heap[insertionPoint] = newPair

    def setdefault(self,key,val):
        '''Reimplement setdefault to call our customized __setitem__.'''
        if key not in self:
            self[key] = val
        return self[key]



class Graph():
    """
    Members:
    
    numObj            = the number of objects expected to be segmented in the 3D
                        data (e.g. if there are 4 plants in an image, numObj = 4)
    eps               = the maximum euclidean distance allowable for a connection 
                        to be made in the fully connected graph
    midPoints         = the coordinates of the mid points of each object in the
                        data
    FCG               = the fully connected graph of the input 3D data
    midIx             = the corresponding indices to the midpoints of the objects
    D                 = the geodesic distances of every point in G from a given 
                        startpoint
    """
  
    def __init__(self, numObj=1, eps=None, midPoints=None):
        if numObj is None:
            self.numObj = 1
        else:
            self.numObj = numObj
            
        if eps is None:
            self.eps = 0.005
        else:
            self.eps = eps
        
        if midPoints is None:
            self.midPoints = [0,0,0]*numObj
        else:
            self.midPoints = midPoints
        self.FCG = {}
        self.midIdx = 0
        self.D = {}	# dictionary of final distances


    def findMidPoint(self, data):
        xObj = np.histogram(data[:,0], bins=10)
        yObj = np.histogram(data[:,1], bins=10)
        xPeaks = argrelextrema(xObj[0], np.greater)
        yPeaks = argrelextrema(yObj[0], np.greater)
        
        ##TODO - NOT COMPLETE ACCOUNTING FOR MULTIPLE PEAKS
        if (self.numObj > 1) and (len(yPeaks) > self.numObj):
            # find the two closest peaks and "merge" them
            peakDiffs = squareform(pdist(xObj[1][yPeaks]))
            
        
        self.midPoints = [xObj[1][xPeaks].mean(), yObj[1][yPeaks].mean(), data[:,2].mean()]
        data[0] = self.midPoints
        dists = squareform(pdist(data, 'euclidean'))
        self.midIdx = np.argmin(dists[0, 1:])
        print('Midpoint(s) established at index', self.midIdx)
    
    
    def plotMidPointHists(self, data):
        plt.rcParams.update({'font.size': 10})
        xObj = plt.hist(data[:,0], bins=10, alpha=0.7, label='X Axis')
        yObj = plt.hist(data[:,1], bins=10, alpha=0.7, label='Y Axis')
        xPeaks = argrelextrema(xObj[0], np.greater)
        yPeaks = argrelextrema(yObj[0], np.greater)
        plt.scatter(xObj[1][xPeaks], xObj[0][xPeaks], c='red', label='Peaks')
        plt.scatter(yObj[1][yPeaks], yObj[0][yPeaks], c='cyan', label='Peaks')
        plt.legend()
        plt.show()

    
    def buildFullyConnectedGraph(self, data):
        """
        Inputs:
        data               = Nx3 array with the x, y, and, z coordinates in their respective columns
                             recommended that you normalize the data.

        Returns:
        self.FGC           = Dictionary object that represents the fully connected graph of the 3D data
        """
        
        # this could be memory optimized by using
        # pdist without squareform
        dists = squareform(pdist(data, 'euclidean'))
        for v, k in enumerate(dists):
            A = {}
            for w, val in enumerate(k):
                if val < self.eps:
                    A[w] = val
            self.FCG[v] = A
        print('FCG is built')

    
    # David Eppstein, UC Irvine, 8 Mar 2002   
    # https://code.activestate.com/recipes/119466-dijkstras-algorithm-for-shortest-paths/?in=user-218935
    def Dijkstra(self):
        """
        Find shortest paths from the start vertex to all
        vertices nearer than or equal to the end.

        Inputs:
        G              = graph which has the following representation: A vertex can be any object that can
                         be used as an index into a dictionary.  G is a dictionary, indexed by vertices.  
                         For any vertex v, G[v] is itself a dictionary, indexed by the neighbors of v.  For 
                         any edge v->w, G[v][w] is the length of the edge. 
                         
        Returns:
        self.D         = the geodesic distances of every point in G from a given startpoint 
        """
        start = self.midIdx
        Q = priorityDictionary()   # est.dist. of non-final vert.
        Q[start] = 0

        for v in Q:
            self.D[v] = Q[v]

            for w in self.FCG[v]:
                vwLength = self.D[v] + self.FCG[v][w]
                if w in self.D:
                    if vwLength < self.D[w]:
                        raise ValueError
                elif w not in Q or vwLength < Q[w]:
                    Q[w] = vwLength