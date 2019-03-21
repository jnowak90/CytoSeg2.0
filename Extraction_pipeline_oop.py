################################################################################
# Module: Extraction_pipeline.py
# Description: Test imports and network extraction
# License: GPL3, see full license in LICENSE.txt
# Web: https://github.com/jnowak90/CytoSeg2.0
################################################################################

################### imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import random
import scipy as sp
import skimage
import skimage.io
import skimage.filters
import sys
import glob
import utils

print("\nYour python version used for this script:")
print(sys.version)

class CytoSeg:

    def __init__(self, pathToFolder, parameterString):
        self.pathToFolder = pathToFolder
        self.parameterString = parameterString
        self.originalData = []
        self.randomData = []

        # set parameters
        self.randw, self.randn, self.depth, self.sigma, self.block, self.small, self.factr = self.parameterString.split(",")

        self.randw = int(self.randw)
        self.randn = int(self.randn)
        self.depth = float(self.depth)
        self.sigma = float(self.sigma)
        self.block = float(self.block)
        self.small = float(self.small)
        self.factr = float(self.factr)

        # go to selected folder and search filtered image
        os.chdir(self.pathToFolder)
        self.filterImg = glob.glob('*_filter.tif')

        self.imgRaw = skimage.io.imread(self.filterImg[0], plugin='tifffile')
        if self.imgRaw.shape[2] in (3,4):
            self.imgRaw = np.swapaxes(self.imgRaw, -1, -3)
            self.imgRaw = np.swapaxes(self.imgRaw, -1, -2)
        self.slices = len(self.imgRaw)
        """
        add analysis for 3D graphs
        self.Z = 1
        self.shape = self.imgRaw.shape
        if (len(self.shape) > 3):
            self.Z = self.shape[3]
        """

        # find and open image mask
        self.maskImg = glob.glob('*_mask.tif')
        self.mask = skimage.io.imread(self.maskImg[0], plugin='tifffile') > 0

        print('Start extraction of image',self.filterImg[0])
        for i in range(self.slices):
            print('\n' , i+1, 'of', self.slices)
            self.imgSlice = self.imgRaw[i]
            self.imgSlice = utils.im2d3d(self.imgSlice)
            self.imgGaussian = skimage.filters.gaussian(self.imgSlice, self.sigma)
            self.imgTube, self.imgSkeleton = utils.skeletonize_graph(self.imgGaussian, self.mask, self.sigma, self.block, self.small, self.factr)
            self.imgNodes = utils.node_graph(self.imgSkeleton > 0, self.imgGaussian)

            self.originalGraph, self.originalPosition = utils.make_graph(self.imgNodes, self.imgGaussian)
            self.originalNormalizedGraph, self.originalProperties, self.unifiedGraph = self.processGraph(self.originalGraph, self.originalPosition, self.imgGaussian, self.mask)
            self.originalData.append([i, self.originalNormalizedGraph, self.originalPosition, self.originalProperties])

            for r in range(self.randn):
                self.randomGraph, self.randomPosition = utils.randomize_graph(self.unifiedGraph, self.originalPosition, self.mask, planar=1, weights=self.randw)
                self.randomNormalizedGraph, self.randomProperties, _ = self.processGraph(self.randomGraph, self.randomPosition, self.imgGaussian, self.mask)
                self.randomData.append([i, self.randomNormalizedGraph, self.randomPosition, self.randomProperties])

            if i==0:
                print('Export plot.')
                self.plotSkeleton(self.originalData, self.randomData)

        print('\nExport data.')
        self.saveData(self.originalData, self.randomData)

    def processGraph(self, graph, graphPosition, Gaussian, mask):
        self.unifiedGraph = utils.unify_graph(graph)
        self.connectedGraph = utils.connect_graph(self.unifiedGraph, graphPosition, Gaussian)
        self.centralizedGraph = utils.centralize_graph(self.connectedGraph)
        self.normalizedGraph = utils.normalize_graph(self.centralizedGraph)
        self.graphProperties = utils.compute_graph(self.normalizedGraph, graphPosition, mask)
        return(self.normalizedGraph, self.graphProperties, self.unifiedGraph)

    def plotSkeleton(self, originalData, randomData):
            originalGraph, originalPosition = originalData[0][1], originalData[0][2]
            randomGraph, randomPosition = randomData[0][1], randomData[0][2]

            plt.clf()
            gs = mpl.gridspec.GridSpec(1, 3, width_ratios=[1,1,1], height_ratios=[1], left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.1, hspace=0.1)
            aspect = 2.0
            alpha = 1.0
            lw = 1.5
            wh = np.array(np.where(self.mask))[::-1]
            axis = np.hstack(zip(np.nanmin(wh, 1), np.nanmax(wh, 1)))

            plt.subplot(gs[0])
            plt.title('biological\nactin network')
            plt.imshow(self.imgRaw[0], cmap='Greys', interpolation='nearest', aspect=aspect)
            originalEdgeCapacity = 1.0 * np.array([d['capa'] for u,v,d in originalGraph.edges(data=True)])
            nx.draw_networkx_edges(originalGraph, originalPosition[:,:2], edge_color=plt.cm.plasma(originalEdgeCapacity / originalEdgeCapacity.max()), width=lw, alpha=alpha)
            plt.axis(axis)
            plt.axis('off')

            plt.subplot(gs[1])
            plt.title('randomized\nactin network')
            plt.imshow(self.imgRaw[0], cmap='Greys', interpolation='nearest', aspect=aspect)
            randomEdgeCapacity = 1.0 * np.array([d['capa'] for u,v,d in randomGraph.edges(data=True)])
            nx.draw_networkx_edges(randomGraph, randomPosition[:,:2], edge_color=plt.cm.plasma(randomEdgeCapacity / randomEdgeCapacity.max()), width=lw, alpha=alpha)
            plt.axis(axis)
            plt.axis('off')
            plt.savefig(self.pathToFolder + '/out_plot.pdf')

    def saveData(self, originalData, randomData):
        properties = ['time','# nodes','# edges','# connected components','avg. edge capacity','assortativity','avg. path length','CV path length','algebraic connectivity','CV edge angles','crossing number']

        # original graphs
        originalPositions = np.array([np.hstack([d[2]]) for d in originalData])
        originalProperties = np.array([np.hstack([d[0],d[-1]]) for d in originalData])
        originalGraphs = [d[1] for d in originalData]
        np.save(self.pathToFolder + '/originalGraphPositions.npy', originalPositions)
        df = pd.DataFrame(originalProperties, columns=properties)
        df.to_csv(self.pathToFolder + '/originalGraphProperties.csv', sep=';', encoding='utf-8')
        nx.write_gpickle(originalGraphs, self.pathToFolder + '/originalGraphs.gpickle')

        # random graphs
        randomPositions = np.array([np.hstack([d[2]]) for d in randomData])
        randomProperties = np.array([np.hstack([d[0],d[-1]]) for d in randomData])
        randomGraphs = [d[1] for d in randomData]
        np.save(self.pathToFolder + '/randomGraphPositions.npy', randomPositions)
        df = pd.DataFrame(randomProperties, columns=properties)
        df.to_csv(self.pathToFolder + '/randomGraphProperties.csv', sep=';', encoding='utf-8')
        nx.write_gpickle(randomGraphs, self.pathToFolder + '/randomGraphs.gpickle')

myExtraction = CytoSeg(sys.argv[1], sys.argv[2])
