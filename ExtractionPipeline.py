################################################################################
# Module: Extraction_pipeline.py
# Description: Test imports and network extraction
# License: GPL3, see full license in LICENSE.txt
# Web: https://github.com/jnowak90/CytoSeg2.0
################################################################################

################### imports
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import networkx as nx
import numpy as np
import os
import pandas as pd
import random
import scipy as sp
import skimage
from skimage import io, filters
import sys
import glob
from collections import Counter
import pickle
import utils

class CytoSeg:

    def __init__(self, pathToPlugin, pathToFilterImage, parameterString, osSystem):

        self.pathToFilterImage = pathToFilterImage
        self.parameterString = parameterString
        self.osSystem = int(osSystem)
        self.originalGraphs, self.originalProperties = [], []
        self.randomGraphs, self.randomProperties = [], []
        if self.osSystem == 1:
            self.pathToPlugin = '\\'.join(pathToPlugin.split('\\')[:-1])
        else:
            self.pathToPlugin = '/'.join(pathToPlugin.split('/')[:-1])

        # set parameters
        self.randn, self.sigma, self.block, self.small, self.factr = self.parameterString.split(",")

        self.randn = int(self.randn)
        self.sigma = float(self.sigma)
        self.block = float(self.block)
        self.small = float(self.small)
        self.factr = float(self.factr)
        self.get_parameters()

        # go to selected folder and search filtered image
        if '_filter.tif' not in self.pathToFilterImage:
            print("ERROR: No pre-processed image ('*_filter.tif') was found. Select pre-processing to create the pre-processed image.")
        else:
            if self.osSystem == 0:
                self.pathToFolder = ('/').join(self.pathToFilterImage.split('/')[:-1]) + '/'
            else:
                self.pathToFolder = ('\\').join(self.pathToFilterImage.split('\\')[:-1]) + '\\'
            self.imgRaw = skimage.io.imread(self.pathToFilterImage, plugin='tifffile')
            if len(self.imgRaw.shape) > 2:
                if self.imgRaw.shape[2] in (3, 4):
                    self.imgRaw = np.swapaxes(self.imgRaw, -1, -3)
                    self.imgRaw = np.swapaxes(self.imgRaw, -1, -2)
                self.slices = len(self.imgRaw)
            else:
                self.slices = 1

            # find and open image mask
            if self.osSystem == 0:
                self.maskImg = ('_').join(self.pathToFilterImage.split('/')[-1].split('_')[:-1])
            else:
                self.maskImg = ('_').join(self.pathToFilterImage.split('\\')[-1].split('_')[:-1])
            self.mask = skimage.io.imread(self.pathToFolder + self.maskImg + '_mask.tif', plugin='tifffile') > 0

            print('Start extraction of image')
            for i in range(self.slices):
                print('\n' , i+1, 'of', self.slices)
                if self.slices == 1:
                    self.imgSlice = self.imgRaw.copy()
                else:
                    self.imgSlice = self.imgRaw[i]
                self.imgGaussian = skimage.filters.gaussian(self.imgSlice, self.sigma)
                self.imgTube, self.imgSkeleton = utils.skeletonize_graph(self.imgGaussian, self.mask, self.sigma, self.block, self.small, self.factr)
                if np.sum(self.imgSkeleton) == 0:
                    print("ERROR: No skeleton was extracted from the selected image. Check the parameters and try again.")
                else:
                    self.imgNodes = utils.node_graph(self.imgSkeleton > 0, self.imgGaussian)

                    self.originalGraph, self.originalPosition = utils.make_graph(self.imgNodes, self.imgGaussian)
                    self.originalNormalizedGraph, self.originalProps, self.unifiedGraph = self.processGraph(self.originalGraph, self.originalPosition, self.imgGaussian, self.mask, i)
                    self.originalGraphs.append(self.originalNormalizedGraph)
                    self.originalProperties.append(self.originalProps)

                    if self.randn != 0:
                        for r in range(self.randn):
                            self.randomGraph, self.randomPosition = utils.randomize_graph(self.unifiedGraph, self.originalPosition, self.mask)
                            self.randomNormalizedGraph, self.randomProps, _ = self.processGraph(self.randomGraph, self.randomPosition, self.imgGaussian, self.mask, r)
                            self.randomGraphs.append(self.randomNormalizedGraph)
                            self.randomProperties.append(self.randomProps)

                    if i==0:
                        print('Export plot.')
                        self.plotSkeleton(self.originalGraphs, self.randomGraphs)

            if np.sum(self.imgSkeleton) != 0:
                print('\nExport data.')
                self.saveData(self.originalGraphs, self.randomGraphs, self.originalProperties, self.randomProperties)

    # save the selected parameters in a file
    def get_parameters(self):
        params = "" + str(self.randn) + "," + str(self.sigma) + "," + str(self.block) + "," + str(self.small) + "," + str(self.factr)
        if self.osSystem == 1:
            np.savetxt(self.pathToPlugin + "\\defaultParameter.txt", [params], fmt='%s')
        else:
            np.savetxt(self.pathToPlugin + "/defaultParameter.txt", [params], fmt='%s')

    def processGraph(self, graph, graphPosition, Gaussian, mask, index):
        self.unifiedGraph = utils.unify_graph(graph)
        self.connectedGraph = utils.connect_graph(self.unifiedGraph, graphPosition, Gaussian)
        self.centralizedGraph = utils.centralize_graph(self.connectedGraph)
        self.normalizedGraph = utils.normalize_graph(self.centralizedGraph)
        self.graphProperties = utils.compute_graph(self.normalizedGraph, graphPosition, mask, index)
        return(self.normalizedGraph, self.graphProperties, self.unifiedGraph)

    def most_frequent(self, List):
        occurence_count = Counter(List)
        return occurence_count.most_common(1)[0][0]

    def plotSkeleton(self, originalGraphs, randomGraphs):
            originalGraph = originalGraphs[0]
            originalPosition = nx.get_node_attributes(originalGraph, 'pos')
            if self.randn != 0:
                randomGraph = randomGraphs[0]
                randomPosition = nx.get_node_attributes(randomGraph, 'pos')

            if self.most_frequent(self.imgRaw[0].flatten()) <= 128:
                cmapImage = 'gray_r'
            else:
                cmapImage = 'gray'

            fig, (ax1, ax2) = plt.subplots(ncols=2)

            if self.slices == 1:
                ax1.imshow(self.imgRaw, cmap=cmapImage)
            else:
                ax1.imshow(self.imgRaw[0], cmap=cmapImage)
            ax1.set_title('Biological actin network', fontsize=7)
            originalEdgeCapacity = 1.0 * np.array([property['capa'] for node1, node2, property in originalGraph.edges(data=True)])
            nx.draw_networkx_edges(originalGraph, originalPosition, edge_color=plt.cm.plasma(originalEdgeCapacity / originalEdgeCapacity.max()), width=1.2, ax=ax1)
            divider = make_axes_locatable(ax1)
            cax1 = divider.append_axes("right", size="5%", pad=0.05)
            m1 = plt.cm.ScalarMappable(cmap="plasma")
            m1.set_array(originalEdgeCapacity)
            cbar1 = fig.colorbar(m1, cax=cax1)
            cbar1.ax.tick_params(labelsize=7)
            ax1.axes.get_yaxis().set_visible(False)
            ax1.axes.get_xaxis().set_visible(False)

            if self.slices == 1:
                ax2.imshow(self.imgRaw, cmap=cmapImage)
            else:
                ax2.imshow(self.imgRaw[0], cmap=cmapImage)
            ax2.set_title('Randomized actin network', fontsize=7)
            if self.randn != 0:
                randomEdgeCapacity = 1.0 * np.array([property['capa'] for node1, node2, property in randomGraph.edges(data=True)])
                nx.draw_networkx_edges(randomGraph, randomPosition, edge_color=plt.cm.plasma(randomEdgeCapacity / randomEdgeCapacity.max()), width=1.2, ax=ax2)
                divider = make_axes_locatable(ax2)
                cax2 = divider.append_axes("right", size="5%", pad=0.05)
                m2 = plt.cm.ScalarMappable(cmap="plasma")
                m2.set_array(randomEdgeCapacity)
                cbar2 = fig.colorbar(m2, cax=cax2)
                cbar2.ax.tick_params(labelsize=7)
            ax2.axes.get_yaxis().set_visible(False)
            ax2.axes.get_xaxis().set_visible(False)

            plt.tight_layout(h_pad=1)

            if self.osSystem == 1:
                fig.savefig(self.pathToFolder + "\\ExtractedNetworks.png", bbox_inches="tight", dpi=300)
            else:
                fig.savefig(self.pathToFolder + "/ExtractedNetworks.png", bbox_inches="tight", dpi=300)

    def saveData(self, originalGraphs, randomGraphs, originalProperties, randomProperties):
        properties = ['time', '# nodes', '# edges', '# connected components', 'avg. edge capacity', 'assortativity', 'avg. path length', 'CV path length', 'algebraic connectivity', 'CV edge angles', 'crossing number']

        # original graphs
        df = pd.DataFrame(originalProperties, columns=properties)
        df = df.astype(dtype={'time' : 'int32', '# nodes': 'int32', '# edges' : 'int32', '# connected components' : 'int32', 'avg. edge capacity' : 'float', 'assortativity' : 'float', 'avg. path length' : 'float', 'CV path length' : 'float', 'algebraic connectivity' : 'float', 'CV edge angles' : 'float', 'crossing number' : 'float'})
        if self.osSystem == 1:
            df.to_csv(self.pathToFolder + '\\originalGraphProperties.csv', sep=';', decimal=',', encoding='utf-8', index=False)
            with open (self.pathToFolder + '\\originalGraphs.gpickle', 'wb') as f:
                pickle.dump(originalGraphs, f, pickle.HIGHEST_PROTOCOL)
        else:
            df.to_csv(self.pathToFolder + '/originalGraphProperties.csv', sep=';', encoding='utf-8', index=False)
            with open (self.pathToFolder + '/originalGraphs.gpickle', 'wb') as f:
                pickle.dump(originalGraphs, f, pickle.HIGHEST_PROTOCOL)

        # random graphs
        if self.randn != 0:
            df = pd.DataFrame(randomProperties, columns=properties)
            df = df.astype(dtype={'time' : 'int32', '# nodes': 'int32', '# edges' : 'int32', '# connected components' : 'int32', 'avg. edge capacity' : 'float', 'assortativity' : 'float', 'avg. path length' : 'float', 'CV path length' : 'float', 'algebraic connectivity' : 'float', 'CV edge angles' : 'float', 'crossing number' : 'float'})
        if self.osSystem == 1:
            df.to_csv(self.pathToFolder + '\\randomGraphProperties.csv', sep=';', decimal=',', encoding='utf-8', index=False)
            with open (self.pathToFolder + '\\randomGraphs.gpickle', 'wb') as f:
                pickle.dump(originalGraphs, f, pickle.HIGHEST_PROTOCOL)
        else:
            df.to_csv(self.pathToFolder + '/randomGraphProperties.csv', sep=';', encoding='utf-8', index=False)
            with open (self.pathToFolder + '/randomGraphs.gpickle', 'wb') as f:
                pickle.dump(originalGraphs, f, pickle.HIGHEST_PROTOCOL)
myExtraction = CytoSeg(sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3])
