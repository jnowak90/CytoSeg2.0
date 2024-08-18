# -*- coding: utf-8 -*-

################################################################################
# Module: utils.py
# Description: network extraction from images of actin filaments.
# License: GPL3, see full license in LICENSE.txt
# Web: https://github.com/janowak90/CytoSeg2.0
################################################################################

import numpy as np
import skimage
from skimage import filters, morphology, feature, segmentation, draw
import scipy as sp
from scipy import ndimage, linalg, spatial
from packaging.version import Version
import networkx as nx
import random
from collections import Counter
##############################################################################

def skeletonize_graph(imageRaw, mask, sigma, block, small, factr):
    """Filter and skeletonize image of filament structures.

    Parameters
    ----------
    imageRaw : original image
    mask : binary array of cellular region of interest
    sigma : width of tubeness filter and filament structures
    block : block size of adaptive median filter
    small : size of smallest components
    factr : fraction of average intensity below which components are removed

    Returns
    -------
    imageTubeness : image after application of tubeness filter
    imageFiltered : filtered and skeletonized image

    """
    imageRaw -= imageRaw[mask].min()
    imageRaw *= 255.0 / imageRaw.max()
    dimensionY, dimensionX = imageRaw.shape
    imageTubeness = imageRaw.copy() * 0
    imageBinary = imageRaw.copy() * 0
    imageTubeness = tube_filter(imageRaw, sigma)
    threshold = skimage.filters.threshold_local(imageTubeness, block)
    imageBinary = imageTubeness > threshold
    imageSkeleton = skimage.morphology.skeletonize(imageBinary > 0)
    ones = np.ones((3, 3))
    imageCleaned = skimage.morphology.remove_small_objects(imageSkeleton, small, connectivity=2) > 0
    imageCleaned = (imageCleaned * mask) > 0
    imageLabeled, labels = sp.ndimage.label(imageCleaned, structure=ones)
    mean = imageRaw[imageCleaned].mean()
    means = [np.mean(imageRaw[imageLabeled == label]) for label in range(1, labels + 1)]
    imageFiltered = 1.0 * imageCleaned.copy()
    for label in range(1, labels + 1):
        if (means[label - 1] < mean * factr):
            imageFiltered[imageLabeled == label] = 0
    imageFiltered = skimage.morphology.remove_small_objects(imageFiltered > 0, 2, connectivity=8)
    return(imageTubeness, imageFiltered)

def tube_filter(imageRaw, sigma):
    """Apply tubeness filter to image.

    Parameters
    ----------
    imageRaw : original two-dimensional image
    sigma : width parameter of tube-like structures

    Returns
    -------
    imageRescaled : filtered and rescaled image

    """
    if Version(skimage.__version__) < Version('0.14'):
        imageHessian = skimage.feature.hessian_matrix(imageRaw, sigma=sigma, mode='reflect')
        imageHessianEigenvalues = skimage.feature.hessian_matrix_eigvals(imageHessian[0], imageHessian[1], imageHessian[2])
    else:
        imageHessian = skimage.feature.hessian_matrix(imageRaw, sigma=sigma, mode='reflect', order='xy')
        imageHessianEigenvalues = skimage.feature.hessian_matrix_eigvals(imageHessian)
    imageFiltered =- 1.0 * imageHessianEigenvalues[1]
    imageRescaled = 255.0 * (imageFiltered - imageFiltered.min()) / (imageFiltered.max() - imageFiltered.min())
    return(imageRescaled)

def node_graph(imageSkeleton, imageGaussian):
    """Construct image indicating background (=0), filaments (=1), and labeled nodes (>1).

    Parameters
    ----------
    imageSkeleton : skeletonized image of filament structures
    imageGaussian : Gaussian filtered image of filament structures

    Returns
    -------
    imageAnnotated : image indicating background (0), filaments (1), and nodes (>1)

    """
    ones = np.ones((3, 3))
    imageFiltered = sp.ndimage.generic_filter(imageSkeleton, node_find, footprint=ones, mode='constant', cval=0)
    imageNodeCondense = node_condense(imageFiltered, imageGaussian, ones)
    imageLabeledNodes = skimage.segmentation.relabel_sequential(imageNodeCondense)[0]
    imageLabeledSkeleton, labels = sp.ndimage.label(imageSkeleton, structure=ones)
    for label in range(1, labels + 1):
        detectedNodes = np.max((imageLabeledSkeleton == label) * (imageLabeledNodes > 0))
        if (detectedNodes == 0):
            imageSkeleton[imageLabeledSkeleton == label] = 0
    imageAnnotated = 1 * ((imageSkeleton + imageLabeledNodes) > 0) + imageLabeledNodes
    return(imageAnnotated)

def node_condense(imageNodes, imageGrayscale, ones):
    """Condense neighboring to single node located at center of mass.

    Parameters
    ----------
    imageNodes : binary node array (0 = background; 1 = nodes)
    imagesGrayscale : gray-scale intensity image
    ones : array defining neighborhood structure

    Returns
    -------
    imageLabeledNodes : condensed and labeled node array (0 = background; 1-N = nodes)

    """
    imageLabeled, labels = sp.ndimage.label(imageNodes, structure=ones)
    sizes = sp.ndimage.sum(imageLabeled > 0, imageLabeled, range(1, labels + 1))
    centerOfMass = sp.ndimage.center_of_mass(imageGrayscale, imageLabeled, range(1, labels + 1))
    for label in range(labels):
        if (sizes[label] > 1):
            idx = (imageLabeled == label + 1)
            idm = tuple(np.add(centerOfMass[label], 0.5).astype('int'))
            imageLabeled[idx] = 0
            imageLabeled[idm] = label + 1
    imageLabeledNodes, _ = sp.ndimage.label(imageLabeled > 0, structure=ones)
    imageLabeledNodes = imageLabeledNodes.astype('int')
    return(imageLabeledNodes)

def node_find(imageBinary):
    """Find nodes in binary filament image.

    Parameters
    ----------
    imageBinary : section of binary filament image

    Returns
    -------
    node : central pixel of image section (0 = not a node; 1 = node)

    """
    imageSection = np.reshape(imageBinary, (3, 3))
    node = 0
    if (imageSection[1, 1] == 1):
        imageSection[1, 1] = 0
        imageLabeled, labels = sp.ndimage.label(imageSection)
        if (labels != 0 and labels != 2):
            node = 1
    return(node)

def make_graph(imageAnnotated, imageGaussian):
    """Construct network representation from image of filament structures.

    Parameters
    ----------
    imageAnnotated : image indicating background (=0), filaments (=1), and labeled nodes (>1)
    imageGaussian : Gaussian filtered image of filament structures

    Returns
    -------
    graph : network representation of filament structures
    nodePositions : node positions

    """
    nodeNumber = imageAnnotated.max() - 1
    distanceDiagonalPixels, distanceDiagonalPixelsCubic = np.sqrt(2.0), np.sqrt(3.0)
    distanceMatrix = np.array([[distanceDiagonalPixelsCubic, distanceDiagonalPixels, distanceDiagonalPixelsCubic], [distanceDiagonalPixels, 1, distanceDiagonalPixels],
    [distanceDiagonalPixelsCubic, distanceDiagonalPixels, distanceDiagonalPixelsCubic]])
    nodePositions = np.transpose(np.where(imageAnnotated > 1))[:, ::-1]
    imagePropagatedNodes = imageAnnotated.copy()
    imageFilamentLength = 1.0 * (imageAnnotated.copy() > 0)
    imageFilamentIntensity = 1.0 * (imageAnnotated.copy() > 0)
    dimensionY, dimensionX = imageAnnotated.shape
    filament = (imagePropagatedNodes == 1).sum()
    while (filament > 0):
        nodePixel = np.transpose(np.where(imagePropagatedNodes > 1))
        for posY, posX in nodePixel:
            xMin, xMax, yMin, yMax = bounds(posX - 1, 0, dimensionX), bounds(posX + 2, 0, dimensionX), bounds(posY - 1, 0, dimensionY), bounds(posY + 2, 0, dimensionY)
            nodeNeighborhood = imagePropagatedNodes[yMin:yMax, xMin:xMax]
            nodeFilamentLength = imageFilamentLength[yMin:yMax, xMin:xMax]
            nodeFilamentIntensity = imageFilamentIntensity[yMin:yMax, xMin:xMax]
            imagePropagatedNodes[yMin:yMax, xMin:xMax] = np.where(nodeNeighborhood == 1, imagePropagatedNodes[posY, posX], nodeNeighborhood)
            imageFilamentLength[yMin:yMax, xMin:xMax] = np.where(nodeFilamentLength == 1, distanceMatrix[0:yMax - yMin, 0:xMax - xMin] + imageFilamentLength[posY, posX], nodeFilamentLength)
            imageFilamentIntensity[yMin:yMax, xMin:xMax] = np.where(nodeFilamentIntensity == 1, imageGaussian[posY, posX] + imageFilamentIntensity[posY, posX], nodeFilamentIntensity)
        filament = (imagePropagatedNodes == 1).sum()
    graph = nx.empty_graph(nodeNumber, nx.MultiGraph())
    filamentY, filamentX = np.where(imagePropagatedNodes > 1)
    for posY, posX in zip(filamentY, filamentX):
        nodeIndex = imagePropagatedNodes[posY, posX]
        xMin, xMax, yMin, yMax = bounds(posX - 1, 0, dimensionX), bounds(posX + 2, 0, dimensionX), bounds(posY - 1, 0, dimensionY), bounds(posY + 2, 0, dimensionY)
        filamentNeighborhood = imagePropagatedNodes[yMin:yMax, xMin:xMax].flatten()
        filamentLength = imageFilamentLength[yMin:yMax, xMin:xMax].flatten()
        filamentIntensity = imageFilamentIntensity[yMin:yMax, xMin:xMax].flatten()
        for index, pixel in enumerate(filamentNeighborhood):
            if (pixel != nodeIndex and pixel > 1):
                node1, node2 = np.sort([nodeIndex - 2, pixel - 2])
                nodeDistance = sp.linalg.norm(nodePositions[node1] - nodePositions[node2])
                filamentLengthSum = imageFilamentLength[posY, posX] + filamentLength[index]
                filamentIntensitySum = imageFilamentIntensity[posY, posX] + filamentIntensity[index]
                minimumEdgeWeight = max(1e-9, filamentIntensitySum)
                edgeCapacity = 1.0 * minimumEdgeWeight / filamentLengthSum
                edgeLength = 1.0 * filamentLengthSum / minimumEdgeWeight
                edgeConnectivity  = 0
                edgeJump = 0
                graph.add_edge(node1, node2, edist=nodeDistance, fdist=filamentLengthSum, weight=minimumEdgeWeight, capa=edgeCapacity, lgth=edgeLength, conn=edgeConnectivity, jump=edgeJump)
    attrPos = {k: list(nodePositions[k] for k in range(len(nodePositions))}
    nx.set_node_attributes(graph, attrPos, 'pos')
    return(graph, attrPos)

def bounds(x, xMin, xMax):
    """Restrict number to interval.

    Parameters
    ----------
    x : number
    xMin : lower bound
    xMax : upper bound

    Returns
    -------
    x : bounded number

    """
    if (x < xMin):
        x = xMin
    elif (x > xMax):
        x = xMax
    return(x)

def multi_line_intersect(segment, segmentsAll):
    """Check intersections of line segments.

    Parameters
    ----------
    segment : single line segment
    segmentsAll : multiple line segments

    Returns
    -------
    intersects : Boolean array indicating intersection

    """
    intersects = np.array([False])
    if (len(segmentsAll) > 0):
        d3 = segmentsAll[:, 1] - segmentsAll[:, 0]
        d1 = segment[1, :] - segment[0, :]
        c1x = np.cross(d3, segment[0, :] - segmentsAll[:, 0])
        c1y = np.cross(d3, segment[1, :] - segmentsAll[:, 0])
        c3x = np.cross(d1, segmentsAll[:, 0] - segment[0, :])
        c3y = np.cross(d1, segmentsAll[:, 1] - segment[0, :])
        intersects = np.logical_and(c1x * c1y < 0, c3x * c3y < 0)
    return(intersects)

def unify_graph(graph):
    """Project multigraph to simple graph.

    Parameters
    ----------
    graph : original graph

    Returns
    -------
    simpleGraph : simple graph

    """
    simpleGraph = nx.empty_graph(graph.number_of_nodes())
    for node1, node2, property in graph.edges(data=True):
        edist = property['edist']
        fdist = property['fdist']
        weight = property['weight']
        capa = property['capa']
        lgth = property['lgth']
        conn = property['conn']
        jump = property['jump']
        multi = 1
        if simpleGraph.has_edge(node1, node2):
            simpleGraph[node1][node2]['multi'] += 1.0
            simpleGraph[node1][node2]['capa'] += capa
            if(simpleGraph[node1][node2]['lgth'] > lgth):
                simpleGraph[node1][node2]['lgth'] = lgth
        else:
            simpleGraph.add_edge(node1, node2, edist=edist, fdist=fdist, weight=weight, capa=capa, lgth=lgth, conn=conn, jump=jump, multi=multi)
    nx.set_node_attributes(simpleGraph, nx.get_node_attributes(graph, 'pos'), 'pos')
    return(simpleGraph)

def connect_graph(graph, nodePositions, imageGaussian):
    """Connect graph by adding edges of minimum edge length.

    Parameters
    ----------
    graph : original graph
    nodePositions : node positions
    imageGaussian : Gaussian filtered image of filament structures

    Returns
    -------
    graphConnected : connect graph

    """
    nodePositionsArray = list(nodePositions.values())
    distanceMatrix = sp.spatial.distance_matrix(nodePositionsArray, nodePositionsArray)
    graphConnected = graph.copy()
    nodeNumber = graphConnected.number_of_nodes()
    connectedComponents = nx.connected_components(graphConnected)
    connectedComponents = sorted(connectedComponents, key=len)[::-1]
    while len(connectedComponents) > 1:
        component = connectedComponents[0]
        componentNodes = list(component)
        remainingNodes = list(set(range(nodeNumber)).difference(component))
        distancesBetweenComponents = distanceMatrix[componentNodes][:, remainingNodes]
        selectedComponentNode, selectedRemainingNode = np.unravel_index(distancesBetweenComponents.argmin(), distancesBetweenComponents.shape)
        positionComponentNode, positionRemainingNode = nodePositions[componentNodes[selectedComponentNode]], nodePositions[remainingNodes[selectedRemainingNode]]
        edgeDistance = sp.linalg.norm(np.array(positionComponentNode) - np.array(positionRemainingNode))
        edgeDistance = max(1.0, edgeDistance)
        filamentLength = 1.0 * np.ceil(edgeDistance)
        edgeDefiningNodes = np.array([positionComponentNode[0], positionComponentNode[1], positionRemainingNode[0], positionRemainingNode[1]])
        edgeCoordinatesY, edgeCoordinatesX = skimage.draw.line(*edgeDefiningNodes.astype('int'))
        edgeWeight = np.sum(imageGaussian[edgeCoordinatesX, edgeCoordinatesY])
        edgeWeight = max(1e-9, edgeWeight)
        edgeCapacity = 1.0 * edgeWeight / filamentLength
        edgeLength = 1.0 * filamentLength / edgeWeight
        edgeConnectivity = 1
        edgeJump = 0
        multi = 1
        graphConnected.add_edge(remainingNodes[selectedRemainingNode], componentNodes[selectedComponentNode], edist=edgeDistance, fdist=filamentLength, weight=edgeWeight, capa=edgeCapacity, lgth=edgeLength, conn=edgeConnectivity, jump=edgeJump, multi=multi)
        connectedComponents = nx.connected_components(graphConnected)
        connectedComponents = sorted(connectedComponents, key=len)[::-1]
    return(graphConnected)

def centralize_graph(graph, epb='lgth', efb='capa', ndg='capa', nec='capa', npr='capa'):
    """Compute edge centralities.

    Parameters
    ----------
    graph : original graph
    epb : edge property used for computation of edge path betweenness
    efb : "                                          flow betweenness
    ndg : "                                          degree centrality
    nec : "                                          eigenvector centrality
    npr : "                                          page rank

    Returns
    -------
    graphCentralities : graph with computed edge centralities

    """
    graphCentralities = graph.copy()
    edges = graphCentralities.edges(data=True)
    edgeCapacity = 1.0 * np.array([property['capa'] for node1, node2, property in edges])
    edgeCapacity /= edgeCapacity.sum()
    edgeLength = 1.0 / edgeCapacity
    for index, (node1, node2, property) in enumerate(edges):
        property['capa'] = edgeCapacity[index]
        property['lgth'] = edgeLength[index]
    edgeBetweenCentrality = nx.edge_betweenness_centrality(graphCentralities, weight=epb)
    edgeFlowBetweennessCentrality = nx.edge_current_flow_betweenness_centrality(graphCentralities, weight=efb)
    lineGraph = nx.line_graph(graphCentralities)
    degree = graphCentralities.degree(weight=ndg)
    for node1, node2, property in lineGraph.edges(data=True):
        intersectingNodes = list(set(node1).intersection(node2))[0]
        property[ndg] = degree[intersectingNodes]
    eigenvectorCentrality = nx.eigenvector_centrality_numpy(lineGraph, weight=ndg)
    pageRank = nx.pagerank(lineGraph, weight=ndg)
    degreeCentrality = dict(lineGraph.degree(weight=ndg))
    for index, (node1, node2, property) in enumerate(edges):
        edge = (node1, node2)
        if (edge in edgeBetweenCentrality.keys()):
            property['epb'] = edgeBetweenCentrality[edge]
        else:
            property['epb'] = edgeBetweenCentrality[edge[::-1]]
        if (edge in edgeFlowBetweennessCentrality.keys()):
            property['efb'] = edgeFlowBetweennessCentrality[edge]
        else:
            property['efb'] = edgeFlowBetweennessCentrality[edge[::-1]]
        if (edge in degreeCentrality.keys()):
            property['ndg'] = degreeCentrality[edge]
        else:
            property['ndg'] = degreeCentrality[edge[::-1]]
        if (edge in eigenvectorCentrality.keys()):
            property['nec'] = eigenvectorCentrality[edge]
        else:
            property['nec'] = eigenvectorCentrality[edge[::-1]]
        if (edge in pageRank.keys()):
            property['npr'] = pageRank[edge]
        else:
            property['npr'] = pageRank[edge[::-1]]
    return(graphCentralities)

def normalize_graph(graph):
    """Normalize edge properties.

    Parameters
    ----------
    graph : original graph

    Returns
    -------
    graph : graph with normalized edge properties

    """
    edgeCapacity = 1.0 * np.array([property['capa'] for node1, node2, property in graph.edges(data=True)])
    edgeCapacity /= edgeCapacity.sum()
    edgeLength = 1.0 / edgeCapacity
    edgeLength /= edgeLength.sum()
    epb = 1.0 * np.array([property['epb'] for node1, node2, property in graph.edges(data=True)])
    epb /= epb.sum()
    efb = 1.0 * np.array([property['efb'] for node1, node2, property in graph.edges(data=True)])
    efb /= efb.sum()
    ndg = 1.0 * np.array([property['ndg'] for node1, node2, property in graph.edges(data=True)])
    ndg /= ndg.sum()
    nec = 1.0 * np.array([property['nec'] for node1, node2, property in graph.edges(data=True)])
    nec /= nec.sum()
    npr = 1.0 * np.array([property['npr'] for node1, node2, property in graph.edges(data=True)])
    npr /= npr.sum()
    for index, (node1, node2, property) in enumerate(graph.edges(data=True)):
        property['capa'] = edgeCapacity[index]
        property['lgth'] = edgeLength[index]
        property['epb'] = epb[index]
        property['efb'] = efb[index]
        property['ndg'] = ndg[index]
        property['nec'] = nec[index]
        property['npr'] = npr[index]
    return(graph)

def compute_graph(graph, nodePositions, mask, index):
    """Compute graph properties.

    Parameters
    ----------
    graph : original graph
    nodePositions : node positions
    mask : binary array of cellular region of interest

    Returns
    -------
    properties : list of graph properties

    """
    nodeNumber = graph.number_of_nodes()
    edgeNumber = graph.number_of_edges()
    connectedComponents = connected_components(graph)
    connectedComponentsNumber = len(connectedComponents)
    edgeCapacity = 1.0 * np.array([property['capa'] for node1, node2, property in graph.edges(data=True)])
    bundling = np.nanmean(edgeCapacity)
    assortativity = nx.degree_pearson_correlation_coefficient(graph, weight='capa')
    shortestPathLength = path_lengths(graph)
    reachability = np.nanmean(shortestPathLength)
    shortestPathLengthSD = np.nanstd(shortestPathLength)
    shortestPathLengthCV = 1.0 * shortestPathLengthSD / reachability
    algebraicConnectivity = np.sort(nx.laplacian_spectrum(graph, weight='capa'))[1]
    edgeAngles = edge_angles(graph, nodePositions, mask)
    edgeAnglesMean = np.nanmean(edgeAngles)
    edgeAnglesSD = np.nanstd(edgeAngles)
    edgeAnglesCV = 1.0 * edgeAnglesSD / edgeAnglesMean
    edgeCrossings = crossing_number(graph, nodePositions)
    edgeCrossingsMean = np.nanmean(edgeCrossings)
    properties = [index, nodeNumber, edgeNumber, connectedComponentsNumber, bundling, assortativity, reachability, shortestPathLengthCV, algebraicConnectivity, edgeAnglesCV, edgeCrossingsMean]
    return(properties)

def connected_components(graph):
    """Compute connected components of graph after removal of edges with capacities below 50th percentile.

    Parameters
    ----------
    graph : original graph

    Returns
    -------
    connectedComponentSizes : list of sizes of connected components

    """
    graphCopy = graph.copy()
    edges = graph.edges(data=True)
    edgeCapacity = 1.0 * np.array([property['capa'] for node1, node2, property in edges])
    percentile = np.percentile(edgeCapacity, 50.0)
    for node1, node2, property in edges:
        if property['capa'] <= percentile:
            graphCopy.remove_edge(node1, node2)
    connectedComponents = nx.connected_components(graphCopy)
    connectedComponentSizes = np.array([len(component) for component in connectedComponents])
    return(connectedComponentSizes)

def path_lengths(graph):
    """Compute shortest path lengths.

    Parameters
    ----------
    graph : original graph

    Returns
    -------
    shortestPathLength : array of shortest path lengths

    """
    allPairsPathLengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight='lgth'))
    shortestPathLength = np.array([[length for length in pair.values()] for pair in allPairsPathLengths.values()])
    shortestPathLength = np.tril(shortestPathLength)
    shortestPathLength[shortestPathLength == 0] = np.nan
    return(shortestPathLength)

def edge_angles(graph, nodePositions, mask):
    """Compute distribution of angles between network edges and cell axis.

    Parameters
    ----------
    graph : original graph
    nodePositions : node positions
    mask : binary array of cellular region of interest

    Returns
    -------
    edgeAngles : list of angles between edges and cell axis

    """
    coordinateCellAxis1, coordinateCellAxis2, centerPointAxis, directionVector, cellAxisAngle, rotationMatrix = mask2rot(mask)
    edgeAngles = []
    for node1, node2, property in graph.edges(data=True):
        edgeAngles.append(np.mod(angle360(1.0 * (np.array(nodePositions[node1]) - np.array(nodePositions[node2]))) + 360.0 - cellAxisAngle, 180.0))
    return(edgeAngles)

def crossing_number(graph, nodePositions):
    """Compute number of edge intersections per edge.

    Parameters
    ----------
    graph : original graph
    nodePositions : node positions

    Returns
    -------
    edgeCrossings : list of edge crossing numbers

    """
    edges = np.array(list(graph.edges()))
    edgeLineSegments = []
    edgeCrossings = []
    for (node1, node2) in graph.edges():
        edge = np.array([[nodePositions[node1][0], nodePositions[node1][1]], [nodePositions[node2][0], nodePositions[node2][1]]])
        edgeLineSegments.append(edge)
    for index, (node1, node2) in enumerate(graph.edges()):
        sharedNodes = (edges[:, 0] != node1) * (edges[:, 1] != node1) * (edges[:, 0] != node2) * (edges[:, 1] != node2)
        sharedNodes[index] = False
        edge = np.array([[nodePositions[node1][0], nodePositions[node1][1]], [nodePositions[node2][0], nodePositions[node2][1]]])
        crossings = multi_line_intersect(np.array(edge), np.array(edgeLineSegments)[index])
        edgeCrossings.append(crossings.sum())
    return(edgeCrossings)

def angle360(vector2d):
    """Compute angle of two-dimensional vector relative to y-axis in degrees.

    Parameters
    ----------
    vector2d : two-dimensional vector

    Returns
    -------
    angle : angle in degrees

    """
    dimensionX, dimensionY = vector2d
    rad2deg = 180.0 / np.pi
    angle = np.mod(np.arctan2(-dimensionX, -dimensionY) * rad2deg + 180.0, 360.0)
    return(angle)

def mask2rot(mask):
    """Compute main axis of cellular region of interest.

    Parameters
    ----------
    mask : binary array of cellular region of interest

    Returns
    -------
    coordinateCellAxis1, coordinateCellAxis2 : coordinates along cell axis
    centerPointAxis, directionVector : center point and direction vector of cell axis
    cellAxisAngle : angle between y-axis and main cell axis
    rotationMatrix : rotation matrix

    """
    skeletonizedMask = skimage.morphology.skeletonize(mask)
    coordinatesSkeleton = np.array(np.where(skeletonizedMask > 0)).T[:, ::-1]
    pointsOnSkeleton = int(len(coordinatesSkeleton) * 0.2)
    coordinateCellAxis1 = coordinatesSkeleton[pointsOnSkeleton]
    coordinateCellAxis2 = coordinatesSkeleton[-pointsOnSkeleton]
    centerPointAxis = coordinatesSkeleton[int(len(coordinatesSkeleton) * 0.5)]
    directionVector = coordinateCellAxis1 - coordinateCellAxis2
    cellAxisAngle = angle360(directionVector)
    cellAxisRadian = cellAxisAngle * np.pi / 180.0
    rotationMatrix = np.array([[np.cos(cellAxisRadian), -np.sin(cellAxisRadian)], [np.sin(cellAxisRadian), np.cos(cellAxisRadian)]])
    return(coordinateCellAxis1, coordinateCellAxis2, centerPointAxis, directionVector, cellAxisAngle, rotationMatrix)

def randomize_graph(graph, nodePositions, mask, planar=0, iterations=1000):
    """Randomize graph by shuffling node positions and edges or edge capacities only.

    Parameters
    ----------
    graph : original graph
    nodePositions : node positions
    mask : binary array of cellular region of interest
    planar : ignore edge crossings (=0) or favor planar graph by reducing number of edge crossings (=1)
    iterations : number of iterations before returning original graph

    Returns
    -------
    randomizedGraph : randomized graph
    nodePositionsRandom : randomized node positions

    """
    nodeNumber = graph.number_of_nodes()
    edgeNumber = graph.number_of_edges()
    randomizedGraph = nx.empty_graph(nodeNumber, nx.MultiGraph())
    edgeLengths = np.array([property['edist'] for node1, node2, property in graph.edges(data=True)])
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 9999]
    binNumber = len(bins) - 1
    edgeBins = np.zeros(edgeNumber).astype('int')
    for index, (bin1, bin2) in enumerate(zip(bins[:-1], bins[1:])):
        edgesInBin = (edgeLengths >= bin1) * (edgeLengths < bin2)
        edgeBins[edgesInBin] = index
    edgeWeights = np.array([property['weight'] for node1, node2, property in graph.edges(data=True)])
    edgeCapacities = np.array([property['capa'] for node1, node2, property in graph.edges(data=True)])
    redoRandomization = 1
    iterationNumber = 0
    while (redoRandomization == 1 and iterationNumber < iterations):
        iterationNumber += 1
        nodePositionsRandom = cell_sample(mask, nodeNumber)[:, ::-1].astype('int')
        distanceMatrix = sp.spatial.distance_matrix(nodePositionsRandom, nodePositionsRandom)
        edgeBinsRandom = np.zeros((nodeNumber, nodeNumber)).astype('int')
        for index, (bin1, bin2) in enumerate(zip(bins[:-1], bins[1:])):
            edgesInBin = (distanceMatrix >= bin1) * (distanceMatrix < bin2)
            edgeBinsRandom[edgesInBin] = index
        edgeBinsRandom[np.tri(nodeNumber) > 0] =- 9999
        redoRandomization = 1 * np.max([(edgeBinsRandom == bins).sum() < (edgeBins == bins).sum() for bins in range(binNumber)])
    if (iterationNumber < iterations):
        sortBins = np.argsort(edgeLengths)[::-1]
        edgeBinsSort = edgeBins[sortBins]
        edgeWeightsSort = edgeWeights[sortBins]
        edgeCapacitiesSort = edgeCapacities[sortBins]
        addedEdges = []
        for edge in range(edgeNumber):
            candidateNodes = np.where(edgeBinsRandom == edgeBinsSort[edge])
            candidateNumber = len(candidateNodes[0])
            edgeCrossings = 9999
            selectedCandidates = random.sample(range(candidateNumber), min(50, candidateNumber))
            for candidate in selectedCandidates:
                node1 = candidateNodes[0][candidate]
                node2 = candidateNodes[1][candidate]
                edgeBetweenNodes = np.array([[nodePositionsRandom[node1][0], nodePositionsRandom[node2][0]], [nodePositionsRandom[node1][1], nodePositionsRandom[node2][1]]]).T
                crossingsOfEdges = planar * multi_line_intersect(np.array(edgeBetweenNodes), np.array(addedEdges)).sum()
                if (crossingsOfEdges < edgeCrossings and edgeBinsRandom[node1, node2] >= 0):
                    edgeCrossings = crossingsOfEdges
                    selectedEdge = edgeBetweenNodes
                    selectedNode1, selectedNode2 = node1, node2
            addedEdges.append(selectedEdge)
            nodeDistanceRandom = distanceMatrix[selectedNode1, selectedNode2]
            filamentLengthRandom = 1.0 * np.ceil(nodeDistanceRandom)
            edgeWeightRandom = edgeWeightsSort[edge]
            edgeCapacityRandom = edgeCapacitiesSort[edge]
            edgeLengthRandom = 1.0 * filamentLengthRandom / edgeWeightRandom
            edgeConnectivityRandom = 0
            edgeJumpRandom = 0
            edgeMultiplicity = 1
            randomizedGraph.add_edge(selectedNode1, selectedNode2, edist=nodeDistanceRandom, fdist=filamentLengthRandom, weight=edgeWeightRandom, capa=edgeCapacityRandom, lgth=edgeLengthRandom, conn=edgeConnectivityRandom, jump=edgeJumpRandom, multi=edgeMultiplicity)
            edgeBinsRandom[selectedNode1, selectedNode2] =- 9999
            edgeBinsRandom[selectedNode2, selectedNode1] =- 9999
    else:
        edgeProperties = np.array([property for node1, node2, property in graph.edges(data=True)])
        random.shuffle(edgeProperties)
        randomizedGraph = graph.copy()
        for index, (node1, node2, properties) in enumerate(randomizedGraph.edges(data=True)):
            for key in properties.keys():
                properties[key] = edgeProperties[index][key]
        nodePositionsRandom = np.array(nodePositions.values())
    attrPos = {k: list(nodePositionsRandom[k]) for k in range(len(nodePositionsRandom))}
    nx.set_node_attributes(randomizedGraph, attrPos, 'pos')
    return(randomizedGraph, attrPos)


def cell_sample(mask, samplingPoints):
    """Sample random points uniformly across masked area.

    Parameters
    ----------
    mask : sampling area
    samplingPoints : number of sampling points

    Returns
    -------
    coordsRandom : sampled random points

    """
    maskedArea = np.array(np.where(mask)).T
    maskedAreaLength = len(maskedArea)
    randomIndex = np.random.randint(0, maskedAreaLength, samplingPoints)
    coordsRandom = maskedArea[randomIndex] + np.random.rand(samplingPoints, 2)
    return(coordsRandom)
