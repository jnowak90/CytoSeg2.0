# -*- coding: utf-8 -*-

################################################################################
# Module: utils.py
# Description: Test imports and network extraction
# License: GPL3, see full license in LICENSE.txt
# Web: https://github.com/DavidBreuer/CytoSeg
################################################################################

############################################################################## imports

import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import random
import scipy as sp
import scipy.misc
import scipy.ndimage
import scipy.optimize
import scipy.spatial
import scipy.stats
import scipy.cluster
import skimage
import skimage.filters
import skimage.morphology
import skimage.feature
import skimage.segmentation
import shapely
import shapely.geometry
import sys
import xml
import xml.dom
import xml.dom.minidom
from collections import Counter
from packaging.version import Version

import utils

############################################################################## help functions

def xmlread(name,threed=0):
    """Read Fiji-Trackmate xml file to Python list of lists.

    Parameters
    ----------
    name : name and directory of xml file
    threed : set to 1 for three-dimensional data

    Returns
    -------
    T : list of tracks

    """
    xmldoc=xml.dom.minidom.parse(name)
    spots=xmldoc.getElementsByTagName('Spot')
    tracs=xmldoc.getElementsByTagName('Track')
    S=[]
    N=[]
    for spot in spots:
        n=int(spot.attributes['ID'].value)
        t=float(spot.attributes['POSITION_T'].value)
        x=float(spot.attributes['POSITION_X'].value)
        y=float(spot.attributes['POSITION_Y'].value)
        if(threed): z=float(spot.attributes['POSITION_Z'].value)
        else: z=0
        mi=float(spot.attributes['MEAN_INTENSITY'].value)
        mt=float(spot.attributes['TOTAL_INTENSITY'].value)
        mq=float(spot.attributes['QUALITY'].value)
        md=float(spot.attributes['ESTIMATED_DIAMETER'].value)
        N.append(n)
        S.append([n,t,x,y,z,mi,mt,mq,md])
    T=[]
    for trac in tracs:
        n=int(trac.attributes['TRACK_ID'].value)
        dur=int(float(trac.attributes['TRACK_DURATION'].value))
        dis=float(trac.attributes['TRACK_DISPLACEMENT'].value)
        edges=trac.getElementsByTagName('Edge')
        E=[]
        for edge in edges:
            id0=int(edge.attributes['SPOT_SOURCE_ID'].value)
            id1=float(edge.attributes['SPOT_TARGET_ID'].value)
            vel=float(edge.attributes['VELOCITY'].value)
            n0=N.index(id0)
            n1=N.index(id1)
            m0,t0,x0,y0,z0,mi0,mt0,mq0,md0=S[n0]
            m1,t1,x1,y1,z1,mi1,mt1,mq1,md1=S[n1]
            E.append([t0,x0,y0,z0,mi0,mt0,mq0,md0,t1,x1,y1,z1,mi1,mt1,mq1,md1])
        E=np.array(E)
        if(len(E)>0):
            E=E[E[:,0].argsort()]
        T.append(E)
    return T

def angle360(dxy):
    """Compute angle of two-dimensional vector relative to y-axis in degrees.

    Parameters
    ----------
    dxy : two-dimensional vector

    Returns
    -------
    angle : angle in degrees

    """
    dx,dy=dxy
    rad2deg=180.0/np.pi
    angle=np.mod(np.arctan2(-dx,-dy)*rad2deg+180.0,360.0)
    return angle

def im2d3d(im):
    """Convert two-dimensional array to three-dimensional array.

    Parameters
    ----------
    im : array or image

    Returns
    -------
    im : array or image

    """
    if(len(im.shape)==2):
        im=im[:,:,np.newaxis]
    else:
        im=im
    return im

def remove_duplicates(points):
    """Remove duplicates from list.

    Parameters
    ----------
    points : list

    Returns
    -------
    pointz : list without duplicates

    """
    pointz=pd.DataFrame(points).drop_duplicates().values
    return pointz

def tube_filter(imO,sigma):
    """Apply tubeness filter to image.

    Parameters
    ----------
    imO : original two-dimensional image
    sigma : width parameter of tube-like structures

    Returns
    -------
    imT : filtered and rescaled image

    """
    if Version(skimage.__version__) < Version('0.14'):
        imH=skimage.feature.hessian_matrix(imO,sigma=sigma,mode='reflect')
        imM=skimage.feature.hessian_matrix_eigvals(imH[0],imH[1],imH[2])
    else:
        imH=skimage.feature.hessian_matrix(imO,sigma=sigma,mode='reflect',order='xy')
        imM=skimage.feature.hessian_matrix_eigvals(imH)
    imR=-1.0*imM[1]
    imT=255.0*(imR-imR.min())/(imR.max()-imR.min())
    return imT

def cell_sample(mask,R):
    """Sample random points uniformly across masked area.

    Parameters
    ----------
    mask : sampling area
    R : number of sampling points

    Returns
    -------
    coords : sampled random points

    """
    wh=np.array(np.where(mask)).T
    W=len(wh)
    idx=sp.random.randint(0,W,R)
    coords=wh[idx]+sp.rand(R,2)
    return coords

def node_randomize(cell,R):
    wh=np.array(np.where(cell)).T
    W=len(wh)
    idx=scipy.random.randint(0,W,R)
    return (wh[idx]+scipy.rand(R,2))

def multi_line_intersect(seg,segs):
    """Check intersections of line segments.

    Parameters
    ----------
    seg : single line segment
    sigma : multiple line segments

    Returns
    -------
    intersects : Boolean array indicating intersects

    """
    intersects=np.array([False])
    if(len(segs)>0):
        d3=segs[:,1,:]-segs[:,0,:]
        d1=seg[1,:]-seg[0,:]
        c1x=np.cross(d3,seg[0,:]-segs[:,0,:])
        c1y=np.cross(d3,seg[1,:]-segs[:,0,:])
        c3x=np.cross(d1,segs[:,0,:]-seg[0,:])
        c3y=np.cross(d1,segs[:,1,:]-seg[0,:])
        intersect=np.logical_and(c1x*c1y<0,c3x*c3y<0)
    return intersects

def bounds(x,xmin,xmax):
    """Restrict number to interval.

    Parameters
    ----------
    x : number
    xmin : lower bound
    xmax : upper bound

    Returns
    -------
    x : bounded number

    """
    if(x<xmin):
        x=xmin
    elif(x>xmax):
        x=xmax
    return x


def node_condense(imM,imG,ones):
    """Condense neighboring to single node located at center of mass.

    Parameters
    ----------
    imM : binary node array (0 = background; 1 = nodes)
    imG : gray-scale intensity image
    ones : array defining neighborhood structure

    Returns
    -------
    imL : condensed and labeled node array (0 = background; 1-N = nodes)

    """
    imL,N=sp.ndimage.label(imM,structure=ones)                                  # label nodes
    sizes=sp.ndimage.sum(imL>0,imL,range(1,N+1))                                # compute size of nodes (clusters)
    coms=sp.ndimage.center_of_mass(imG,imL,range(1,N+1))                        # compute center of mass of nodes (clusters)
    for n in range(N):                                                          # for each node...
        if(sizes[n]>1):                                                         # if cluster...
            idx=(imL==n+1)                                                      # get cluster coordinates
            idm=tuple(np.add(coms[n],0.5).astype('int'))                        # get center of mass coordinates
            imL[idx]=0                                                          # remove node cluster
            imL[idm]=n+1                                                        # set node at center of mass
    imL,N=sp.ndimage.label(imL>0,structure=ones)                                # label nodes
    imL=imL.astype('int')
    return imL

def node_find(im):
    """Find nodes in binary filament image.

    Parameters
    ----------
    im : section of binary filament image

    Returns
    -------
    val : central pixel of image section (0 = not a node; 1 = node)

    """
    ims=np.reshape(im,(3,3,3))                                                  # convert image section of 3x3x3 array
    val=0
    if(ims[1,1,1]==1):                                                          # if central pixel lies on filament...
        ims[1,1,1]=0                                                            # remove central pixel
        iml,L=sp.ndimage.label(ims)                                             # label remaining filaments
        if(L!=0 and L!=2):                                                      # if there is one (set end node) or more than two filaments (set crossing node)...
            val=1                                                               # set node
    return val

def perc_random(a):
    r=random.random()
    if r<a:
        R=1
    else:
        R=0
    return R

def connected_components(graph):
    """Compute connected components of graph after removal of edges with capacities below 50th percentile.

    Parameters
    ----------
    graph : original graph

    Returns
    -------
    ca : list of sizes of connected components

    """
    gc=graph.copy()
    edges=graph.edges(data=True)
    ec=1.0*np.array([d['capa'] for u,v,d in edges])
    perc=np.percentile(ec,50.0)
    for u,v,d in edges:
        if d['capa']<=perc:
            gc.remove_edge(u,v)
    cc=nx.connected_components(gc)
    ca=np.array([len(c) for c in cc])
    return ca

def path_lengths(graph):
    """Compute shortest path lengths.

    Parameters
    ----------
    graph : original graph

    Returns
    -------
    dist : array of shortest path lengths

    """
    dists=dict(nx.all_pairs_dijkstra_path_length(graph,weight='lgth'))
    dist=np.array([[v for v in u.values()] for u in dists.values()])
    dist=np.tril(dist)
    dist[dist==0]=np.nan
    return dist

def edge_angles(graph,pos,mask):
    """Compute distribution of angles between network edges and cell axis.

    Parameters
    ----------
    graph : original graph
    pos : node positions
    mask : binary array of cellular region of interest

    Returns
    -------
    degs : list of angles between edges and cell axis

    """
    c0,c1,vc,vd,an,rot=utils.mask2rot(mask)                             # compute angle of cell axis
    degs=[]
    for u,v,d in graph.edges(data=True):                                        # for each edge...
        degs.append(np.mod(utils.angle360(1.0*(pos[u]-pos[v]))+360.0-an,180.0)) # compute angle between edge and cell axis
    return degs

def crossing_number(graph,pos):
    """Compute number of edge intersections per edge.

    Parameters
    ----------
    graph : original graph
    pos : node positions

    Returns
    -------
    cns : list of edge crossing numbers

    """
    ee=np.array(graph.edges())                                                  # get edge edges
    edges=[]
    cns=[]
    for (n1,n2) in graph.edges():                             # for each edge...
        edge=np.array([[pos[n1][0],pos[n1][1]],[pos[n2][0],pos[n2][1]]])        # append edge as line segment
        edges.append(edge)
    for i,(n1,n2) in enumerate(graph.edges()):                             # for each edge...
        idx=(ee[:,0]!=n1)*(ee[:,1]!=n1)*(ee[:,0]!=n2)*(ee[:,1]!=n2)             # exclude edge that share a node with the selected edge
        idx[i]=False                                                            # exclude selected edge itself
        edge=np.array([[pos[n1][0],pos[n1][1]],[pos[n2][0],pos[n2][1]]])        # treat edge as line segment
        cross=utils.multi_line_intersect(np.array(edge),np.array(edges)[idx]) # check intersections of selected edge with remaining edges
        cns.append(cross.sum())                                                 # append crossing number of selected edge
    return cns

############################################################################## graph functions

def skeletonize_graph(imO,mask,sigma,block,small,factr):
    """Filter and skeletonize image of filament structures.

    Parameters
    ----------
    imO : original image
    mask : binary array of cellular region of interest
    sigma : width of tubeness filter and filament structures
    block : block size of adaptive median filter
    small : size of smallest components
    factr : fraction of average intensity below which components are removed

    Returns
    -------
    imR : image after application of tubeness filter
    imA : filtered and skeletonized image

    """
    imO-=imO[mask].min()
    imO*=255.0/imO.max()
    ly,lx,lz=imO.shape
    imR=imO.copy()*0
    imT=imO.copy()*0
    for z in range(lz):
        imR[:,:,z]=tube_filter(imO[:,:,z],sigma)
        threshold = skimage.filters.threshold_local(imR[:,:,z],block)
        imT[:,:,z]=imR[:,:,z]>threshold
    imS=skimage.morphology.skeletonize_3d(imT>0)
    ones=np.ones((3,3,3))
    imC=skimage.morphology.remove_small_objects(imS,small,connectivity=2)>0
    for z in range(lz):
        imC[:,:,z]=imC[:,:,z]*mask
    imC=imC>0
    imL,N=sp.ndimage.label(imC,structure=ones)
    mean=imO[imC].mean()
    means=[np.mean(imO[imL==n]) for n in range(1,N+1)]
    imA=1.0*imC.copy()
    for n in range(1,N+1):
        if(means[n-1]<mean*factr):
            imA[imL==n]=0
    imA=skimage.morphology.remove_small_objects(imA>0,2,connectivity=8)
    return imR,imA

def node_graph(imA,imG):
    """Construct image indicating background (=0), filaments (=1), and labeled nodes (>1).

    Parameters
    ----------
    imA : skeletonized image of filament structures
    imG : Gaussian filtered image of filament structures

    Returns
    -------
    imE : image indicating background, filaments, and nodes

    """
    ones=np.ones((3,3,3))                                                       # neighborhood structure of pixel
    imM=sp.ndimage.generic_filter(imA,utils.node_find,footprint=ones,mode='constant',cval=0) # find nodes as endpoints or crossings of filaments
    imN=utils.node_condense(imM,imG,ones)                               # condense neighboring nodes
    imL=skimage.segmentation.relabel_sequential(imN)[0]                         # relabel nodes
    imB,B=sp.ndimage.label(imA,structure=ones)                                  # label components of skeletoninzed image
    for b in range(1,B+1):                                                      # for each component...
        no=np.max((imB==b)*(imL>0))                                             # if component does not contain node...
        if(no==0):
            imA[imB==b]=0                                                       # remove component
    imE=1*((imA+imL)>0)+imL                                                     # construct image indicating background (=0) filaments (=1) and labeled nodes (>1).
    return imE

def make_graph(imE,imG):
    """Construct network representation from image of filament structures.

    Parameters
    ----------
    imE : image indicating background (=0), filaments (=1), and labeled nodes (>1)
    imG : Gaussian filtered image of filament structures

    Returns
    -------
    graph : network representation of filament structures
    pos : node positions

    """
    N=imE.max()-1                                                               # number of nodes
    sq2=np.sqrt(2.0)                                                            # distance between diagonal pixels
    sq3=np.sqrt(3.0)                                                            # distance between room diagonal pixels
    diag=np.array([[[sq3,sq2,sq3],[sq2,1,sq2],[sq3,sq2,sq3]],[[sq2,1,sq2],[1,0,1],[sq2,1,sq2]],[[sq3,sq2,sq3],[sq2,1,sq2],[sq3,sq2,sq3]]]) # distance matrix of 3x3x3 neighborhood
    pos=np.array(np.where(imE>1)).T[:,::-1].astype('int')                       # node positions
    pos=pos[:,[1,2,0]]                                                          # change order of node positions (x,y,z)
    imY=imE.copy()                                                              # array to propagate nodes
    imL=1.0*(imE.copy()>0)                                                      # array to remember summed length of filament up to current position
    imS=1.0*(imE.copy()>0)                                                      # array to remember summed intensity of filament up to current position
    ly,lx,lz=imE.shape                                                          # get image dimensions
    ys=(imY==1).sum()                                                           # get points in image which are neither background (=0), nor nodes (>1), but filament (=1)
    while(ys>0):                                                                # while there is still "filament" in the image
        c=np.transpose(np.where(imY>1))                                         # positions of node pixels (>1)
        for y,x,z in c:                                                         # for each node pixel (>1)...
            xmin,xmax=utils.bounds(x-1,0,lx),utils.bounds(x+2,0,lx) # consider 3x3x3 neighborhood around our pixel of interest which is cropped at the borders of the image
            ymin,ymax=utils.bounds(y-1,0,ly),utils.bounds(y+2,0,ly)
            zmin,zmax=utils.bounds(z-1,0,lz),utils.bounds(z+2,0,lz)
            sec=imY[ymin:ymax,xmin:xmax,zmin:zmax]                              # get 3x3x3 neighborhood of node array
            lgt=imL[ymin:ymax,xmin:xmax,zmin:zmax]                              # get 3x3x3 neighborhood of filament length array
            stg=imS[ymin:ymax,xmin:xmax,zmin:zmax]                              # get 3x3x3 neighborhood of filament intensity array
            imY[ymin:ymax,xmin:xmax,zmin:zmax]=np.where(sec==1,imY[y,x,z],sec)  # if 3x3x3 neighborhood contains node (>1) set all filament pixels to this node index
            imL[ymin:ymax,xmin:xmax,zmin:zmax]=np.where(lgt==1,diag[0:ymax-ymin,0:xmax-xmin,0:zmax-zmin]+imL[y,x,z],lgt) # if 3x3x3 neighborhood contains filament, increase straight/diagonal/room diagonal surrounding pixels in length array by 1/sqrt(2)/sqrt(3), respectively
            imS[ymin:ymax,xmin:xmax,zmin:zmax]=np.where(stg==1,imG[y,x,z]+imS[y,x,z],stg)  # if 3x3x3 neighborhood contains filament, increase intensity array by intensity of the original image
        ys=(imY==1).sum()                                                       # compute remaining amout of filament
    graph=nx.empty_graph(N,nx.MultiGraph())                                     # create empty multi graph
    ys,xs,zs=np.where(imY>1)                                                    # get all labeled filament pixels
    for y,x,z in zip(ys,xs,zs):                                                 # for each labeled filament pixel...
        xy=imY[y,x,z]                                                           # get node index
        xmin,xmax=utils.bounds(x-1,0,lx),utils.bounds(x+2,0,lx) # consider 3x3x3 neighborhood around our pixel of interest which is cropped at the borders of the image
        ymin,ymax=utils.bounds(y-1,0,ly),utils.bounds(y+2,0,ly)
        zmin,zmax=utils.bounds(z-1,0,lz),utils.bounds(z+2,0,lz)
        sec=imY[ymin:ymax,xmin:xmax,zmin:zmax].flatten()                        # get 3x3x3 neighborhood of filament image
        lgt=imL[ymin:ymax,xmin:xmax,zmin:zmax].flatten()
        stg=imS[ymin:ymax,xmin:xmax,zmin:zmax].flatten()
        for idx,i in enumerate(sec):                                            # check all pixels in 3x3x3 neighborhood...
            if(i!=xy and i>1):                                                  # if the center and neighboring pixels have different labels...
                u,v=np.sort([xy-2,i-2])                                         # sort nodes to avoid adding bidirectional edges (A->B and B->A)
                edist=sp.linalg.norm(pos[u]-pos[v])                             # compute Euklidean distance between the corresponding nodes
                fdist=imL[y,x,z]+lgt[idx]                                       # compute sum of the two partial filament lengths
                weight=imS[y,x,z]+stg[idx]                                      # compute sum of the two partial filament intensities
                weight=max(1e-9,weight)                                         # set minimum edge weight
                capa=1.0*weight/fdist                                           # compute edge capacity as ration of filament weight and length
                lgth=1.0*fdist/weight                                           # compute edge length as inverse capacity
                conn=0                                                          # set edge connectivity variable indicating that edge belongs to original, non-connected network
                jump=0                                                          # set edge jump variable indicating that edge belongs to original, non-periodic network
                graph.add_edge(u,v,edist=edist,fdist=fdist,weight=weight,capa=capa,lgth=lgth,conn=conn,jump=jump) # add edge to network
    return graph,pos

def unify_graph(graph):
    """Project multigraph to simple graph.

    Parameters
    ----------
    graph : original graph

    Returns
    -------
    graphz : simple graph

    """
    graphz=nx.empty_graph(graph.number_of_nodes())                              # construct new empty graph with the same number of nodes
    for u,v,d in graph.edges(data=True):                                        # for each edge in the multigraph...
        edist=d['edist']                                                        # get edge properties
        fdist=d['fdist']
        weight=d['weight']
        capa=d['capa']
        lgth=d['lgth']
        conn=d['conn']
        jump=d['jump']
        multi=1                                                                 # set edge multiplicity to one
        if graphz.has_edge(u,v):                                                # if simple graph already contains the edge in question...
            graphz[u][v]['multi']+=1.0                                          # increase edge multiplicity by one
            graphz[u][v]['capa']+=capa                                          # compute sum of edge capacities
            if(graphz[u][v]['lgth']>lgth):                                      # compute minimum of edge lengths
                graphz[u][v]['lgth']=lgth
        else:
            graphz.add_edge(u,v,edist=edist,fdist=fdist,weight=weight,capa=capa,lgth=lgth,conn=conn,jump=jump,multi=multi) # add edge to simple graph otherwise
    return graphz

def connect_graph(graph,pos,imG):
    """Connect graph by adding edges of minimum edge length.

    Parameters
    ----------
    graph : original graph
    pos : node positions
    imG : Gaussian filtered image of filament structures

    Returns
    -------
    graphz : connect graph

    """
    dists=sp.spatial.distance_matrix(pos,pos)                                   # compute distance matrix between all node positions
    graphz=graph.copy()                                                         # copy original graph
    N=graphz.number_of_nodes()                                                  # get number of nodes
    comp=nx.connected_components(graphz)                                        # compute connected components
    comp=sorted(comp,key=len)[::-1]                                             # sort connected components in descending order according to size
    while len(comp)>1:                                                          # while network is disconnected...
        compo=comp[0]                                                           # get nodes in largest component
        compl=list(compo)
        compi=list(set(range(N)).difference(compo))                             # get remaining nodes
        dist=dists[compl][:,compi]                                              # get distance matrix between nodes of largest component and remaining network
        n0,ni=np.unravel_index(dist.argmin(),dist.shape)                        # find pair of nodes with minimum distance
        p0,pi=pos[compl][n0],pos[compi][ni]
        edist=sp.linalg.norm(p0-pi)                                             # compute distance between nodes
        edist=max(1.0,edist)                                                    # set minimum distance between nodes
        fdist=1.0*np.ceil(edist)                                                # approximate filament length by rounding node distance
        aa=np.array([p0[0],p0[1],pi[0],pi[1]])                                  # draw line between nodes
        yy,xx=skimage.draw.line(*aa.astype('int'))
        zz=(np.linspace(p0[2],pi[2],len(xx))).astype('int')
        weight=np.sum(imG[xx,yy,zz])                                            # compute edge weight as image intensity along line
        weight=max(1e-9,weight)                                                 # set minimum edge weight
        capa=1.0*weight/fdist                                                   # compute edge capacity as ration of filament weight and length
        lgth=1.0*fdist/weight                                                   # compute edge length as inverse capacity
        conn=1                                                                  # set edge connectivity variable indicating that edge belongs to new, connected network
        jump=0                                                                  # set edge jump variable indicating that edge belongs to original, non-periodic network
        multi=1                                                                 # set edge mutiplicity variable
        graphz.add_edge(compi[ni],compl[n0],edist=edist,fdist=fdist,weight=weight,capa=capa,lgth=lgth,conn=conn,jump=jump,multi=multi) # add edge to network
        comp=nx.connected_components(graphz)                                    # compute connected components
        comp=sorted(comp,key=len)[::-1]                                         # sort connected components in descending order according to size
    return graphz

def randomize_graph(graph,pos,mask,planar=0,weights=0,iterations=1000):
    """Randomize graph by shuffling node positions and edges or edge capacities only.

    Parameters
    ----------
    graph : original graph
    pos : node positions
    mask : binary array of cellular region of interest
    planar : ignore edge crossings (=0) or favor planar graph by reducing number of edge crossings (=1)
    weights : shuffle only edge capacities (=0) or node positions and edges (=1)
    iterations : number of iterations before returning original graph

    Returns
    -------
    graphz : randomized graph
    poz : randomized node positions

    """
    if(weights==0):                                                             # if shuffling of edge capacities only...
        ec=np.array([d for u,v,d in graph.edges(data=True)])                    # get edge properties
        random.shuffle(ec)                                                      # shuffle edge capacities
        graphz=graph.copy()                                                     # copy graph
        for j,(u,v,d) in enumerate(graphz.edges(data=True)):                    # for each edge...
            for k in d.keys():                                                  # copy shuffled edge properties
                d[k]=ec[j][k]
        poz=pos                                                                 # copy node positions
    else:                                                                       # shuffling of node positions and edges otherwise
        N=graph.number_of_nodes()                                               # get node number
        E=graph.number_of_edges()                                               # get edge number
        graphz=nx.empty_graph(N,nx.MultiGraph())                                # create new, empty multigraph
        diste=np.array([d['edist'] for u,v,d in graph.edges(data=True)])        # get Euclidean edge lengths
        bins=[0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,9999] # set bin boundaries for edge lengths
        B=len(bins)-1                                                           # get number of bins
        dibse=np.zeros(E).astype('int')                                         # create array for assigning bin numbers to edges
        for i,(b1,b2) in enumerate(zip(bins[:-1],bins[1:])):                    # for each bin...
            ide=(diste>=b1)*(diste<b2)                                          # get edges with Euclidean lengths in the given bin
            dibse[ide]=i                                                        # assign bin number to edges
        eweight=np.array([d['weight'] for u,v,d in graph.edges(data=True)])     # get edge weights
        ecapa=np.array([d['capa'] for u,v,d in graph.edges(data=True)])         # get edge capacities
        redo=1                                                                  # variable indicating that no suitable randomization was obtained yet
        iteration=0                                                             # number of iterations
        while(redo==1 and iteration<iterations):                                # while neither a suitable randomization nor the number of allowed iterations were reached yet...
            iteration+=1                                                        # increase iteration by one
            poz=utils.cell_sample(mask,N)[:,::-1].astype('int') # shuffle xy-components of node positions
            zzz=pos[:,2]                                                        # keep z-component of node positions
            poz=np.vstack([poz.T,zzz]).T                                        # merge xyz-components of node positions
            dista=scipy.spatial.distance_matrix(poz,poz)                        # compute distance matrix between new node positions
            dibsa=np.zeros((N,N)).astype('int')                                 # assign bin numbers to all new, potential edges
            for i,(b1,b2) in enumerate(zip(bins[:-1],bins[1:])):
                ida=(dista>=b1)*(dista<b2)
                dibsa[ida]=i
            dibsa[np.tri(N)>0]=-9999                                            # set lower part of the bin number matrix to negativ number to exclude loops (A->A) and bidirectional edges (A->B and B->A)
            redo=1*np.max([(dibsa==b).sum()<(dibse==b).sum() for b in range(B)]) # check that each original edge can be accommodated given the new node positions
        if(iteration<iterations):                                               # if the number of allowed iterations was not reached yet...
            isort=np.argsort(diste)[::-1]                                       # sort bin assignments, edge weights, and edge capacities by Euclidean length
            diste=diste[isort]
            dibse=dibse[isort]
            eweight=eweight[isort]
            ecapa=ecapa[isort]
            edges=[]                                                            # list of added edges
            for e in range(E):                                                  # for each edge...
                candidates=np.where(dibsa==dibse[e])                            # get candidate pairs of new nodes whose distance matches the Euclidean length of the selected edge
                C=len(candidates[0])                                            # get number of candidate pairs
                cromm=9999                                                      # dummy variable for number of edge crossings
                ii=random.sample(range(C),min(50,C))                            # select up to 50 candidate pairs
                for i in ii:                                                    # for each candidate pair...
                    n1=candidates[0][i]                                         # get nodes
                    n2=candidates[1][i]
                    edge=np.array([[poz[n1][0],poz[n2][0]],[poz[n1][1],poz[n2][1]]]).T # create line segment between candidate nodes
                    cross=planar*utils.multi_line_intersect(np.array(edge),np.array(edges)).sum() # compute number of line segment crossings with existing edges
                    if(cross<cromm and dibsa[n1,n2]>=0):                        # if edge is allowed and number of crossings is smaller than for previous candidates...
                        cromm=cross                                             # store crossing number
                        edgem=edge                                              # store edge
                        m1,m2=n1,n2                                             # store nodes
                edges.append(edgem)                                             # add edge to list of edges
                edist=dista[m1,m2]                                              # set Euclidean distance
                fdist=1.0*np.ceil(edist)                                        # approximate filament length by rounding node distance
                weight=eweight[e]                                               # set edge weight
                capa=ecapa[e]                                                   # set edge capacity
                lgth=1.0*fdist/weight                                           # compute edge length as inverse capacity
                conn=0                                                          # set edge connectivity variable indicating that edge belongs to randomized, non-connected connected network
                jump=0                                                          # set edge jump variable indicating that edge belongs to randomized, non-periodic network
                multi=1                                                         # set edge mutiplicity variable
                graphz.add_edge(m1,m2,edist=edist,fdist=fdist,weight=weight,capa=capa,lgth=lgth,conn=conn,jump=jump,multi=multi) # add edge to network
                dibsa[m1,m2]=-9999                                              # remove edge from allowed edges
                dibsa[m2,m1]=-9999
        else:
            #print('No randomized graph found')
            graphz,poz=utils.new_randomize_graph(graph,pos,mask,planar=1)
    return graphz,poz

def new_randomize_graph(graph,pose,mask,planar=1):#graph,pose,mask,planar=g1u,pos1,imMask,1

    N=graph.number_of_nodes()
    E=graph.number_of_edges()
    diste=np.array([d['edist'] for u,v,d in graph.edges(data=True)])
    eweight=np.array([d['weight'] for u,v,d in graph.edges(data=True)])
    ecapa=np.array([d['capa'] for u,v,d in graph.edges(data=True)])

    bins=[0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,9999]
    B=len(bins)-1
    dibse=np.zeros(E).astype('int')
    for i,(b1,b2) in enumerate(zip(bins[:-1],bins[1:])):
        ide=(diste>=b1)*(diste<b2)
        dibse[ide]=i
    c=Counter(dibse)

#####New addition to number of nodes in new graph
    factor=0.9
    counts=[False]
    while sum(counts)!=len(counts):
        factor=factor+0.1
        if factor>=3.0:
            break
        for j in range(20):
            posi=utils.cell_sample(mask,int(factor*N))[:,::-1].astype('int')
            delaunay=scipy.spatial.Delaunay(posi)
            dista=scipy.spatial.distance_matrix(posi,posi)

            graphi=nx.empty_graph(int(factor*N),nx.MultiGraph())
            for i in range(len(delaunay.simplices)):
                graphi.add_edge(delaunay.simplices[i][0],delaunay.simplices[i][1])
                graphi.add_edge(delaunay.simplices[i][1],delaunay.simplices[i][2])
                graphi.add_edge(delaunay.simplices[i][2],delaunay.simplices[i][0])
            Ei=graphi.number_of_edges()

            b=np.zeros(Ei).astype('int')
            edgei=list(graphi.edges())
            for l in range(len(edgei)):
                disti=dista[edgei[l][0]][edgei[l][1]]
                for i,(b1,b2) in enumerate(zip(bins[:-1],bins[1:])):
                    ida=(disti>=b1)*(disti<b2)
                    if ida==True:
                        b[l]=i
            ca=Counter(b)
            counts=[False]*len(c)
            index=0
            for m in c.keys():
                if c[m]<=ca[m]:
                    counts[index]=True
                index+=1
            if sum(counts)==len(counts):
                break

    if factor<3.0:
        New=nx.empty_graph(int(factor*N),nx.MultiGraph())
        dibsi=np.zeros(B)
        dibsj=np.zeros(B)
        index=random.sample(range(Ei),Ei)

        for ind in index:
            edge=edgei[ind]
            if New.degree(edge[0])>0 or New.degree(edge[1])>0:
                R=perc_random(0.9)
            else:
                R=perc_random(0.1)

            if R==1:
                disti=dista[edge[0]][edge[1]]
                for i,(b1,b2) in enumerate(zip(bins[:-1],bins[1:])):
                    ide=(disti>=b1)*(disti<b2)
                    if ide==True:
                        dibsi[i]+=1
                        dibsj[i]+=1
                        bin_nr=i
                if dibsi[bin_nr]>c[bin_nr]:
                    dibsj[bin_nr]-=1
                else:
                    w=np.where(dibse==bin_nr)[0]
                    v=random.choice(w)
                    edist=dista[edge[0]][edge[1]]
                    fdist=1.0*np.ceil(edist)
                    New.add_edge(edge[0],edge[1],edist=edist,fdist=fdist,weight=eweight[v],capa=ecapa[v],lgth=1.0*fdist/eweight[v],conn=0,jump=0)

        degrees=[]
        for i in New.nodes():
            if New.degree(i)==0:
                degrees.append(i)
        degrees=sorted(degrees,reverse=True)
        posi2=list(posi)
        for n in degrees:
            New.remove_node(n)
            del posi2[n]
        posi2=np.asarray(posi2)
        zzz=pose[0:len(posi2),2]
        posi2=np.vstack([posi2.T,zzz]).T
        New=nx.convert_node_labels_to_integers(New)

        #cn=crossing_number(New,posi2)
        return New,posi2

    else:
        New,posi2=utils.randomize_graph(graph,pose,mask,planar=1,weights=0)
        return New,posi2

def centralize_graph(graph,epb='lgth',efb='capa',ndg='capa',nec='capa',npr='capa'):
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
    graphz : graph with computed edge centralities

    """
    graphz=graph.copy()                                                         # copy graph
    edges=graphz.edges(data=True)                                               # get edge capacities
    ec=1.0*np.array([d['capa'] for u,v,d in edges])
    ec/=ec.sum()                                                                # normalize edge capacities
    el=1.0/ec
    for i,(u,v,d) in enumerate(edges):                                          # update edge capacities and lengths
        d['capa']=ec[i]
        d['lgth']=el[i]
    epb=nx.edge_betweenness_centrality(graphz,weight=epb)                       # compute edge path betweenness
    efb=nx.edge_current_flow_betweenness_centrality(graphz,weight=efb)          # compute edge flow betweenness
    lineg=nx.line_graph(graphz)                                                 # compute line graph
    degree=graphz.degree(weight=ndg)                                            # get capacity weighted edge degree
    for u,v,d in lineg.edges(data=True):                                        # set edge capacity of linegraph to node degree of original graph
        n=list(set(u).intersection(v))[0]
        d[ndg]=degree[n]
    nec=nx.eigenvector_centrality_numpy(lineg,weight=ndg) # compute edge degree, eigenvector, and page rank centrality
    npr=nx.pagerank(lineg,weight=ndg)
    ndg=dict(lineg.degree(weight=ndg))
    for i,(u,v,d) in enumerate(edges):                                          # set edge centralities
        e=(u,v)
        if(e in epb.keys()):
            d['epb']=epb[e]
        else:
            d['epb']=epb[e[::-1]]
        if(e in efb.keys()):
            d['efb']=efb[e]
        else:
            d['efb']=efb[e[::-1]]
        if(e in ndg.keys()):
            d['ndg']=ndg[e]
        else:
            d['ndg']=ndg[e[::-1]]
        if(e in nec.keys()):
            d['nec']=nec[e]
        else:
            d['nec']=nec[e[::-1]]
        if(e in npr.keys()):
            d['npr']=npr[e]
        else:
            d['npr']=npr[e[::-1]]
    return graphz

def normalize_graph(graph):
    """Normalize edge properties.

    Parameters
    ----------
    graph : original graph

    Returns
    -------
    graph : graph with normalized edge properties

    """
    ec=1.0*np.array([d['capa'] for u,v,d in graph.edges(data=True)])
    ec/=ec.sum()
    el=1.0/ec
    el/=el.sum()
    epb=1.0*np.array([d['epb'] for u,v,d in graph.edges(data=True)])
    epb/=epb.sum()
    efb=1.0*np.array([d['efb'] for u,v,d in graph.edges(data=True)])
    efb/=efb.sum()
    ndg=1.0*np.array([d['ndg'] for u,v,d in graph.edges(data=True)])
    ndg/=ndg.sum()
    nec=1.0*np.array([d['nec'] for u,v,d in graph.edges(data=True)])
    nec/=nec.sum()
    npr=1.0*np.array([d['npr'] for u,v,d in graph.edges(data=True)])
    npr/=npr.sum()
    for i,(u,v,d) in enumerate(graph.edges(data=True)):
        d['capa']=ec[i]
        d['lgth']=el[i]
        d['epb']=epb[i]
        d['efb']=efb[i]
        d['ndg']=ndg[i]
        d['nec']=nec[i]
        d['npr']=npr[i]
    return graph

def boundary_graph(jnet,graph,pos,SP,SL,JV,JH,imG,dthres=10.0,jthres=2.5):
    """Generate graph with periodic boundary conditions.

    Parameters
    ----------
    jnet : jump network
    graph : original graph
    pos : node positions
    SP : shortest paths
    SL : shortest path lengths
    JV : number of vertical jumps along shortest path
    JH : number of horizontal jumps along shortest path
    imG : Gaussian filtered image of filament structures

    Returns
    -------
    graphz : graph with periodic boundary conditions

    """
    B=jnet.number_of_nodes()                                                    # get number of nodes of jump network
    C=np.tril((SL<dthres)*((JV+JH)>0)*((JV+JH)<jthres))[B:,B:]                  # get pairs of nodes in jump network that are less than dthres apart and that are connected by at least/most 0/jthres
    wh=np.array(np.where(C)).T
    graphz=nx.MultiGraph(graph.copy())                                          # create new, empty multigraph
    for idx,(w1,w2) in enumerate(wh):                                           # for each pair of nodes, i.e., each potential edge...
        path=SP[B+w1][B+w2]                                                     # get shortest path between selected nodes
        pairs=zip(path[0:],path[1:])
        weight=0.0
        for n0,n1 in pairs:                                                     # for each edge along path...
            if(jnet[n0][n1]['jump']==0):                                        # if it is not a jump edge...
                rr,cc=skimage.draw.line(pos[n0][1],pos[n0][0],pos[n1][1],pos[n1][0]) # draw line along edge
                weight+=imG[cc,rr].sum()                                        # add edge weight as sum of intensities in the underlying image along the line
        edist=SL[B+w1,B+w2]                                                     # set edge Euclidean length
        edist=max(1.0,edist)
        fdist=1.0*np.ceil(edist)                                                # approximate filament arc length
        weight=max(1e-9,weight)
        capa=1.0*weight/fdist                                                   # compute edge capacity
        lgth=1.0*fdist/weight                                                   # compute edge length as inverse capacity
        conn=0                                                                  # set edge connectivity variable indicating that edge belongs to periodic, non-connected connected network
        jump=1                                                                  # set edge jump variable indicating that edge belongs to periodic network
        multi=1                                                                 # set edge mutiplicity variable
        graphz.add_edge(w2,w1,edist=edist,fdist=fdist,weight=weight,capa=capa,lgth=lgth,conn=conn,jump=jump,multi=multi) # add edge
    return graphz

def compute_graph(graph,pos,mask):
    """Compute graph properties.

    Parameters
    ----------
    graph : original graph
    pos : node positions
    mask : binary array of cellular region of interest

    Returns
    -------
    quanta : list of graph properties

    """
    N=graph.number_of_nodes()                                                   # number of nodes
    E=graph.number_of_edges()                                                   # number of edges
    ca=utils.connected_components(graph)                                # compute sizes of connected components
    C=len(ca)                                                                   # number of connected components
    ec=1.0*np.array([d['capa'] for u,v,d in graph.edges(data=True)])            # get edge capacities
    bund=np.nanmean(ec)                                                         # compute average edge capacity ('bundling')
    assort=nx.degree_pearson_correlation_coefficient(graph,weight='capa')       # compute assortativity ('heterogeneity')
    dist=utils.path_lengths(graph)                                      # compute shortest path lengths
    distMU=np.nanmean(dist)                                                     # compute average path length ('reachability')
    distSD=np.nanstd(dist)                                                      # compute standard deviation of path lengths
    distCV=1.0*distSD/distMU                                                    # compute coefficient of variation of path lengths ('disperal')
    ac=np.sort(nx.laplacian_spectrum(graph,weight='capa'))[1]                   # compute algebraic connectivity ('robustness')
    degs=utils.edge_angles(graph,pos[:,:2],mask)                        # compute edge angles relative to cell axis
    angleMU=np.nanmean(degs)                                                    # compute average angle
    angleSD=np.nanstd(degs)                                                     # compute standard deviation of angles
    angleCV=1.0*angleSD/angleMU                                                 # compute coefficient of variation of angles ('contortion')
    cns=utils.crossing_number(graph,pos[:,:2])                          # compute number of edge crossings per edge
    crossing=np.nanmean(cns)                                                  # compute average crossing number
    quants=['# nodes','# edges','# connected components','avg. edge capacity','assortativity','avg. path length','CV path length','algebraic connectivity','CV edge angles','crossing number'] # list of graph property names
    quanta=[N,E,C,bund,assort,distMU,distCV,ac,angleCV,crossing]                # list of graph properties
    return quanta

############################################################################## periodic functions

def mask2rot(mask):
    """Compute main axis of cellular region of interest.

    Parameters
    ----------
    mask : binary array of cellular region of interest

    Returns
    -------
    c0,c1 : coordinates along cell axis
    vc,vd : center point and direction vector of cell axis
    angle : angle between y-axis and main cell axis
    rot : rotation matrix

    """
    line=skimage.morphology.skeletonize(mask)                                   # skeletonize mask
    co=np.array(np.where(line>0)).T[:,::-1]                                     # get coordinates of skeleton line
    L=int(len(co)*0.2)                                                          # get points 20% and 80%  along the cell axis
    c0=co[L]
    c1=co[-L]
    vc=co[int(len(co)*0.5)]                                                     # get center point and direction vector of cell axis
    vd=c0-c1
    angle=utils.angle360(vd)                                            # compute angle of cell axis
    angli=angle*np.pi/180.0                                                     # convert angle to radian
    rot=np.array([[np.cos(angli),-np.sin(angli)],[np.sin(angli),np.cos(angli)]]) # compute rotation matrix
    return c0,c1,vc,vd,angle,rot

def mask2poly(mask):
    """Convert cellular region of interest to polygon.

    Parameters
    ----------
    mask : binary array of cellular region of interest

    Returns
    -------
    polya : original polygon
    polyn : rotated polygon aligned with y-axis

    """
    maski=sp.ndimage.minimum_filter(mask,3,mode='constant',cval=0)              # shrink mask
    polya=skimage.measure.find_contours(maski,0)[0]                             # find contours
    polya=skimage.measure.approximate_polygon(polya,tolerance=0.0)              # approximate polygon
    polya=1.0*remove_duplicates(polya)                                          # remove duplicate points
    c0,c1,vc,vd,an,rot=mask2rot(maski)                                          # compute cell axis
    polyn=np.dot(polya,rot)                                                     # rotate polygon
    return polya[:,::-1],polyn[:,::-1]

def pbc_jnet_border(polyn):
    """Compute border of jump network.

    Parameters
    ----------
    polyn : rotated polygon of cellular region of interest

    Returns
    -------
    graph : border of jump network

    """
    polyi=1.0*polyn.astype('int')                                               # convert coordinates to integers
    polys=shapely.geometry.Polygon(polyi)                                       # convert polygon to shapely polygon
    B=len(polyi)                                                                # get number of polygon points
    graph=nx.empty_graph(B)                                                     # create new, empty graph
    for i in range(2):                                                          # for both x- and y-components...
        bx=polyi[:,i]                                                           # get coordinate
        for idx,x in enumerate(set(bx)):                                        # for each point
            yy=np.sort(np.where(x==bx)[0])                                      # get other points with same coordinate
            Y=len(yy)
            for y in range(Y-1):                                                # for each other point with same coordinate
                y1,y2=yy[y],yy[y+1]
                line=shapely.geometry.LineString([polyi[y1],polyi[y2]])         # draw line between the two selected points
                if(line.within(polys)):                                         # if the line is fully contained within the polygon...
                    graph.add_edge(y1,y2,weight=0.0,jump=0.001**i)              # add the to network (jump=0.001/0.00001 lines parallel to along x/y-axis)
    distb=sp.spatial.distance_matrix(polyn,polyn)                               # compute distance matrix between point of polygon
    for b1 in range(B):                                                         # for each point along polygon
        b2=np.mod(b1+1,B)
        graph.add_edge(b1,b2,weight=distb[b1,b2],jump=0.0)                      # add edge no neighboring point
    return graph

def pbc_jnet_interior(pos,polya,jborder,cthres=10.0):
    """Compute interier of jump network.

    Parameters
    ----------
    pos : node positions
    polya : original polygon of cellular region of interest
    jborder : border of jump network
    cthres : maximum edge length between nodes of original network and border of jump network

    Returns
    -------
    jnet : complete jump network
    SP : array of shortest path lengths
    SL : array of jump sizes
    JV : get number of vertical jumps
    JH : get number of horizonal jumps

    """
    jnet=jborder.copy()                                                         # copy border of jump network
    B=jnet.number_of_nodes()                                                    # get number of nodes
    distn=sp.spatial.distance_matrix(pos,polya)                                 # compute distances between node positions and border of jump network
    for n in range(len(pos)):                                                   # for each node...
        jnet.add_node(B+n)                                                      # add node to jump network
        for e in np.where(distn[n]<cthres)[0]:                                  # add edge if node is close enough to border of jump network
            jnet.add_edge(B+n,e,weight=distn[n,e],jump=0.0)
    for n in range(len(pos)):                                                   # for each node...
        if(jnet.degree(B+n)==0):                                                # add dummy edge to make network connected if node is disconnected
            jnet.add_edge(B+n,0,weight=9999.0,jump=0.0)
    SP=utils.all_pairs_dijkstra_path(jnet,weight='weight',jump='jump') # compute all shortest path in jump network
    SX=utils.all_pairs_dijkstra_path_length(jnet,weight='weight',jump='jump') # compute all shortest path lengths in jump network
    SL=1.0*np.array([[d1 for d1 in d2[0].values()] for d2 in SX.values()])      # array of shortest path lengths
    SJ=1.0*np.array([[d1 for d1 in d2[1].values()] for d2 in SX.values()])      # array of jump sizes
    JV=np.floor(SJ+0.5)                                                         # get number of vertical jumps
    JH=np.floor(np.mod(SJ,1.0)*1000.0+0.5)                                      # get number of horizonal jumps
    return jnet,SP,SL,JV,JH

############################################################################## NetworkX: shortest path algorithms for weighed graphs

# -*- coding: utf-8 -*-
#"""
#Shortest path algorithms for weighed graphs.
#"""
#__author__ = """\n""".join(['Aric Hagberg <hagberg@lanl.gov>',
#                            'Loïc Séguin-C. <loicseguin@gmail.com>',
#                            'Dan Schult <dschult@colgate.edu>'])
##    Copyright (C) 2004-2011 by
##    Aric Hagberg <hagberg@lanl.gov>
##    Dan Schult <dschult@colgate.edu>
##    Pieter Swart <swart@lanl.gov>
##    All rights reserved.
##    BSD license.
#
#__all__ = ['dijkstra_path',
#           'dijkstra_path_length',
#           'bidirectional_dijkstra',
#           'single_source_dijkstra',
#           'single_source_dijkstra_path',
#           'single_source_dijkstra_path_length',
#           'all_pairs_dijkstra_path',
#           'all_pairs_dijkstra_path_length',
#           'dijkstra_predecessor_and_distance',
#           'bellman_ford','negative_edge_cycle']

import heapq
import networkx as nx
from networkx.utils import generate_unique_node

def dijkstra_path(G, source, target, weight='weight',jump= 'jump'):
    """Returns the shortest path from source to target in a weighted graph G.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node

    target : node
       Ending node

    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight

    Returns
    -------
    path : list
       List of nodes in a shortest path.

    Raises
    ------
    NetworkXNoPath
       If no path exists between source and target.

    Examples
    --------
    >>> G=nx.path_graph(5)
    >>> print(nx.dijkstra_path(G,0,4))
    [0, 1, 2, 3, 4]

    Notes
    ------
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    See Also
    --------
    bidirectional_dijkstra()
    """
    (length,path)=single_source_dijkstra(G, source, target=target,
                                         weight=weight,jump=jump)
    try:
        return path[target]
    except KeyError:
        raise nx.NetworkXNoPath("node %s not reachable from %s"%(source,target))


def dijkstra_path_length(G, source, target, weight='weight',jump= 'jump'):
    """Returns the shortest path length from source to target
    in a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
       starting node for path

    target : node label
       ending node for path

    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight

    Returns
    -------
    length : number
        Shortest path length.

    Raises
    ------
    NetworkXNoPath
        If no path exists between source and target.

    Examples
    --------
    >>> G=nx.path_graph(5)
    >>> print(nx.dijkstra_path_length(G,0,4))
    4

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    See Also
    --------
    bidirectional_dijkstra()
    """
    length=single_source_dijkstra_path_length(G, source, weight=weight,jump= jump)
    try:
        return length[target]
    except KeyError:
        raise nx.NetworkXNoPath("node %s not reachable from %s"%(source,target))


def single_source_dijkstra_path(G,source, cutoff=None, weight='weight',jump= 'jump'):
    """Compute shortest path between source and all other reachable
    nodes for a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path.

    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight

    cutoff : integer or float, optional
       Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
    paths : dictionary
       Dictionary of shortest path lengths keyed by target.

    Examples
    --------
    >>> G=nx.path_graph(5)
    >>> path=nx.single_source_dijkstra_path(G,0)
    >>> path[4]
    [0, 1, 2, 3, 4]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    See Also
    --------
    single_source_dijkstra()

    """
    (length,path)=single_source_dijkstra(G,source, weight = weight,jump= jump)
    return path


def single_source_dijkstra_path_length(G, source, cutoff= None,
                                       weight= 'weight',jump= 'jump'):
    """Compute the shortest path length between source and all other
    reachable nodes for a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
       Starting node for path

    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight.

    cutoff : integer or float, optional
       Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
    length : dictionary
       Dictionary of shortest lengths keyed by target.

    Examples
    --------
    >>> G=nx.path_graph(5)
    >>> length=nx.single_source_dijkstra_path_length(G,0)
    >>> length[4]
    4
    >>> print(length)
    {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    See Also
    --------
    single_source_dijkstra()

    """
    dist = {}  # dictionary of final distances
    jumq={}
    seen = {source:0}
    fringe=[] # use heapq with (distance,label) tuples
    heapq.heappush(fringe,(0,source,0))
    while fringe:
        (d,v,j)=heapq.heappop(fringe)
        if v in dist:
            continue # already searched this node.
        dist[v] = d
        jumq[v] = j#jumq[v]+vw_jumq
        #for ignore,w,edgedata in G.edges_iter(v,data=True):
        #is about 30% slower than the following
        if G.is_multigraph():
            edata=[]
            for w,keydata in G[v].items():
                minweight=min((dd.get(weight,1)
                               for k,dd in keydata.items()))
                edata.append((w,{weight:minweight}))
        else:
            edata=iter(G[v].items())

        for w,edgedata in edata:
            vw_jumq = jumq[v] + edgedata.get(jump,1)
            ddist=edgedata.get(weight,1)
            vw_dist = dist[v] + ddist
            if(vw_dist<9999.0):
                if(int(vw_jumq)>1 or int(vw_jumq%1.0*1000.0+0.5)>1):
                    ddist=9999.0
            vw_dist = dist[v] + ddist

            if cutoff is not None:
                if vw_dist>cutoff:
                    continue
            if w in dist:
                if vw_dist < dist[w]:
                    raise ValueError('Contradictory paths found:',
                                     'negative weights?')
            elif w not in seen or vw_dist < seen[w]:
                seen[w] = vw_dist
                heapq.heappush(fringe,(vw_dist,w,vw_jumq))
    return dist,jumq


def single_source_dijkstra(G,source,target=None,cutoff=None,weight='weight',jump='jump'):
    """Compute shortest paths and lengths in a weighted graph G.

    Uses Dijkstra's algorithm for shortest paths.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
       Starting node for path

    target : node label, optional
       Ending node for path

    cutoff : integer or float, optional
       Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
    distance,path : dictionaries
       Returns a tuple of two dictionaries keyed by node.
       The first dictionary stores distance from the source.
       The second stores the path from the source to that node.


    Examples
    --------
    >>> G=nx.path_graph(5)
    >>> length,path=nx.single_source_dijkstra(G,0)
    >>> print(length[4])
    4
    >>> print(length)
    {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    >>> path[4]
    [0, 1, 2, 3, 4]

    Notes
    ---------
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    Based on the Python cookbook recipe (119466) at
    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/119466

    This algorithm is not guaranteed to work if edge weights
    are negative or are floating point numbers
    (overflows and roundoff errors can cause problems).

    See Also
    --------
    single_source_dijkstra_path()
    single_source_dijkstra_path_length()
    """
    if source==target:
        return ({source:0}, {source:[source]})
    dist = {}  # dictionary of final distances
    paths = {source:[source]}  # dictionary of paths
    seen = {source:0}
    fringe=[] # use heapq with (distance,label) tuples
    heapq.heappush(fringe,(0,source))
    while fringe:
        (d,v)=heapq.heappop(fringe)
        if v in dist:
            continue # already searched this node.
        dist[v] = d
        if v == target:
            break
        #for ignore,w,edgedata in G.edges_iter(v,data=True):
        #is about 30% slower than the following
        if G.is_multigraph():
            edata=[]
            for w,keydata in G[v].items():
                minweight=min((dd.get(weight,1)
                               for k,dd in keydata.items()))
                edata.append((w,{weight:minweight}))
        else:
            edata=iter(G[v].items())

        for w,edgedata in edata:
            vw_dist = dist[v] + edgedata.get(weight,1)
            if cutoff is not None:
                if vw_dist>cutoff:
                    continue
            if w in dist:
                if vw_dist < dist[w]:
                    raise ValueError('Contradictory paths found:',
                                     'negative weights?')
            elif w not in seen or vw_dist < seen[w]:
                seen[w] = vw_dist
                heapq.heappush(fringe,(vw_dist,w))
                paths[w] = paths[v]+[w]
    return (dist,paths)


def dijkstra_predecessor_and_distance(G,source, cutoff=None, weight='weight'):
    """Compute shortest path length and predecessors on shortest paths
    in weighted graphs.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
       Starting node for path

    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight

    cutoff : integer or float, optional
       Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
    pred,distance : dictionaries
       Returns two dictionaries representing a list of predecessors
       of a node and the distance to each node.

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The list of predecessors contains more than one element only when
    there are more than one shortest paths to the key node.
    """
    push=heapq.heappush
    pop=heapq.heappop
    dist = {}  # dictionary of final distances
    pred = {source:[]}  # dictionary of predecessors
    seen = {source:0}
    fringe=[] # use heapq with (distance,label) tuples
    push(fringe,(0,source))
    while fringe:
        (d,v)=pop(fringe)
        if v in dist: continue # already searched this node.
        dist[v] = d
        if G.is_multigraph():
            edata=[]
            for w,keydata in G[v].items():
                minweight=min((dd.get(weight,1)
                               for k,dd in keydata.items()))
                edata.append((w,{weight:minweight}))
        else:
            edata=iter(G[v].items())
        for w,edgedata in edata:
            vw_dist = dist[v] + edgedata.get(weight,1)
            if cutoff is not None:
                if vw_dist>cutoff:
                    continue
            if w in dist:
                if vw_dist < dist[w]:
                    raise ValueError('Contradictory paths found:',
                                     'negative weights?')
            elif w not in seen or vw_dist < seen[w]:
                seen[w] = vw_dist
                push(fringe,(vw_dist,w))
                pred[w] = [v]
            elif vw_dist==seen[w]:
                pred[w].append(v)
    return (pred,dist)


def all_pairs_dijkstra_path_length(G, cutoff=None, weight='weight',jump= 'jump'):
    """ Compute shortest path lengths between all nodes in a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight

    cutoff : integer or float, optional
       Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
    distance : dictionary
       Dictionary, keyed by source and target, of shortest path lengths.

    Examples
    --------
    >>> G=nx.path_graph(5)
    >>> length=nx.all_pairs_dijkstra_path_length(G)
    >>> print(length[1][4])
    3
    >>> length[1]
    {0: 1, 1: 0, 2: 1, 3: 2, 4: 3}

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The dictionary returned only has keys for reachable node pairs.
    """
    paths={}
    for n in G:
        paths[n]=single_source_dijkstra_path_length(G,n, cutoff=cutoff,
                                                    weight=weight,jump=jump)
    return paths

def all_pairs_dijkstra_path(G, cutoff=None, weight='weight',jump='jump'):
    """ Compute shortest paths between all nodes in a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight

    cutoff : integer or float, optional
       Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
    distance : dictionary
       Dictionary, keyed by source and target, of shortest paths.

    Examples
    --------
    >>> G=nx.path_graph(5)
    >>> path=nx.all_pairs_dijkstra_path(G)
    >>> print(path[0][4])
    [0, 1, 2, 3, 4]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    See Also
    --------
    floyd_warshall()

    """
    paths={}
    for n in G:
        paths[n]=single_source_dijkstra_path(G, n, cutoff=cutoff,
                                             weight=weight,jump=jump)
    return paths

def bellman_ford(G, source, weight = 'weight'):
    """Compute shortest path lengths and predecessors on shortest paths
    in weighted graphs.

    The algorithm has a running time of O(mn) where n is the number of
    nodes and m is the number of edges.  It is slower than Dijkstra but
    can handle negative edge weights.

    Parameters
    ----------
    G : NetworkX graph
       The algorithm works for all types of graphs, including directed
       graphs and multigraphs.

    source: node label
       Starting node for path

    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight

    Returns
    -------
    pred, dist : dictionaries
       Returns two dictionaries keyed by node to predecessor in the
       path and to the distance from the source respectively.

    Raises
    ------
    NetworkXUnbounded
       If the (di)graph contains a negative cost (di)cycle, the
       algorithm raises an exception to indicate the presence of the
       negative cost (di)cycle.  Note: any negative weight edge in an
       undirected graph is a negative cost cycle.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.path_graph(5, create_using = nx.DiGraph())
    >>> pred, dist = nx.bellman_ford(G, 0)
    >>> pred
    {0: None, 1: 0, 2: 1, 3: 2, 4: 3}
    >>> dist
    {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

    >>> from nose.tools import assert_raises
    >>> G = nx.cycle_graph(5, create_using = nx.DiGraph())
    >>> G[1][2]['weight'] = -7
    >>> assert_raises(nx.NetworkXUnbounded, nx.bellman_ford, G, 0)

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The dictionaries returned only have keys for nodes reachable from
    the source.

    In the case where the (di)graph is not connected, if a component
    not containing the source contains a negative cost (di)cycle, it
    will not be detected.

    """
    if source not in G:
        raise KeyError("Node %s is not found in the graph"%source)
    numb_nodes = len(G)

    dist = {source: 0}
    pred = {source: None}

    if numb_nodes == 1:
       return pred, dist

    if G.is_multigraph():
        def get_weight(edge_dict):
            return min([eattr.get(weight,1) for eattr in edge_dict.values()])
    else:
        def get_weight(edge_dict):
            return edge_dict.get(weight,1)

    for i in range(numb_nodes):
        no_changes=True
        # Only need edges from nodes in dist b/c all others have dist==inf
        for u, dist_u in list(dist.items()): # get all edges from nodes in dist
            for v, edict in G[u].items():  # double loop handles undirected too
                dist_v = dist_u + get_weight(edict)
                if v not in dist or dist[v] > dist_v:
                    dist[v] = dist_v
                    pred[v] = u
                    no_changes = False
        if no_changes:
            break
    else:
        raise nx.NetworkXUnbounded("Negative cost cycle detected.")
    return pred, dist

def negative_edge_cycle(G, weight = 'weight'):
    """Return True if there exists a negative edge cycle anywhere in G.

    Parameters
    ----------
    G : NetworkX graph

    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight

    Returns
    -------
    negative_cycle : bool
        True if a negative edge cycle exists, otherwise False.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.cycle_graph(5, create_using = nx.DiGraph())
    >>> print(nx.negative_edge_cycle(G))
    False
    >>> G[1][2]['weight'] = -7
    >>> print(nx.negative_edge_cycle(G))
    True

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    This algorithm uses bellman_ford() but finds negative cycles
    on any component by first adding a new node connected to
    every node, and starting bellman_ford on that node.  It then
    removes that extra node.
    """
    newnode = generate_unique_node()
    G.add_edges_from([ (newnode,n) for n in G])

    try:
        bellman_ford(G, newnode, weight)
    except nx.NetworkXUnbounded:
        G.remove_node(newnode)
        return True
    G.remove_node(newnode)
    return False


def bidirectional_dijkstra(G, source, target, weight = 'weight'):
    """Dijkstra's algorithm for shortest paths using bidirectional search.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node.

    target : node
       Ending node.

    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight

    Returns
    -------
    length : number
        Shortest path length.

    Returns a tuple of two dictionaries keyed by node.
    The first dictionary stores distance from the source.
    The second stores the path from the source to that node.

    Raises
    ------
    NetworkXNoPath
        If no path exists between source and target.

    Examples
    --------
    >>> G=nx.path_graph(5)
    >>> length,path=nx.bidirectional_dijkstra(G,0,4)
    >>> print(length)
    4
    >>> print(path)
    [0, 1, 2, 3, 4]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    In practice  bidirectional Dijkstra is much more than twice as fast as
    ordinary Dijkstra.

    Ordinary Dijkstra expands nodes in a sphere-like manner from the
    source. The radius of this sphere will eventually be the length
    of the shortest path. Bidirectional Dijkstra will expand nodes
    from both the source and the target, making two spheres of half
    this radius. Volume of the first sphere is pi*r*r while the
    others are 2*pi*r/2*r/2, making up half the volume.

    This algorithm is not guaranteed to work if edge weights
    are negative or are floating point numbers
    (overflows and roundoff errors can cause problems).

    See Also
    --------
    shortest_path
    shortest_path_length
    """
    if source == target: return (0, [source])
    #Init:   Forward             Backward
    dists =  [{},                {}]# dictionary of final distances
    paths =  [{source:[source]}, {target:[target]}] # dictionary of paths
    fringe = [[],                []] #heap of (distance, node) tuples for extracting next node to expand
    seen =   [{source:0},        {target:0} ]#dictionary of distances to nodes seen
    #initialize fringe heap
    heapq.heappush(fringe[0], (0, source))
    heapq.heappush(fringe[1], (0, target))
    #neighs for extracting correct neighbor information
    if G.is_directed():
        neighs = [G.successors_iter, G.predecessors_iter]
    else:
        neighs = [G.neighbors_iter, G.neighbors_iter]
    #variables to hold shortest discovered path
    #finaldist = 1e30000
    finalpath = []
    dir = 1
    while fringe[0] and fringe[1]:
        # choose direction
        # dir == 0 is forward direction and dir == 1 is back
        dir = 1-dir
        # extract closest to expand
        (dist, v )= heapq.heappop(fringe[dir])
        if v in dists[dir]:
            # Shortest path to v has already been found
            continue
        # update distance
        dists[dir][v] = dist #equal to seen[dir][v]
        if v in dists[1-dir]:
            # if we have scanned v in both directions we are done
            # we have now discovered the shortest path
            return (finaldist,finalpath)

        for w in neighs[dir](v):
            if(dir==0): #forward
                if G.is_multigraph():
                    minweight=min((dd.get(weight,1)
                               for k,dd in G[v][w].items()))
                else:
                    minweight=G[v][w].get(weight,1)
                vwLength = dists[dir][v] + minweight #G[v][w].get(weight,1)
            else: #back, must remember to change v,w->w,v
                if G.is_multigraph():
                    minweight=min((dd.get(weight,1)
                               for k,dd in G[w][v].items()))
                else:
                    minweight=G[w][v].get(weight,1)
                vwLength = dists[dir][v] + minweight #G[w][v].get(weight,1)

            if w in dists[dir]:
                if vwLength < dists[dir][w]:
                    raise ValueError("Contradictory paths found: negative weights?")
            elif w not in seen[dir] or vwLength < seen[dir][w]:
                # relaxing
                seen[dir][w] = vwLength
                heapq.heappush(fringe[dir], (vwLength,w))
                paths[dir][w] = paths[dir][v]+[w]
                if w in seen[0] and w in seen[1]:
                    #see if this path is better than than the already
                    #discovered shortest path
                    totaldist = seen[0][w] + seen[1][w]
                    if finalpath == [] or finaldist > totaldist:
                        finaldist = totaldist
                        revpath = paths[1][w][:]
                        revpath.reverse()
                        finalpath = paths[0][w] + revpath[1:]
    raise nx.NetworkXNoPath("No path between %s and %s." % (source, target))
