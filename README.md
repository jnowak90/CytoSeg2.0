# CytoSeg2.0
Fiji Macro and GUI for CytoSeg to automatically extract and analyze the actin cytoskeleton from microscopy images. Please cite the following paper if you use the tool:

   Breuer, D., Nowak, J., Ivakov, A., Somssich, M., Persson, S. and Nikoloski, Z.
   System-wide organization of actin cytoskeleton determines organelle transport in hypocotyl plant cells.
   *PNAS*, **2017**, 114: E5741-5749.
   
## Contents
 - [Requirements](#requirements)
 - [Installation](#installation)
 - [Workflow](#workflow)
 - [Output](#output)
 - [Demo](#demo)
 - [Troubleshooting](#troubleshooting)


## Requirements

- Only tif files are supported.
- The usage of image stacks is preferred, although using single images is possible, too (but will raise a warning).
- The following plugins have to be installed in Fiji (in the mentioned order):
  - TurboReg (http://bigwww.epfl.ch/thevenaz/turboreg/)
  - StackReg (http://bigwww.epfl.ch/thevenaz/stackreg/)
  
  To install, extract the downloaded files, then install the plugins with **Fiji > Plugins > Install PlugIn...** and selecting the corresponding *\*.jar* file. Restart Fiji after installing.
- An installation of Python 3 is required. Following modules have to be installed (used versions during development in *brackets*):
  - numpy (*1.14.0*)
  - scipy (*1.2.1*)
  - matplotlib (*2.02*) 
  - scikit-image (*0.18.2*)
  - PIL (*5.0.0*)
  - networkx (*2.1*)
  - pandas (*0.20.3*)
  - shapely (*1.5.17*)
  - packaging (*16.8*)
  - tkinter (*8.5*)

## Installation

1. Make sure all required Fiji plugins and Python 3 modules are installed (see Requirements).

2. Download the zip file and decompress. Rename the decompressed folder to "CytoSeg".

3. Copy "CytoSeg" folder to the plugins folder of the Fiji directory (Fiji.app).

4. Start Fiji. If Fiji was already open, restart it.

5. The macro should be now in Plugins > CytoSeg (at the bottom).


## Workflow

### Getting started 
When you first start the plugin, it will prompt you to input the path to your Python 3 (Fiji will otherwise use the system  version of Python 2). You can find the Python 3 path by typing "which python3" in your terminal (Mac OS, Linux). If you use Windows search for python.exe with the Search button and open and copy the file location. Press "OK" to continue. Your Python 3 path will be saved for future sessions. You can change it by selecting "Reset Python3 path" in the CytoSeg2.0 main menu. After you input the Python path, the plugin will check if all necessary Fiji plugins and Python modules are installed.

### Analysis 
You can choose whether to do a complete CytoSeg analysis or a specific step in the analysis.

#### Complete Analysis
If you selected to do the complete analysis, you will be guided trough different steps:
      
*Pre-processing:*

   Select a name for the output folder, the current date will be suggested otherwise for the output folder. You can also decide whether to continue in silent mode, which will repress the intermediate pre-processing steps (default). Furthermore, you can decide if you want to analyse a single image or multiple images, which you have to select afterwards. Fiji will start pre-processing and you have to select the region of interset (ROI) for all images and click 'OK'. If you turned of the silent mode, select the ROI from the top image (*MAX_\*_.tif*). The pre-processed image and the mask with the selected ROI will be saved in the output folder (as '\*\_filter.tif' and '\*\_mask.tif', respectively, see Output).
 
*Gauging:*

   The GUI will prompt you to select an image for parameter gauging if you analyze multiple images. If you chose to analyze a single image, this image will be used for gauging. In the gauging step, the optimal parameters for the segmentation of the cytoskeleton are determined by opening the gauging window. Press "Open Image" to see your selected image. Drag the parameter sliders (v<sub>width</sub>, v<sub>thres</sub>, v<sub>size</sub>, v<sub>int</sub>) to see the segmentation results (that might take some seconds). If you are satisfied with the segmentation, press "Choose Parameters" and your chosen parameters will be saved for the extraction process. Click "Back to Main Menu" to continue. Open another image with the "Open Image" button, which will immediately show the extracted skeleton using the last chosen set of parameters.
   
    
*Extraction:*

   The extraction is done in Python 3. The pre-processed image and the mask will be used to extract a network from the cytoskeleton of every slice in the image. Additionally, for every extracted network a user-defined number of random network will be generated (default=1). The extracted and random networks and their node positions will be saved in the output folder, as well as a plot of the first image slice and the overlayed extracted network (colored according to the edge capacity). Calculated  network properties are saved in tables for both extracted and random networks.
    
#### Selecte specific analysis steps
    
*Redraw mask:*

   Select if you want to draw or redraw the mask of a single or multiple images. Make your selection and press "OK". The mask will be saved in the same folder as the selected image. If the image folder contains other directories it can be chosen in which directory the mask should be saved. 
   
    
*Gauging:*

   The gauging GUI will open, where you have to select an image. Note that the gauging is only working if you have a mask for your image. If not, first draw a mask for that image (using "Select specific CytoSeg step" > "Redraw mask"). Once you selected your parameters, click "Choose Parameters" and return to the main menu. You can open another image with the "Open Image" button, which will immediately display the skeleton of the selected image using the last set of parameters. Please choose a pre-processed image for gauging (*\*\_filter.tif*) or the resulting skeleton might not match the segmentation results of the network extraction. 
    
    
*Pre-processing and extraction:*

   Select the name of your output folder and if you want to use already existing masks (applicable if you already ran this part before). You can also select the silent mode here and change the different parameters. If you selected parameters before during gauging, the parameters will be selected here automatically. Additionally, you can decide if you want to do both pre-processing and network extraction or only one of the processes.

## Output
The following outputs are generated when using the pre-processing and extraction pipeline (example outputs are shown in the DemoImages folder):

Pre-processing:
  - **\*\_filter.tif**: pre-processed image
  - **\*\_mask.tif**: mask of ROI for image
  
Network extraction:
  - **originalGraphs.gpickle**: collection of all extracted networks from the input image (one network per image slice)
  - **randomGraphs.gpickle**: collection of randomized networks for input image (one randomized network per extracted network)
  - **originalGraphPositions.npy**: node positions for original networks
  - **randomGraphPositions.npy**: node positions for random networks
  - **ExtractedNetworks.pdf**: plot of the original and randomized extracted network of the first image slice
  - **originalGraphProperties.csv**: table of graph properties for the original networks 
  - **randomGraphProperties.csv**: table of graph properties for the random networks
  
Parameter gauging:
  - **skeletonOnImage.png**: image of resulting segmentation from chosen gauging parameters
 Furthermore, the Python3 path, selected gauging parameters and the log file of each session are saved in the **Fiji.app > plugins > CytoSeg** folder.

## Demo
The DemoImages folder contains example image that can be used to test the plugin. 
The Extraction folder contains two .tif images of actin filaments under control and LatB treatment. The images can be used for the complete analysis, as well as for the mask redrawing and the pre-processing and extraction.
The Gauging folder contains a .tif image of the actin cytoskeleton and a mask for the ROI. The images can be used for the gauging and all the steps mentioned above.
The ExampleOutput folder contains the expected output after extraction the networks as described in Output. The output can be further analyzed in Python 3, as shown in the following.

```python
import numpy as np
import pandas as pd
import networkx as nx
import pickle
import matplotlib.pyplot as plt

# import the extracted networks
originalGraphs = pickle.load(open('originalGraphs.gpickle', 'rb'))

# import node positions of networks
originalPositions = np.load('originalGraphPositions.npy')

# plot first graph 
graph, nodePositions = originalGraphs[0], originalPositions[0]
edgeCapacity = 1.0 * np.array([property['capa'] for node1, node2, property in graph.edges(data=True)])

fig, ax = plt.subplots(1, 1, figsize=(3,3))
nx.draw_networkx(graph, nodePositions, with_labels=False, node_size=0, edge_color=plt.cm.plasma(edgeCapacity/ edgeCapacity.max()))
plt.show()

# import network property table
properties = pd.read_csv('originalGraphProperties.csv', sep=';')
properties.head()
```

## Troubleshooting
In case of errors, here are some suggestions on how to fix them.

### The plugin works, but the gauging GUI is not opening and the extraction is not working
Check if you added the correct Python path when prompted at the beginning. You can test if your python path is working by opening a terminal (Linux, MacOs) or the CMD (Windows) and typing the following:
```bash
YOURPYTHONPATH -c "print('Hello World')"
```
YOURPYTHONPATH is the path you copied into the plugin at the beginning. If you don't get an output (Hello World is printed in the terminal/CMD), your python path is wrong.

### StackReg or TurboReg raise an error
Check if you correctly installed the plugins. You should find TurboReg and StackReg in the Fiji Plugins Menu. MultiStackReg should be in Plugins > Registration. If you can't find the plugins there, the installation didn't work. To install the plugins correctly, download them from the links in Requirements and decompress. Then go to Fiji > Plugins > Install Plugin... and choose the corresponding .jar file of the plugin. Restart Fiji to see if the plugin was installed.

### All required plugins are installed, but StackReg still raises an error
Try to install StackReg from: https://sites.imagej.net/BIG-EPFL/plugins/

### The gauging GUI opened the image, but nothing happens when moving the sliders
Make sure you created a mask (\*\_mask.tif) for the selected image (\*\_filter.tif) that is in the same folder. If the mask is missing an Error message will be shown. You can create a mask by choosing ""Select specific CytoSeg step" > "Redraw mask".

### The plugin raises an error: java.io.FileNotFoundException (Mac OS)
Move the Fiji application into the Applications folder.
