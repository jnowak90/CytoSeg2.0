# CytoSeg2.0
Fiji Macro and GUI for CytoSeg to automatically extract and analyze the actin cytoskeleton from microscopy images. Please cite the following paper if you use the tool:

   Breuer, D., Nowak, J., Ivakov, A., Somssich, M., Persson, S. and Nikoloski, Z.
   System-wide organization of actin cytoskeleton determines organelle transport in hypocotyl plant cells.
   *PNAS*, **2017**, 114: E5741-5749.

### Requirements

- Only tif files are supported.
- Only image stacks are supported, single images will raise an error.
- The following plugins have to be installed in Fiji (in the mentioned order):
  - StackReg (http://bigwww.epfl.ch/thevenaz/stackreg/)
  - TurboReg (http://bigwww.epfl.ch/thevenaz/turboreg/)
  - MultiStackReg (http://bradbusse.net/sciencedownloads.html)
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

### Installation

1. Make sure all required Fiji plugins and Python 3 modules are installed (see Requirements).

2. Download the zip file and decompress. Rename the decompressed folder to "CytoSeg".

3. Copy "CytoSeg" folder to the plugins folder of the Fiji directory (Fiji.app).

4. Start Fiji. If Fiji was already open, restart it.

5. The macro should be now in Plugins > CytoSeg.

### Workflow

#### Getting started 
If you first start the plugin, it will prompt you to input the path to your Python 3 (Fiji will otherwise use the system  version of Python). You can find the Python 3 path by typing "which python3" in your terminal (Mac OS, Linux). If you use Windows search for python.exe with the Search button and open and copy the file location. Press "OK" to continue. Your Python 3 path will be saved for future sessions. You can change it by selecting "Reset Python3 path" in the CytoSeg2.0 main menu.

#### Analysis 
You can choose whether to do a complete CytoSeg analysis or a specific step in the analysis.

##### Complete Analysis
If you selected to do the complete analysis, you will be guided trough different steps:
  
*Gauging:*

   The GUI will prompt you to select an image for parameter gauging. In the gauging step the optimal parameters for the segmentation of the cytoskeleton are determined. Press "OK" to select your image for the gauging. You have to select the region of interest (ROI), here you can redraw your selection as often as you like. Press "OK" when you are done. The gauging window will appear. Press "Open Image" to see your selected image. Drag the parameter sliders (sigma, block, small, factr) to see the segmentation results (that might take some seconds). If you are satisfied with the segmentation, press "Choose Parameters" and your chosen parameters will be saved for the extraction process. Click "Back to Main Menu" to continue.
      
*Pre-processing:*

   Select a name for the output folder, a folder name with the current date will be suggested otherwise. You can also decide whether to continue in silent mode, will repress the intermediate pre-processing steps. In the next window you can decide if you want to analyse a single image or multiple images, which you have to select afterwards. Fiji will start pre-processing and you have to select the ROI for all images you selected. The pre-processed image and the mask with the selected ROI will be saved in the output folder (as '*_filter.tif' and '*_mask.tif', respectively).
    
*Extraction:*

   The extraction is done in Python 3. The pre-processed image and the mask will be used to extract a network from the cytoskeleton of every slice in the image. Additionally, for every extracted network a random network will be generated. The extracted and random networks and their node positions will be saved in the output folder, as well as a plot of the first image slice and the overlayed extracted network (colored according to the edge capacity). Calculated  network properties are saved in tables for both extracted and random networks.
    
##### Selecte specific analysis steps
    
*Gauging:*

   The gauging GUI will open, where you have to select an image. Note that the gauging is only working, if you have a mask for your image. If not, first draw a mask for that image. Once you selected your parameters, click "Choose Parameters" and return to the main menu.
    
*Redraw mask:*

   Select if you want to draw or redraw the mask of a single or multiple images. Make your selection and press "OK". The mask will be saved in the same folder as the selected image. 
    
*Pre-processing and extraction:*

   Select the name of your output folder and if you want to use already existing masks (applicable if you already ran this part before). You can also select the silent mode here and the different parameters. If you selected parameters before during gauging, the parameters will be selected here automatically. 
  
### Output
The following outputs are generated when using the pre-processing and extraction pipeline:
  - **\*\_filter.tif**: pre-processed image
  - **\*\_mask.tif**: mask of ROI for image
  - **originalGraphs.gpickle**: collection of all extracted networks from the input image (one network per image slice)
  - **randomGraphs.gpickle**: collection of randomized networks for input image (one randomized network per extracted network)
  - **originalGraphPositions.npy**: node positions for original networks
  - **randomGraphPositions.npy**: node positions for random networks
  - **ExtractedNetworks.pdf**: plot of the original and randomized extracted network of the first image slice
  - **originalGraphProperties.csv**: table of graph properties for the original networks 
  - **randomGraphProperties.csv**: table of graph properties for the random networks
