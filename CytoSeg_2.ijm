////////// CytSeg GUI for Fiji  ////////////////
///////////////////////////////////////////////
/// https://github.com/jnowak90/CytoSeg2.0 ///
/////////////////////////////////////////////

// check the user's system
osSystem = getInfo("os.name");

/////////////////////
///// FUNCTIONS /////
// test if all neccessary Fiji plugins are installed
function systemTestFiji() {
	counter = 0;
	functions = newArray("StackReg ", "Bleach Correction", "TurboReg ", "8-bit", "Properties...", "Z Project...", 
	"Clear Outside", "Fill", "Measure", "Divide...", "Subtract Background...", "Despeckle", "Enhance Contrast...");

	List.setCommands;
	for (i=0; i<functions.length; i++) {
    	if (List.get(functions[i])!="") {
    		counter += 0;
        } else {
        	print('"' + functions[i] + '"' + ' command was not found. Please install the plugin before continuing.');
    		counter += 1;
    			    }
	}

	if (counter==0) {
		print("### All necessary Fiji functions were found.");
	}
}

// test Python 3 path and if all necessary Python modules are installed
function systemTestPython(osSystem, pathToPython3, pathToCytoSeg) {
	if (startsWith(osSystem, "Windows")) {
		pathToTestArray = getDirectory('plugins') + "CytoSeg\\TestArray.npy";
		pathToTest = getDirectory("plugins") + "CytoSeg\\SystemTest.py";
		exec("cmd /c " + pathToPython3 + " " + pathToTest + " " + pathToTestArray);
	
	} else {
		pathToTestArray = getDirectory('plugins') + "CytoSeg/TestArray.npy";
		pathToTest = getDirectory("plugins") + "CytoSeg/SystemTest.py";
		exec(pathToPython3, pathToTest, pathToTestArray);
	}

	if 	(File.exists(pathToTestArray)){
		ok = File.delete(pathToTestArray);
	} else {
		print("No working Pyhton 3 version was found. Please check your Python 3 path.");
		pathToPython3 = selectPythonPath(pathToCytoSeg);
		systemTestPython(osSystem, pathToPython3, pathToCytoSeg);
	}
}

// find minimum
function min(x, y) {
	if (x < y) {
		return(x);
	} else {
		return(y);
	}
}

// find maximum
function max(x, y) {
	if (x > y) {
		return(x);
	}else {
		return(y);
	}
}

// get index of element in array
function get_index(array, value) {
    for (i=0; i<array.length; i++) {
        if (endsWith(array[i], value)) {  
            return(i);
        } 
    }
} 

// check if file exists in path
function checkFileExists(file, path){
	fileList = getFileList(path);
	for (i=0; i<fileList.length; i++){
		if (file == fileList[i]){
			return(true);
		}
	}
	return(false);
}

// check if directories exist in path
function checkDirectoriesExist(path) {
	fileList = getFileList(path);
	directories = newArray(0);
	for (i=0; i<fileList.length; i++){
		if (endsWith(fileList[i], "/") || endsWith(fileList[i], "\\")){
			directories = Array.concat(directories, fileList[i]);
		}
	}
	directories = Array.concat(directories, "Current directory");
	return directories;
}

// input custom parameters
function createParameterDialog(){
	Dialog.create("CytoSeg - Settings");
	Dialog.addMessage("Please name your output folder \n(will be generated in the folder where your images are)");
	outputFolderName = "" + dayOfMonth + "-" + month + "-" + year;
	Dialog.addString("Folder", outputFolderName);
	Dialog.addMessage("");
	Dialog.addMessage("Do you want to use existing masks?");
	items = newArray("yes", "no");
	Dialog.addChoice("Choose", items, "no")
	Dialog.addMessage("");
	Dialog.addMessage("Do you want to proceed in silent mode?");
	Dialog.addChoice("Choose", items, "yes");
	Dialog.addMessage("");
	Dialog.addMessage("Parameters");
	Dialog.addNumber("rolling ball", roll);
	Dialog.addMessage("");
	//Dialog.addNumber("depth", depth);
	//Dialog.addMessage("");
	Dialog.addNumber("Vwidth", sigma);
	Dialog.addMessage("");
	Dialog.addNumber("Vthres", block);
	Dialog.addMessage("");
	Dialog.addNumber("Vsize", small);
	Dialog.addMessage("");
	Dialog.addNumber("Vint", factr);
	 html = "<html>"
    +"<h2>Parameter information</h2>"
    +"<p><b>output folder</b>:  The specified output folder is created in the same folder as the selected images for analysis. For each image a subfolder with the name of the image is created in this output folder.</p>"
    +"<p><b>masks</b>: If you want to reuse already created masks, choose <b color='red'>yes</b>.<br>If <b color='red'>yes</b> was selected, it will be checked if the masks exist in the specified output folder.</p>"
    +"<p><b>silent mode</b>: If you choose <b color='red'>yes</b>, all processes will be run in the background.</p>"
    +"<p><b>rolling ball</b>: Rolling ball size in pixel for background subtraction.</p>"
    +"<p><b>v<sub>width</sub></b>: Width of filamentous structures to enhance with a 2D tubeness filter.</p>"
    +"<p><b>v<sub>thres</sub></b>: Block size for adaptive median threshold.</p>"
    +"<p><b>v<sub>size</sub></b>: Size of small objects to be removed.</p>"
    +"<p><b>v<sub>int</sub></b>: Lowest average intensity of a filament.</p>";
	Dialog.addHelp(html);
	Dialog.show();
	//return outputName;
	}
// create new output folder
function createOutputFolder(path, outputFolder) {
	if(endsWith(path, "tif") || endsWith(path, "TIF") || endsWith(path, "tiff") || endsWith(path, "TIFF")){
		open(path);
		filename = getTitle();
		dir = replace(path, filename, "");
		/*//check if the output folder already exists
		outputNameNew = outputFolder + "/";
		if (checkFileExists(outputNameNew, dir)){
			Dialog.create("WARNING");
			Dialog.addMessage("NOTICE\nThe output folder you named already exists.\nPress OK to continue or CANCEL to start again");
			Dialog.show();
		}*/
		//creating Output directory
		pathOutput = dir + outputFolder;
		File.makeDirectory(pathOutput);
		filename = replaceFileFormat(filename);
		if (startsWith(osSystem, "Windows")) {
			pathOutputImages = pathOutput + "\\" + filename;
		} else {
			pathOutputImages = pathOutput + "/" + filename;
		}
		File.makeDirectory(pathOutputImages);
		return pathOutputImages;
	}
	else {
		Dialog.create("ERROR");
		Dialog.addMessage("Please select a .tif image");
		Dialog.show();
	}
}
// remove .tif ending from filename
function replaceFileFormat(filename) {
	newFilename =replace(filename, ".tif", "");
	newFilename =replace(newFilename, ".TIF", "");
	newFilename =replace(newFilename, ".tiff", "");
	newFilename =replace(newFilename, ".TIFF", "");
	return newFilename;
}
// generate mask of image ROI
function generateMask(pathToFilter, pathOutputMask){
	//open filter image
	setBatchMode(true);
	openImages = getList("image.titles");
	if (openImages.length == 0) {
		open(pathToFilter);
	} 
	getDimensions(lx, ly, lc, lz, lt);
	lzz = min(lz, lt);
	ltt = max(lz, lt);
	I = lc * lzz * ltt;
	run('8-bit');
	// use maximum projection of filter image to select ROI
	run('Properties...', 'channels=1 slices=1 frames=I unit=pixel pixel_width=1.0000 pixel_height=1.0000 voxel_depth=1.0000');
	run('Z Project...', 'projection=[Max Intensity]');
	// freehand tool to select ROI
	setBatchMode(false);
	setTool('Freehand');
	waitForUser("Select the area of interest FIRST. \nYou can redraw the mask as often as you like. \nWhen you are done,  press 'OK'.");
	setBatchMode(true);
	// prepare mask
	run('Clear Outside');                  
	run('Fill', 'slice')  ;                                               
	run('Measure');                                                       
	meanValue = getResult('Mean');
	run('Divide...', 'value=' + meanValue);
	// save mask
	run('8-bit');
	pathOutputMask = replace(pathOutputMask, "_filter", "");
	saveAs('Tiff', pathOutputMask);                                            
	// close open windows
	selectWindow('Results');                                
	run('Close');
	run('Close All');
}
// update parameters
function updateParameters() {
	if (startsWith(osSystem, "Windows")) {
		pathDefaultParameters = getDirectory("plugins") + "CytoSeg\\defaultParameter.txt";
	} else {
		pathDefaultParameters = getDirectory("plugins") + "CytoSeg/defaultParameter.txt";
	}
	if (File.exists(pathDefaultParameters)){
		parametersDefault = File.openAsString(pathDefaultParameters);
		parametersDefaultArray = split(parametersDefault, ",");
		roll = parametersDefaultArray[0];
		randw = parametersDefaultArray[1];
		randn = parametersDefaultArray[2];
		depth = parametersDefaultArray[3];
		sigma = parametersDefaultArray[4];
		block = parametersDefaultArray[5];
		small = parametersDefaultArray[6];
		factr = parametersDefaultArray[7];
	}
}
// pre-process image
function preprocessImage(filename, pathOutputImages) {
	print("Convert image to 8-bit");
	run('8-bit');
	getDimensions(lx, ly, lc, lz, lt);
	lzz = min(lz, lt);
	ltt = max(lz, lt);
	I = lc * lzz * ltt;
	print("Start stack registration");
	run('StackReg ', 'transformation=[Rigid Body]');
	run('8-bit');
	run('Properties...', 'channels=1 slices=1 frames=I unit=pixel pixel_width=1.0000 pixel_height=1.0000 voxel_depth=1.0000');
	print("Bleach correction and background subtraction");
	run('Bleach Correction', 'correction=[Simple Ratio] background=0');
	run('Subtract Background...', 'rolling=roll stack');
	run('Despeckle', 'stack');               
	run('Enhance Contrast...', 'saturated=0.5 process_all use');  
	newTitle = filename + "_filter";
	if (startsWith(osSystem, "Windows")) {
		outputFilter = pathOutputImages + "\\" + newTitle;
	} else {
		outputFilter = pathOutputImages + "/" + newTitle;
	}
	saveAs('Tiff', outputFilter);
	run('Close All');
}
// call a python script which is inside the fiji.app/plugins/Cyto_Seg folder (a bridge bash script needed to execute the python script in python3, otherwise fiji uses python2)
function pythonGraphAnalysis(pathToPython3, pathToFolder, parameters){
	if (startsWith(osSystem, "Windows")) {
		pathToPostprocessing = getDirectory("plugins") + "CytoSeg\\ExtractionPipeline.py";
		exec("cmd /c " + pathToPython3 + " " + pathToPostprocessing + " " + pathToFolder + " " + parameters + " 1");
	} else {
		pathToPlugin = getDirectory("plugins") + "CytoSeg/bridge.sh";
		pathToPostprocessing = getDirectory("plugins") + "CytoSeg/ExtractionPipeline.py";
		exec('sh', pathToPlugin, pathToPython3, pathToPostprocessing, pathToFolder ,parameters, "0");
	}
}

//user input for python path
function selectPythonPath(pathToCytoSeg) {
	Dialog.create("CytoSeg - Welcome");
	Dialog.addMessage("Welcome to CytoSeg Analysis. \nBefore you start using CytoSeg, please set your full Python 3 path.");
	Dialog.addMessage("Linux/Mac User: type 'which python3' in your Terminal and copy the path into the field below.\nYour path should look something like /home/myComputer/anaconda/bin/python3");
	Dialog.addMessage("Windows User: search for python.exe with the Search button, open the file location and copy the path in the field below. \nYour path should look something like C:\\Users\\myName\\AppData\\Local\\Programs\\Python\\Python37\\python.exe");
	Dialog.addString("Path:","");
	Dialog.show();
	pathToPython3 = Dialog.getString();
	pathToPythonPathFile = pathToCytoSeg + "python3path.txt";
	File.saveString(pathToPython3, pathToPythonPathFile);
	return pathToPython3;
}

/////////////////////
///// VARIABLES /////
getDateAndTime(year, month, dayOfWeek, dayOfMonth, hour, minute, second, msec);
var month = month + 1;    			// bug of the getDateAndTime function, month start with 0
var roll = 50;
var randw = 1;
var randn = 1;
var depth = 7.75;
var sigma = 2.0;
var block = 101.0;
var small = 25.0;
var factr = 0.5;
updateParameters();
if (startsWith(osSystem, "Windows")) {
	pathToGUI = getDirectory("plugins") + "CytoSeg\\gaugingGui.py";
	pathToCytoSeg = getDirectory("plugins") + "CytoSeg\\";
} else {
	pathToGUI = getDirectory("plugins") + "CytoSeg/gaugingGui.py";
	pathToBridge = getDirectory("plugins") + "CytoSeg/bridge.sh";
	pathToCytoSeg = getDirectory("plugins") + "CytoSeg/";
}

/////////////////////
///// GUI LOGIC /////

//// PYTHONPATH: set the path to Python3 if not done yet and save in file
if (startsWith(osSystem, "Windows")) {
	pathToPython3TXT = getDirectory("plugins") + "CytoSeg\\python3path.txt";
} else {
	pathToPython3TXT = getDirectory("plugins") + "CytoSeg/python3path.txt";
}
if (!(File.exists(pathToPython3TXT))) {
	pathToPython3 = selectPythonPath(pathToCytoSeg);
	systemTestFiji();
	systemTestPython(osSystem, pathToPython3, pathToCytoSeg);
} else {
	pathToPython3 = File.openAsString(pathToPython3TXT);
}

//// MENU: choose mode
function mainMenu() {	
	Dialog.create("CytoSeg - Menu");
	Dialog.addMessage("Welcome to CytoSeg Analysis \n");
	items = newArray("Complete CytoSeg analysis", "Select specific CytoSeg step", "Reset Python 3 path");
	Dialog.addRadioButtonGroup("What do you want to do?", items, 3, 1, "yes");
	Dialog.show();
	choiceMenu = Dialog.getRadioButton();

	// MODE: complete CytoSeg analysis
	if(choiceMenu == "Complete CytoSeg analysis"){
		// pre-processing
		Dialog.create("CytoSeg - Settings");
		Dialog.addMessage("Please name your Outputfolder \n(will be created in the folder where your images are).");
		outputFolderName = "" + dayOfMonth + "-" + month + "-" + year;
		Dialog.addString("Folder", outputFolderName);
		Dialog.addMessage("Do you want to proceed in silent mode?");
		items = newArray("yes","no");
		Dialog.addChoice("Choose", items, "yes");
		Dialog.addMessage("Do you want to process a single image or all images in a folder ?");
		items = newArray("single","all");
		Dialog.addChoice("process:", items, "all");
		Dialog.show();
		outputFolder = Dialog.getString();
		choiceBatch = Dialog.getChoice();
		choiceImage = Dialog.getChoice();
		if (choiceBatch == "yes"){
			setBatchMode(true);
		}		
		// single image
		if (choiceImage == "single"){
			pathToImage = File.openDialog("Select .tif Image");
			pathOutputImages = createOutputFolder(pathToImage, outputFolder);
			print("\n#########################################\n\nCyto Seg is running...\n");
			filename = getTitle();
			newFilename = replaceFileFormat(filename);
			preprocessImage(newFilename, pathOutputImages);
			if (startsWith(osSystem, "Windows")) {
				generateMask(filename, pathOutputImages + "\\" + newFilename + "_mask.tif");
			} else {
				generateMask(filename, pathOutputImages + "/" + newFilename + "_mask.tif");
			}
			Dialog.create("CytoSeg - Gauging");
			Dialog.addMessage("Before starting the network extraction, the right parameters have to be selected for the images.");
			Dialog.show();
			if (startsWith(osSystem, "Windows")) {
				exec("cmd /c " + pathToPython3 + " " + pathToGUI + " " + pathToImage + " 1");
			} else {
				exec("sh", pathToBridge, pathToPython3, pathToGUI, pathToImage, "0");
			}
			updateParameters();
			parameters = "" + randw + "," + randn + "," + depth + "," + sigma + "," + block + "," + small + "," + factr;
			print("Used parameters for network extraction: \nv_width: " + sigma + "\nv_thres: " + block + "\nv_size: " + small + "\nv_int: " + factr);
			print("\n#########################################\n\nPre-processing finished, post-processing in progress...\n");
			// network extraction
			pythonGraphAnalysis(pathToPython3, pathOutputImages, parameters);
			print("\n#########################################\n\nAnalysis done.\n");	
		}
		// multiple images
		else {
			pathToImageFolder = getDirectory("Choose a Directory");
			imageList = getFileList(pathToImageFolder);
			counter = 0;
			extractionArray = newArray(0);
			filenameArray = newArray(0);
			print("\n#########################################\n\nCyto Seg is running...\n");
			for (i=0; i<imageList.length; i++) {
				if(endsWith(imageList[i],"tif") || endsWith(imageList[i], "TIF") || endsWith(imageList[i], "tiff") || endsWith(imageList[i], "TIFF")){
					if(endsWith(imageList[i],"_mask.tif") || endsWith(imageList[i], "_mask.TIF") || endsWith(imageList[i], "_mask.tiff") || endsWith(imageList[i], "_mask.TIFF")){
						print("A mask image was skipped.");
						counter += 1;
					} else {
						pathOutputImages = createOutputFolder(pathToImageFolder + imageList[i], outputFolder);
						print("\nFinshed pre-processing of image " + i + 1 + " of " + imageList.length + "...\n");
						filename = replaceFileFormat(imageList[i]);
						preprocessImage(filename, pathOutputImages);
						if (startsWith(osSystem, "Windows")) {
							generateMask(imageList[i], pathOutputImages + "\\" + filename + "_mask.tif");
						} else {
							generateMask(imageList[i], pathOutputImages + "/" + filename + "_mask.tif");
						}
						array = newArray(pathOutputImages);
						extractionArray = Array.concat(extractionArray, array);
						filenameArray = Array.concat(filenameArray, filename);
					}
				}
				else {					
					print("A file that is not a .tif file was skipped.");
					counter += 1;
				}
			}
			if (imageList.length == counter) {
				Dialog.create("WARNING");
				Dialog.addMessage("Non of the files inside the selected folder \ncontained tif images to process.");
				Dialog.show();
			}
			Dialog.create("CytoSeg - Gauging");
			Dialog.addMessage("Before starting the network extraction, the right parameters have to be selected for the images. \nPlease select an image for the gauging of those parameters.");
			Dialog.addChoice("Choose", filenameArray, filenameArray[0]);
			Dialog.show();
			filenameChoice = Dialog.getChoice();  
			indexChoice = get_index(extractionArray, filenameChoice);
			pathToFile = extractionArray[indexChoice];
			if (startsWith(osSystem, "Windows")) {
				exec("cmd /c " + pathToPython3 + " " + pathToGUI + " " + pathToFile + " 1");
			} else {
				exec("sh", pathToBridge, pathToPython3, pathToGUI, pathToFile, "0");
			}
			updateParameters();
			parameters = "" + randw + "," + randn + "," + depth + "," + sigma + "," + block + "," + small + "," + factr;
			print("Used parameters for network extraction: \nv_width: " + sigma + "\nv_thres: " + block + "\nv_size: " + small + "\nv_int: " + factr);
			print("\n#########################################\n\nPre-processing finished, post-processing in progress...\n");
			// network extraction
			for (j=0; j<extractionArray.length; j++){
				pythonGraphAnalysis(pathToPython3, extractionArray[j], parameters);
			}
			print("\n#########################################\n\nAnalysis done.\n");
		}
	}		
	
	// MODE: select specific CytoSeg step
	if(choiceMenu == "Select specific CytoSeg step"){
		Dialog.create("CytoSeg - Settings");
		Dialog.addMessage("Select the step you want to repeat.\n");
		items = newArray("Redraw mask", "Parameter gauging", "Pre-processing and extraction");
		Dialog.addRadioButtonGroup("What do you want to do?", items, 3, 1, "yes");
		Dialog.show();
		choiceStep = Dialog.getRadioButton();

		// redrawing mask
		if (choiceStep == "Redraw mask") {
			Dialog.create("CytoSeg - Settings");
			Dialog.addMessage("Do you want to process a single image or all images in a folder ?");
			items = newArray("single", "all");
			Dialog.addChoice("process:", items, "single");
			Dialog.show();
			choiceData = Dialog.getChoice();
			if (choiceData == "single") {
				pathToImage = File.openDialog("Select .tif Image");
				open(pathToImage);
				filename = getTitle();
				newFilename = replaceFileFormat(filename);
				folder = replace(pathToImage, filename, "");
				directories = checkDirectoriesExist(folder);
				if (directories.length > 1){
					Dialog.create("CytoSeg - Settings");
					Dialog.addMessage("In which output folder should the mask be saved?");
					Dialog.addChoice("Folder", directories, "Current directory");
					Dialog.show();
					outputFolder = Dialog.getChoice();
					if (outputFolder == "Current directory") {
						if (startsWith(osSystem, "Windows")) {
							generateMask(filename, folder + "\\" + newFilename + "_mask.tif");
						} else {
							generateMask(filename, folder + "/" + newFilename + "_mask.tif");
						}
					} else {
						if (startsWith(osSystem, "Windows")) {
							pathOutputFolder = folder + outputFolder + "\\" + newFilename + "\\";
						} else {
							pathOutputFolder = folder + outputFolder + "/" + newFilename + "/";
						}	
						if (!File.exists(pathOutputFolder)){
							File.makeDirectory(pathOutputFolder);
						}
						generateMask(filename, pathOutputFolder + newFilename + "_mask.tif");
					}
				} else {
					if (startsWith(osSystem, "Windows")) {
						generateMask(filename, folder + "\\" + newFilename + "_mask.tif");
					} else {
						generateMask(filename, folder + "/" + newFilename + "_mask.tif");
					}
				}
			}
			else {
				pathToImageFolder = getDirectory("Choose a Directory");
				imageList = getFileList(pathToImageFolder);
				directories = checkDirectoriesExist(pathToImageFolder);
				if (directories.length > 1){
					Dialog.create("CytoSeg - Settings");
					Dialog.addMessage("In which output folder should the masks be saved?");
					Dialog.addChoice("Folder", directories);
					Dialog.show();
					outputFolder = Dialog.getChoice();
				}
				for (i=0; i<imageList.length; i++) {
					if(endsWith(imageList[i],"tif") || endsWith(imageList[i], "TIF") || endsWith(imageList[i], "tiff") || endsWith(imageList[i], "TIFF")){
						if(endsWith(imageList[i],"_mask.tif") || endsWith(imageList[i], "_mask.TIF") || endsWith(imageList[i], "_mask.tiff") || endsWith(imageList[i], "_mask.TIFF")){
							print("A mask image was skipped.");
						} else {
							open(imageList[i]);
							filename = getTitle();
							newFilename = replaceFileFormat(filename);
							if (directories.length > 1){
								if (outputFolder == "Current directory"){
									if (startsWith(osSystem, "Windows")) {
										generateMask(filename, pathToImageFolder + newFilename + "_mask.tif");
									} else {
										generateMask(filename, pathToImageFolder + newFilename + "_mask.tif");
									}
								} else {
									if (startsWith(osSystem, "Windows")) {
										pathOutputFolder = pathToImageFolder + outputFolder + newFilename + "\\";
									} else {
										pathOutputFolder = pathToImageFolder + outputFolder + newFilename + "/";
									}	
									if (!File.exists(pathOutputFolder)){
										File.makeDirectory(pathOutputFolder);
									}
									generateMask(filename, pathOutputFolder + newFilename + "_mask.tif");
								}
							} else {
								if (startsWith(osSystem, "Windows")) {
									generateMask(filename, pathToImageFolder + newFilename + "_mask.tif");
								} else {
									generateMask(filename, pathToImageFolder + newFilename + "_mask.tif");
								}	
							}
						}
					}
				}
			}
		}

		// parameter gauging
		if (choiceStep == "Parameter gauging") {
			if (startsWith(osSystem, "Windows")) {
				exec("cmd /c " + pathToPython3 + " " + pathToGUI + " None" + " 1")
			} else {
				exec('sh', pathToBridge, pathToPython3, pathToGUI, "None", "0");
			}
			updateParameters();
			mainMenu();
		}

		// pre-processing and extraction
		if (choiceStep == "Pre-processing and extraction") {
			createParameterDialog();
			outputFolder = Dialog.getString();
			choiceMask = Dialog.getChoice();
			choiceBatch = Dialog.getChoice();
			roll = Dialog.getNumber();
			//depth = Dialog.getNumber();
			sigma = Dialog.getNumber();
			block = Dialog.getNumber();
			small = Dialog.getNumber();
			factr = Dialog.getNumber();
			if (choiceBatch == "yes"){
				setBatchMode(true);
			}
			parameters = "" + randw + "," + randn + "," + depth + "," + sigma + "," + block + "," + small + "," + factr;
			Dialog.create("CytoSeg - Settings");
			Dialog.addMessage("Do you want to process a single image or all images in a folder ?");
			items = newArray("single", "all");
			Dialog.addChoice("process:", items, "all");
			Dialog.show();
			choiceImage = Dialog.getChoice();
			// single image
			if (choiceImage == "single"){
				pathToImage = File.openDialog("Select .tif Image");
				pathOutputImages = createOutputFolder(pathToImage, outputFolder);
				print("\n#########################################\n\nCytoSeg pre-processing is running...\n");
				filename = getTitle();
				newFilename = replaceFileFormat(filename);
				newFilename = replace(newFilename, "_filter", "");
				preprocessImage(newFilename, pathOutputImages);
				if (choiceMask == "no") {
					if (startsWith(osSystem, "Windows")) {
						generateMask(filename, pathOutputImages + "\\" + newFilename + "_mask.tif");
					} else {
						generateMask(filename, pathOutputImages + "/" + newFilename + "_mask.tif");
					}
				}
				else {
					dir = replace(pathToImage, filename, "");
					if (checkFileExists(newFilename + "_mask.tif", pathOutputImages) == false) {
						Dialog.create("WARNING");
						Dialog.addMessage("No mask was found for this image.");
						Dialog.show();
					}
				}
				print("\n#########################################\n\nPre-processing finished, post-processing in progress...\n");
				// network extraction
				pythonGraphAnalysis(pathToPython3, pathOutputImages, parameters);
				print("\n#########################################\n\nAnalysis done.\n");			
			}
			// multiple images
			else {
				pathToImageFolder = getDirectory("Choose a Directory");
				imageList = getFileList(pathToImageFolder);
				counter = 0;
				extractionArray = newArray(0);
				print("\n#########################################\n\nCyto Seg is running...\n");
				if (choiceMask == "yes") {
					for (i=0; i<imageList.length; i++) {
						if(endsWith(imageList[i],"tif") || endsWith(imageList[i], "TIF") || endsWith(imageList[i], "tiff") || endsWith(imageList[i], "TIFF")){
							if(endsWith(imageList[i],"_mask.tif") || endsWith(imageList[i], "_mask.TIF") || endsWith(imageList[i], "_mask.tiff") || endsWith(imageList[i], "_mask.TIFF")){
								print("A mask image was skipped.");
								counter += 1;
							}
							else {
								pathOutputImages = createOutputFolder(pathToImageFolder + imageList[i], outputFolder);
								print("\nFinished pre-processing of image " + i + 1 + " of " + imageList.length + "...\n");
								filename = replaceFileFormat(imageList[i]);
								preprocessImage(filename, pathOutputImages);
								if (checkFileExists(filename + "_mask.tif", pathToImageFolder) == false) {
									if (startsWith(osSystem, "Windows")) {
										generateMask(imageList[i], pathOutputImages + "\\" + filename + "_mask.tif");
									} else {
										generateMask(imageList[i], pathOutputImages + "/" + filename + "_mask.tif");
									}
								}
								else {
									if (startsWith(osSystem, "Windows")) {
										File.copy(pathToImageFolder + filename + "_mask.tif", pathOutputImages + "\\" + filename + "_mask.tif");
									} else {
										File.copy(pathToImageFolder + filename + "_mask.tif", pathOutputImages + "/" + filename + "_mask.tif");
									}
								}
							}
								array = newArray(pathOutputImages);
								extractionArray = Array.concat(extractionArray, array);
						}
						else {
							print("A file that is not a .tif file was skipped.");
							counter += 1;
						}
					}
				}
				else {					
					for (i=0; i<imageList.length; i++) {
						if(endsWith(imageList[i],"tif") || endsWith(imageList[i], "TIF") || endsWith(imageList[i], "tiff") || endsWith(imageList[i], "TIFF")){
							if(endsWith(imageList[i],"_mask.tif") || endsWith(imageList[i], "_mask.TIF") || endsWith(imageList[i], "_mask.tiff") || endsWith(imageList[i], "_mask.TIFF")){
								print("A mask image was skipped.");
								counter += 1;
							}
							else {
								pathOutputImages = createOutputFolder(pathToImageFolder + imageList[i], outputFolder);
								print("\nFinshed pre-processing of image " + i + 1 + " of " + imageList.length + "...\n");
								filename = replaceFileFormat(imageList[i]);
								preprocessImage(filename, pathOutputImages);
								if (startsWith(osSystem, "Windows")) {
									generateMask(imageList[i], pathOutputImages + "\\" + filename + "_mask.tif");
								} else {
									generateMask(imageList[i], pathOutputImages + "/" + filename + "_mask.tif");
								}
								array = newArray(pathOutputImages);
								extractionArray = Array.concat(extractionArray, array);
							}
						}
						else {					
							print("A file that is not a .tif file was skipped.");
							counter += 1;
						}
					}
				}
				if (imageList.length == counter) {
					Dialog.create("WARNING");
					Dialog.addMessage("Non of the files inside the selected folder \ncontained tif images to process.");
					Dialog.show();
				}
				print("\n#########################################\n\nPre-processing finished, post-processing in progress...\n");
				// network extraction
				for (j=0; j<extractionArray.length; j++){
					pythonGraphAnalysis(pathToPython3, extractionArray[j], parameters);
				}
				print("\n#########################################\n\nAnalysis done.\n");	
			}
		}
	}

	// MODE: reset Python 3 path
	if(choiceMenu == "Reset Python 3 path"){
		pathToPython3 = selectPythonPath(pathToCytoSeg);
		systemTestPython(osSystem, pathToPython3, pathToCytoSeg);
		mainMenu();
	}
	if (isOpen("Log")) {
   		//saves and closes the log
		selectWindow("Log");
		pathOutput = pathToCytoSeg + "log.txt";
		saveAs('txt', pathOutput);
		//run("Close");
	}
	Dialog.create("Goodbye");
	Dialog.addMessage("CytoSeg Analysis finished.");
	items = newArray("continue","quit");
	Dialog.addChoice("Do you want to continue or quit the application?", items, "quit");
	Dialog.show();
	choiceApplication = Dialog.getChoice();
	if (choiceApplication == "continue") {
		mainMenu();
	}
}
//// MENU: calling MainMenu
mainMenu();
