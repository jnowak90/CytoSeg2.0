////////// CytSeg GUI for Fiji  ////////////////
///////////////////////////////////////////////
/// https://github.com/jnowak90/CytoSeg2.0 ///
/////////////////////////////////////////////

// check the user's system
osSystem = getInfo("os.name");

/////////////////////
///// FUNCTIONS /////
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
// input custom parameters
function createParameterDialog(){
	Dialog.create("CytoSeg - Settings");
	Dialog.addMessage("Please name your Outputfolder \n(will be generated in the folder where your images are)");
	outputFolderName = "" + dayOfMonth + "-" + month + "-" + year;
	Dialog.addString("Folder", outputFolderName);
	Dialog.addMessage("");
	Dialog.addMessage("Do you want to use existing masks?");
	items = newArray("yes", "no");
	Dialog.addChoice("Choose", items, "no")
	Dialog.addMessage("");
	Dialog.addMessage("Do you want to proceed in silent mode?");
	Dialog.addChoice("Choose", items, "yes");
	Dialog.addNumber("rolling ball", roll);
	Dialog.addMessage("");
	//Dialog.addNumber("depth", depth);
	//Dialog.addMessage("");
	Dialog.addNumber("sigma", sigma);
	Dialog.addMessage("");
	Dialog.addNumber("block size", block);
	Dialog.addMessage("");
	Dialog.addNumber("component size", small);
	Dialog.addMessage("");
	Dialog.addNumber("average intensity", factr);
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
var month = month+1;    			// bug of the getDateAndTime function, month start with 0
// path to the file which contains default parameters, reload them for the next run
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
		// gauging
		Dialog.create("CytoSeg - Gauging");
		Dialog.addMessage("Before starting CytoSeg, the right parameters have to be selected for the images. \nPlease select an image for the gauging of those parameters.");
		Dialog.show();
		pathToFile = File.openDialog("Select .tif Image");
		newFilename = replaceFileFormat(pathToFile);
		pathOutputMask = newFilename + "_mask.tif";
		generateMask(pathToFile, pathOutputMask);
		if (startsWith(osSystem, "Windows")) {
			exec("cmd /c " + pathToPython3 + " " + pathToGUI + " " + pathToFile + " 1");
		} else {
			exec("sh", pathToBridge, pathToPython3, pathToGUI, pathToFile, "0");
		}
		
		File.delete(pathOutputMask);
		// extraction
		Dialog.create("CytoSeg - Settings");
		Dialog.addMessage("Please name your Outputfolder \n(will be created in the folder where your images are)");
		outputFolderName = "" + dayOfMonth + "-" + month + "-" + year;
		Dialog.addString("Folder", outputFolderName);
		Dialog.addMessage("Do you want to proceed in silent mode?");
		items = newArray("yes","no");
		Dialog.addChoice("Choose", items, "yes");
		Dialog.show();
		outputFolder = Dialog.getString();
		choiceBatch = Dialog.getChoice();
		if (choiceBatch == "yes"){
			setBatchMode(true);
		}
		updateParameters();
		parameters = "" + randw + "," + randn + "," + depth + "," + sigma + "," + block + "," + small + "," + factr;
		Dialog.create("CytoSeg - Settings");
		Dialog.addMessage("Do you want to process a single image or all images in a folder ?");
		items = newArray("single","all");
		Dialog.addChoice("process:", items, "all");
		Dialog.show();
		choiceImage = Dialog.getChoice();
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
				if (startsWith(osSystem, "Windows")) {
					generateMask(filename, folder + "\\" + newFilename + "_mask.tif");
				} else {
					generateMask(filename, folder + "/" + newFilename + "_mask.tif");
				}
			}
			else {
				pathToImageFolder = getDirectory("Choose a Directory");
				imageList = getFileList(pathToImageFolder);
				for (i=0; i<imageList.length; i++) {
					if(endsWith(imageList[i],"tif") || endsWith(imageList[i], "TIF") || endsWith(imageList[i], "tiff") || endsWith(imageList[i], "TIFF")){
						if(endsWith(imageList[i],"_mask.tif") || endsWith(imageList[i], "_mask.TIF") || endsWith(imageList[i], "_mask.tiff") || endsWith(imageList[i], "_mask.TIFF")){
							print("A mask image was skipped.");
						} else {
							open(imageList[i]);
							filename = getTitle();
							newFilename = replaceFileFormat(filename);
							if (startsWith(osSystem, "Windows")) {
								generateMask(filename, pathToImageFolder + "\\" + newFilename + "_mask.tif");
							} else {
								generateMask(filename, pathToImageFolder + "/" + newFilename + "_mask.tif");
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
				print("\n#########################################\n\nCyto Seg is running...\n");
				filename = getTitle();
				newFilename = replaceFileFormat(filename);
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
					if (checkFileExists(newFilename + "_mask.tif", dir) == false) {
						Dialog.create("WARNING");
						Dialog.addMessage("No mask was found for this image.");
						Dialog.show();
					}
					else {
						if (startsWith(osSystem, "Windows")) {
							File.copy(dir + newFilename + "_mask.tif", pathOutputImages + "\\" + newFilename + "_mask.tif");
						} else {
							File.copy(dir + newFilename + "_mask.tif", pathOutputImages + "/" + newFilename + "_mask.tif");
						}
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
								print("\nFinshed pre-processing of image " + i + 1 + " of " + imageList.length + "...\n");
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
	Dialog.addMessage("CytoSeg Analysis finished");
	Dialog.show();
}
//// MENU: calling MainMenu
mainMenu();
