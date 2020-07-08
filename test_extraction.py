from ExtractionPipeline import CytoSeg
import platform
import os


# This funciton will extract the network from the already preprocessed file in '/DemoImages/Extraction/output/actin_control' and the corresponding
# mask which is stored in the same location. 
def test_extraction():
    if platform.system() != "Linux":
        print("Test might not work on Windows")

    with open('defaultParameter.txt', 'r') as file:
        parameterString = file.read()
    cwd = os.getcwd()
    dir = cwd + '/DemoImages/Extraction/output/actin_control'
    myExtraction = CytoSeg('./ExtractionPipeline.py' , dir, parameterString, '0')
    return myExtraction

test_extraction()
