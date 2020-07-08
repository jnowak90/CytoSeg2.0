from tkinter import Tk
from gaugingGui import *
import platform

# Select Demoimage/Gauging/actin_control_filter.tif when prompted by the Gui"
def test_for_gauging():
    if platform.system() != "Linux":
        print("Test might not work on Windows")
    master = Tk()
    
    my_gui = GaugingGui(master, "./GaugingGui.py", "None" , "0")
    master.mainloop()
test_for_gauging()
