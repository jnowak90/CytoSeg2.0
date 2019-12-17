#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 12:09:16 2019

@author: root
"""

### System test ###
import importlib
import sys

print("\nYour python version used for this script:")
print(sys.version)

path = sys.argv[1]
f = open("TestFile.txt", "w")
f.write("This is a file to test if the right Python 3 path was found.")
f.close()

modules = ['matplotlib', 'networkx', 'numpy', 'pandas', 'PIL', 'scipy', 'skimage', 'shapely', 'packaging', 'collections', 'glob', 'itertools', 'tkinter']

counter = 0
for module in modules:
    try:
        importlib.import_module(module)
    except ImportError:
        print('Python module: ' + '"' + module + '"' + ' not found. Please install before continuing.')
        counter += 1

if counter == 0:
    print("\n### All necessary Python modules were found.")