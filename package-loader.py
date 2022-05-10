'''
this file automatically imports packages inside the packages folder. Use settings.json to configure
'''
import os, sys, importlib.util
from pathlib import Path
importlib.util.module_from_spec()
"""
"""
def get_modules():
    '''Gets the paths of all module directories'''
    path = str(Path(__file__).parent) + "/packages/"
    paths = []
    for it in os.scandir(path):
        if it.is_dir():
            paths.append(str(it.path))
    return paths
