'''
this file automatically imports packages inside the packages folder. Use settings.json to configure
'''
'''
import os
 
rootdir = '.../packages'
for it in os.scandir(rootdir):
    if it.is_dir():
        print(it.path)
        '''
from pathlib import Path
import os

def get_modules():
    '''Gets the paths of all'''
    path = str(Path(__file__).parent) + "/packages/"
    paths = []
    for it in os.scandir(path):
        if it.is_dir():
            paths.append(str(it.path))
    return paths