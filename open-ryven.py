'''
This file opens Ryven and circumvents issues relating to Qt on Mac
'''
from platform import system # for OS detection

command = "export QT_MAC_WANTS_LAYER=1" #only executes if on macOS
import os# for running shell command above
if system() == "Darwin":
    os.environ['QT_MAC_WANTS_LAYER'] = 1

import ryven
ryven.run_ryven()
