from imp import is_builtin
from inspect import isbuiltin
import json, torch,pprint

def dump(obj, level=0, blacklist = []):
   for a in dir(obj):

      val = getattr(obj, a)
      if a not in blacklist:
        print (level*' ', val)
      else:
        print(a)
        blacklist.append(a)
        dump(val, level=level+1, blacklist = blacklist)
dump(torch)
