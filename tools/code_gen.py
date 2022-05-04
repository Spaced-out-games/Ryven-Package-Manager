import pprint as pp
import torch
from param_counter import get_parameter_count
import inspect#getfullargspec,getmodule, getmodulename
import cmath

def mrd(obj, depth = 0, path = []):
    #if inspect.ismodule(object):
    out = []
    #path doubles as a blacklist
    attributes = dir(obj)
    for i in attributes:
        if "__" in i:
            attributes.pop(attributes.index(i))
        else:
            continue




    for attr_name in attributes:#method name from list of them
        try:
            attr = getattr(obj, attr_name)#Get the actual attribute
        except:
            continue
        if attr_name not in path:
            path.append(attr_name)
            out.append(
                {attr_name: mrd(attr, depth + 1, path)}
            )
        else:
            out.append(str(attr_name))
    return out

m = mrd(torch)

