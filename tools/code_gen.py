
import pprint
from numpy import isin
import torch
from param_counter import get_parameter_count
import inspect#getfullargspec,getmodule, getmodulename
import cmath
'''
DESCRIPTION
This file can help generate simple nodes, namely those that do not need a visual representation or are Python - implemented functions.
It is mainly a dev tool and as such is NOT suitable for 1 to 1 renditions of module functions. Code generated by this function still needs tested and debugged
'''
"""
{warning}
class {node_name}(Node):
    \"\"\"{obj.__doc__}\"\"\"
    title = '{name}'
    init_inputs = [
        {inputs}
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '{color}'
    def update_event(self, inp=-1):
        self.set_output_val(0, {m_name}.{call}({
            ', '.join([f'self.input({i})' for i in range(len(args))]) 
                                        }))
"""
def module_recursive_dict(object, depth = 0, path = [], master = {}):
    #if inspect.ismodule(object):
    out = []
    if depth == 0:
        master = object
    attrs = dir(object)

    for attr in attrs:#scan thru all attrs
        a = getattr(object, attr)
        if inspect.ismodule(a) and path.count(attr) <=1: #if it is a module, recursively iterate through children
            path.append(attr)
            out.append({attr: module_recursive_dict(a, depth + 1, path, master)})
            #print(path)
        elif not inspect.ismodule(a) and path.count(attr) <=1:
            out.append(attr)
    return out
import torch

print(vars(torch))