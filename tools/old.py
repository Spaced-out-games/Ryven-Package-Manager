from ast import arg
from string import ascii_uppercase
import sys
from inspect import isfunction, getargspec,getfullargspec
def num_args(f):
  if isfunction(f):
    return len(getargspec(f).args)
  else:
    try:
        spec = f.__doc__.split('\n')[0]
        args = spec[spec.find('(')+1:spec.find(')')]
        return args.count(',')+1 if args else 0
    except AttributeError as A:
        return None
def node_from_function(name: str, call: str, obj: object, color: str,m_name = ""):
    """Create a node class definition from a given function.

    Parameters
    ----------
    name: str
        Becomes the title of the node.
    call: str
        Represents the full function call without parentheses, e.g. `math.sqrt`
    obj: object
        The function object. Its source will be parsed using the inspect module,
        which might not work for some functions, expecially those of C bindings.
    color: str
        The color hex code the node will get.

    Returns
    -------
    node_def: str
        The code defining the node class.
    """
    
    try:
        sig = getfullargspec(obj)
        args = sig.args
        warning = ""
    except Exception as e:
        #raise Exception(f"Could not parse source of {name}.")
        #"""
        
        warning = f'''\n
"""
WARNING: Module {name} was generated using fallback option. May contain bugs
"""
'''
        
        #"""
        argcnt = num_args(obj)
        args = []
        if argcnt == None:#placeholder
            argcnt = 0
        for i in range(argcnt):
            args.append(chr(i + 97))

    
    inputs = '\n'.join([f"NodeInputBP('{param_name}')," for param_name in args])
    node_name = f'{name}_Node'

    node_def = f"""
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

    return node_def
def attributes_without_builtins(o):
    """_summary_

    Rarameters:
    ----------
        o (object): object to filter

    Returns:
    ----------
        list: dir(o) with builtin attributes like __eq__ removed
    """
    d = dir(o)
    for i in range(len(d)):
        if d[i].find("__") == 0:
            d[i] = ""
    while d.count("") > 0:
        d.pop(d.index(""))
    return d
def nodify_module(module, color):
    code = ""
    m_name = ""#module.__package__
    attrs = []
    d = attributes_without_builtins(module)
    for i in d:
        f = getattr(module,i)
        name = i.capitalize() + "Node"
        func= node_from_function(name,i,f,color,m_name)
        if isinstance(func, str):
            code += func
            attrs.append(name)
    s_attr = ""
    for i in range(len(attrs)):
        s_attr+= (attrs[i] + "\t\n") if i==0 else (", " + attrs[i] + "\t\n")
    return f'''
from ryven.NENV import *
import {m_name}
{code}
{m_name}_nodes = [{s_attr}]
export_nodes(*{m_name}_nodes)
    '''

import torch

code = node_from_function("abs","abs",abs,"")
print(code)

    
        
