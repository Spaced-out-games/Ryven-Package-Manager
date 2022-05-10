#import pprint as pp
#from reprlib import recursive_repr
#import torch
from param_counter import get_params, param_dict
from inspect import getdoc, cleandoc
MODULE = None
class Ryven_Nodifier:
    def __init__(self):
        pass
    def _canon_name(self, obj):
        r = repr(obj).replace("<class '", "") 
        r = r.replace("'>","") if r.endswith("'>") else r
        #print(r)
    def walk_dict(self, d, depth = 0, path = []):
        prep = '.'.join(path)
        print(prep)
        for k in d:
            v = d[k]
            if isinstance(v, dict):
                path.append(v)
                return self.walk_dict(v, depth + 1)
            elif isinstance(v, list):
                d = lambda x, path: path.copy().append(x)
                out = []
                for i in v:
                    out.append(d(i, path))
                return out
            else:
                return path

    def filtered_dir(self, obj):
        '''
        Filter the contents of dir(obj)
        Removes builtins like __init__ and __call__, and removes non-callable attributes or attributes that cannot be retrieved via getattr.
        May add ability to conditionally filter these elements
        '''
        d = dir(obj)
        for i in range(len(d)):
            if d[i].count("__") == 2:
                d[i] = ""#"" instead of pop() to prevent index out of range errors;No method or attribute can be "" and will later be popped
            elif d[i][0] == "_" or d[i][-1] == "_":#not replaced
                d[i] = ""
            else:
                try: #to get attribute
                    g = getattr(obj, d[i])
                    #if isinstance(g, [int, float, ])
                    if not callable(g):
                        d[i] = ""





                except NotImplementedError:
                    d[i] = ""#remove this element, as it cannot be retreived via getattr
                except AttributeError:
                    d[i] = ""#remove this element, as it cannot be retreived via getattr
                except RuntimeError:
                    d[i] = ""#remove this element, as it cannot be retreived via getattr
                except Exception as e:
                    raise(e.args[0])
        while d.count(""):
            d.remove("")
        return sorted(d)
    def _mrd(self, obj, depth = 0, path = []):
        if depth > 5:
            return []
        out = []
        #path doubles as a blacklist
        attributes = self.filtered_dir(obj)
        for attr_name in attributes:#method name from list of them
            attr = getattr(obj, attr_name)#Get the actual attribute

            if attr_name not in path:
                path.append(attr_name)
                m = self._mrd(attr, depth + 1, path)
                if m != []:
                    out.append(
                        {attr_name: m}
                    )
                else:
                    out.append(attr_name)
            else:
                out.append(str(attr_name))
        return out
    def nodify(self, func, node_name: str = "", color: str = "#ffffff"):
        try:
            documentation = cleandoc(getdoc(func))
        except:
            documentation = ""
        """Creates A Ryven Node class of a function

        Args:
            func(callable): The `callable` that is to be ported to Ryven
            node_name(str): The name of the newly created class. Defaults to "". If no name is passed, a fetch will be attempted.
            color(str): hex color code for the color the node is in Ryven
        Returns:
            (str): A string containing the code, that when ran creates a Ryven node that has the same functionality as `func`
        """
        uncallable = not callable(func)
        warning = ""
        pd = get_params(func)#param dictionary.
        errcode = pd['error']
        args = pd['args']
        ka = pd['kargs']
        va = pd['vargs']
        nameless = errcode == 2
        argless  = errcode == 1
        argcnt = len(args)
        init_inputs, init_outputs = ("","")
        cname = func.__name__
        name = node_name if node_name != "" else self._canon_name(func)
        sov = f"\t\tself.set_output_value({cname}(*self.inputs))"
        #Psuedo - code
        #uncallable:
        #   init_inputs = []
        #   init_outputs = [NodeOutputBP()]
        #   @update_event:
        #       self.set_output_value(0, _canon_name)
        if uncallable:
            init_inputs = "init_inputs = []"
            init_outputs = "init_outputs = [NodeOutputBP()]"
            warning = f'"""WARNING: an uncallable object was passed (ie a variable or constant). This should not be an issue, but still be weary.'
            sov = f"self.set_output_value({cname})"

        #No names:
        #   init_inputs = NodeInputBP() * arg_cnt
        #   @update_event:
        #       self.set_output_value(0, _canon_name(self.input(0), self.input(1)...(self.input(n))))

        elif nameless:
            init_inputs = f"init_inputs = [{'NodeInputBP(), ' * argcnt}]"\
                .replace(", ]", "]")#removes comma on the last one
            init_outputs = "init_outputs = NodeOutputBP()"
            warning = f'"""WARNING: {node_name} arg count was found, but argument names were not. Solution has been implemented, should still work"""'
        #No arguments:
        #   init_inputs = []
        #   @update_event:
        #       self.set_output_value(0, _canon_name())

        elif argless:
            init_inputs = "init_inputs = []"
            init_outputs = "init_outputs = [NodeOutputBP()]"
            warning = f'"""WARNING: {node_name} takes no arguments. Should not cause problems, but may"""'

        #No error:
        #   init_inputs = NodeInputBP(label = 'argname') for each argument, in list
        #   @update_event:
        #       self.set_output_value(0, _canon_name(self.input(0), self.input(1)...(self.input(n))))
        elif errcode == 0:
            for i in args:
                init_inputs += f"NodeInputBP(label = '{i}'), "
            if init_inputs.endswith(", "):
                init_inputs = init_inputs[0: len(init_inputs) -2]
            init_inputs = f"init_inputs = [{init_inputs}]"
            init_outputs = f"init_outputs = [NodeOutputBP()]"
            a = [f"self.input({i}), " for i in range(argcnt)]

            warning = ""
        else:
            pass#raise Exception(f"Unsupported error code. This should never happen. Error code: {errcode}")
        return f'''
{warning}
class {node_name}(Node):
    """{documentation}"""
    title = '{name}'
    {init_inputs}
    {init_outputs}
    color = '{color}'
    
    def update_event(self, inp = -1):
        {sov}
'''
    def nodify_module(self, module, color: str = "#ffffff"):
        d = self._mrd(module)
        t = self.walk_dict(d)
        print(t)

import math as m

rn = Ryven_Nodifier()
l = rn.filtered_dir(m)
code = ""
for i in l:
    code += rn.nodify(getattr(m, i), node_name = i+"_Node", color = "#aa2352")

code = f"""
from ryven.NENV import *
from math import *
{code}
math_nodes = [{", ".join(l)}]
export_nodes(*math_nodes)
"""
with open("math_nodes.py","w") as o:
    o.write(code)
