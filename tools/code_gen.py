import pprint as pp
from reprlib import recursive_repr
import torch
from param_counter import get_params, param_dict
from inspect import getdoc
MODULE = None
class Ryven_Nodifier:
    def __init__(self):
        pass
    def _canon_name(self, obj):
        r = repr(obj).replace("<class '", "") 
        r = r.replace("'>","") if r.endswith("'>") else r
        print(r)
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
    def nodify(self, func, node_name = None, color = "#aabbcc", m_name = ""):
        #get node name, if not provided
        if node_name == None:
            try:
                node_name = func.__name__
            except AttributeError:
                try:
                    node_name = func.__qualname__
                except AttributeError as e:
                    raise AttributeError(e)
        #Initialize variables
        params = get_params(func)
        errcode = params['error']
        nameless = False
        message = ""
        arg_cnt = None
        Uncall = False



        if errcode == None:#Errorless
            param_names = params['args']#Set param names to the parameters
        elif errcode == "unsupported callable":#Callable is unsupported, raise an error
            raise ValueError(f"{func.__name__} is not a supported callable.")
        elif errcode == "uncallable":#object is not callable; may just be a normal attribute
            Uncall = True#so flag it
            param_names = []#and have no parameters
        elif errcode == "No Names":#object has no named arguments
            nameless = True#so flag it
            param_names = None#
            arg_cnt = len(params["args"])
        elif errcode == "No Arguments":
            param_names = []
        else:
            pass
        arg_cnt = len(param_names) if arg_cnt == None else None

        if nameless:
            inputs = "NodeInputBP()\n" * arg_cnt
        else:
            if param_names != []:
                inputs = ["NodeInputBP(label = '{i}')\n" for i in param_names]
            else:
                inputs = []
        call_name = repr(func)

        if Uncall:
            ov = f"""
        self.set_output_val(0, {self._canon_name(func)}"""
        else:

            ov = f"""
        self.set_output_val(0, {self._canon_name(func)}("""




        warning = f'''\n
"""
WARNING: {message}
"""
'''




        if errcode == None:#reset
            warning = ""
        
        




        node_def = f"""
{warning}
class {node_name}(Node):
    \"\"\"{func.__doc__}\"\"\"

    title = '{func.__name__}'
    init_inputs = [
        {inputs}
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '{color}'

    def update_event(self, inp=-1):
        {ov}
"""

