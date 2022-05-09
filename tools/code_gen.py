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
    def nodify(self, func, node_name: str = "", color: str = "#ffffff"):
        """Creates A Ryven Node class of a function

        Args:
            func(callable): The `callable` that is to be ported to Ryven
            node_name(str): The name of the newly created class. Defaults to "". If no name is passed, a fetch will be attempted.
            color(str): hex color code for the color the node is in Ryven
            m_name(str): the name of the module this callable is inherited by.
            
            `m_name` needs to be the same as the module's `import` name. For example, if you `import numpy as np`,`m_name` should be `np`
        Returns:
            (str): A string containing the code, that when ran creates a Ryven node that has the same functionality as `func`
        """
        
        #Psuedo - code
        #uncallable:
        #   init_inputs = []
        #   @update_event:
        #       self.set_output_value(0, _canon_name)


        #No names:
        #   init_inputs = NodeInputBP() * arg_cnt
        #   @update_event:
        #       self.set_output_value(0, _canon_name(self.input(0), self.input(1)...(self.input(n))))

        #No arguments:
        #   init_inputs = []
        #   @update_event:
        #       self.set_output_value(0, _canon_name())

        #No error:
        #   init_inputs = NodeInputBP(label = 'argname') for each argument, in list
        #   @update_event:
        #       self.set_output_value(0, _canon_name(self.input(0), self.input(1)...(self.input(n))))



