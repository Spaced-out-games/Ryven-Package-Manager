import pprint as pp
from reprlib import recursive_repr
import torch
from param_counter import get_parameter_count
import inspect#getfullargspec,getmodule, getmodulename
import cmath
MODULE = None
class Ryven_Nodifier:
    def __init__(self):
        pass
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
        out = []
        #path doubles as a blacklist
        attributes = self.filtered_dir(obj)
        for attr_name in attributes:#method name from list of them
            attr = getattr(obj, attr_name)#Get the actual attribute

            if attr_name not in path:
                path.append(attr_name)
                m = mrd(attr, depth + 1, path)
                if m != []:
                    out.append(
                        {attr_name: m}
                    )
                else:
                    out.append(attr_name)
            else:
                out.append(str(attr_name))
        return out
    def module_attribute_dictify(self, module):
        module = self._mrd(module)

def iterdict(d, path = []):
  for k,v in d.items():        
     if isinstance(v, dict):
        path.append(v)
        iterdict(v, path)
     else:            
        print(".".join(path))
RN = Ryven_Nodifier()
d = RN._mrd(torch)
iterdict(d)


