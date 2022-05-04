import builtins
from importlib.machinery import ModuleSpec
from operator import methodcaller
from posixpath import isabs
from torch import ModuleDict, ThroughputBenchmark
from param_counter import get_parameter_count
from inspect import isbuiltin, isclass, ismodule, ismethod
import time as t
import types
from ast import literal_eval
def getdir(obj: object,  blacklist = []):
    s = dir(obj)
    flag = True
    for i in blacklist:
        if i in s:
            while i in s:
                s.pop(s.index(i))
    return s
def IsError(obj):
    """Returns True if obj is an error (ie TypeError) and False otherwise

    Args:
        obj (Any): Any object

    Returns:
        bool: Whether or not the given object is an Error
    """
    if callable(obj):#Exceptions are callable
        try:#Attempt to call the object. 
            e = obj()#Exception objects can take zero arguments
            return isinstance(e, Exception)#If the output of obj is an Exception instance, obj must be an Error instance
        except Exception as e:
            if isinstance(e, TypeError) or isinstance(e, ValueError):
                return False
            else:
                raise e.args[0]
        return False
    return False
def HasUniqueChildren(obj:object, blacklist = []):
    b_ins = dir(type)
    objattrs = getdir(obj, blacklist = b_ins + blacklist)
    return (objattrs, len(objattrs)>0)
l = [
        [
            [
                [40,30,77]
            ]
        ]
    ]
'''
fetch root attributes
iterate through them:
    if one has children attributes:
        iterate through them:
            if one has children attributes:...(recursively)
    else:
        append child-less attribute to tree
        
'''
def module_dict(module, path = [],skip_errors = True):
    root_attrs = getdir(module)
    #path #Traceback path
    for attr in root_attrs:
        
        try:
            instance = getattr(module, attr)
        except Exception as e:
            raise e.args[0]
        if IsError(instance):
            continue
        if HasUniqueChildren(module)[1]:
            path.append(attr)
            if (path.count(attr)>1):#detecting endless loop
                return#stop digging
            else:
                #not endless (yet, if ever)
                print("continue with"+ ".".join(path))
                
                t.sleep(0.5)
                #return module_dict(getattr(module, attr), path)
def var_readout(var):
    try:
        print(var.__name__)
    except Exception: pass
    print("isbuiltin:",str(isbuiltin(var)))
    print("ismethod:",str(ismethod(var)))
    print("ismodule:",str(ismodule(var)))
    print("isclass:",str(isclass(var)))
    print("is callable:",str(callable(var)))

