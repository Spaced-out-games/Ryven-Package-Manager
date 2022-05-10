
from inspect import getargspec, getfullargspec
from re import match
from types import BuiltinFunctionType
'''
Source: https://stackoverflow.com/questions/48567935/get-parameterarg-count-of-builtin-functions-in-python
Edited to provide additional functionality for functions that fail due to Runtime Errors
'''
def param_dict(args = [], kwargs = False, varargs = False, error = 0):
    return{
        "args": args,
        "kargs": kwargs,
        "vargs": varargs,
        "error": error
    }

def get_params(func): 
    """
    Count parameters of a function. and try to retreive their names, if possible

    Supports Python functions (and built-in functions).
    If a function takes *args, then -1 is returned

    Example:
        import os
        arg = get_parameter_count(os.chdir)
        print(arg)  # Output: 1

    -- For C devs:
    In CPython, some built-in functions defined in C provide
    no metadata about their arguments. That's why we pass a
    list with 999 None objects (randomly choosen) to it and
    expect the underlying PyArg_ParseTuple fails with a
    corresponding error message.
    """
    #Error guide
    #-2:    unsupported callable
    #-1:    no args found.
    #0:     No error. Good
    #1:     no args
    #2:     no names


    # If the function is a builtin function we use our
    # approach. If it's an ordinary Python function we
    # fallback by using the the built-in extraction
    # functions (see else case), otherwise
    try:
        #version conditioning may be important for a later build
        argspec = getargspec(func)
        v = argspec.varargs
        k = argspec.keywords
        if v is not None:
            v = True if len(v)>0 else False
        if k is not None:
            k = True if len(k)>0 else False

        return param_dict(argspec.args,varargs = v, kwargs=k)
    except Exception as e:
        if str(e) == "unsupported callable":
            return param_dict(error = -2)
    if not callable(func):
        return param_dict(error = "uncallable")
    if isinstance(func, BuiltinFunctionType):
        try:
            arg_test = 999
            s = [None] * arg_test
            func(*s)
        except TypeError as e:
            message = str(e)
            found = match(
                r"[\w]+\(\) takes ([0-9]{1,3}) positional argument[s]* but " +
                str(arg_test) + " were given", message)
            if found:
                r = [None] * int(found.group(1))
                return param_dict(args = r, error = 2)

            if "takes no arguments" in message:
                return param_dict(error = 1)
            elif "takes at most" in message:
                found = match(
                    r"[\w]+\(\) takes at most ([0-9]{1,3}).+", message)
                if found:
                    r = [None] * int(found.group(1))
                    return param_dict(args = r, error = 2)
            elif "takes exactly" in message:
                # string can contain 'takes 1' or 'takes one',
                # depending on the Python version
                found = match(
                    r"[\w]+\(\) takes exactly ([0-9]{1,3}|[\w]+).+", message)
                if found:
                    r = [''] if found.group(1) == "one" else [''] * int(found.group(1))
                    return param_dict(args = r, error = 2)
        return param_dict(varargs = True)  # *args
    return param_dict(error = -1)#bad error. 
    #raise ValueError("Could not find arguments for function "+ func.__name__)
#import torch.nn
#r = repr(torch.nn.LSTM).replace("<class '", "")
#r = r.replace("'>","") if r.endswith("'>") else r
