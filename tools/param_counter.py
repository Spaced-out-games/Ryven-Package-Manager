import inspect
import time
import re
import types
import sys


def get_parameter_count(func): 
    """Count parameter of a function.

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

    # If the function is a builtin function we use our
    # approach. If it's an ordinary Python function we
    # fallback by using the the built-in extraction
    # functions (see else case), otherwise
    if isinstance(func, types.BuiltinFunctionType):
        try:
            arg_test = 999
            s = [None] * arg_test
            func(*s)
        except TypeError as e:
            message = str(e)
            found = re.match(
                r"[\w]+\(\) takes ([0-9]{1,3}) positional argument[s]* but " +
                str(arg_test) + " were given", message)
            if found:
                return int(found.group(1))

            if "takes no arguments" in message:
                return 0
            elif "takes at most" in message:
                found = re.match(
                    r"[\w]+\(\) takes at most ([0-9]{1,3}).+", message)
                if found:
                    return int(found.group(1))
            elif "takes exactly" in message:
                # string can contain 'takes 1' or 'takes one',
                # depending on the Python version
                found = re.match(
                    r"[\w]+\(\) takes exactly ([0-9]{1,3}|[\w]+).+", message)
                if found:
                    return 1 if found.group(1) == "one" \
                            else int(found.group(1))
        return -1  # *args
    else:
        try:
            if (sys.version_info > (3, 0)):
                argspec = inspect.getfullargspec(func)
            else:
                argspec = inspect.getargspec(func)
        except:
            raise TypeError("unable to determine parameter count")# inp = str

        return [] if argspec.varargs else argspec.args



