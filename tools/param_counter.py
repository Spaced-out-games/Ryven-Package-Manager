from imp import is_builtin
import inspect
from pprint import pprint
import time
import re
import types
import sys

'''
Source: https://stackoverflow.com/questions/48567935/get-parameterarg-count-of-builtin-functions-in-python
Edited to provide additional functionality for functions that fail due to Runtime Errors
'''
def get_parameter_count(func, request_names = True):
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
			try:
				func(*s)
			#this except block was an addition to the original function
			except RuntimeError as e:
				message = e.args[0] #get error message
				message = message.split("Declaration: ")[1]#Remove noise behind declaration
				message = message.split(" -> ")[0]#remove noise after arroe
				index = message.find("(")#find first parenthesis
				message = message[index+1:len(message)-1]#remove everything outside parenthesis, including parenthesis
				if message.count("(") == 0:
					arglines = message.split(", ")
					args = []
					if request_names:
						print(arglines)
						#return arglines
					else:
						return len(arglines)
				else:
					print("Failed to parse function " + func.__name__)

				#print(message)
		except TypeError as e:
			message = str(e)
			found = re.match(
				r"[\w]+\(\) takes ([0-9]{1,3}) positional argument[s]* but " +
				str(arg_test) + " were given", message)
			if found:
				if request_names:
					return (["No Name Found"],int(found.group(1)))
				else:
					return int(found.group(1))

			if "takes no arguments" in message:
				if request_names:
					return []
				else:
					return 0
			elif "takes at most" in message:
				found = re.match(
					r"[\w]+\(\) takes at most ([0-9]{1,3}).+", message)
				if found:
					print("found: ", found)
					if request_names:
						return (["No Name Found"],int(found.group(1)))
					else:
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
			#print("type; "+str(type(func)))
			return -2
			#raise TypeError("unable to determine parameter count")
		#TypeError: square() takes 1 positional argument but 2 were given	<- too many arguments passed, use as a break statement
		#TypeError: square(): argument 'input' (position 1) must be Tensor, not int <- Wrong argument type, skip this case
			return -1 if argspec.varargs else len(argspec.args)



def get_parameter_count_module(mod):
	out = {}
	for x in dir(mod):
		e = mod.__dict__.get(x)
		if isinstance(e, types.BuiltinFunctionType):
			key = "{}.{}".format(mod.__name__, e.__name__)
			out[key] = get_parameter_count(e)
			#print("{}.{} takes {} argument(s)".format(mod.__name__, e.__name__, get_parameter_count(e)))
	return(out)
