from inspect import getargspec, getfullargspec #Python function introspection
from re import match #regex
from inspect import getdoc, cleandoc, ismodule, ismethod,isclass, getmembers

class Ryven_Nodifier:

	class helpers:
		def param_getter(self):
			raise NotImplementedError("Not yet implemented. Currently in the works")
			"""
			Gets the name of parameters for all python functions and MOST functions implemented in C/C++
			"""
			pass
		def dir_to_dict(self, obj):
			raise NotImplementedError("Not yet implemented")
			"""Creates a recursive dictionary from an object's inspect.getmembers result

			Args:
				obj (any): Any object with methods and modules you wish to get a dictionary of
			"""
			pass
	def __init__(self):
		pass
	def nodify(self, func, node_name, color):
		"""Converts a function into a Ryven Node, and returns the associated code

		Args:
			func (callable): Any callable, whether it be a method or function
			node_name (_type_): _description_
			color (_type_): _description_
		"""