from inspect import ismodule
import torch



def step(obj, path = []):
	path = path.copy()
	attrs = [getattr(obj, attr) for attr in dir(obj)]
	focus = attrs[0]
	str(type(focus))
	if str(type(focus)) not in path:
		path.append(str(type(obj)))
		return step(focus, path)
	return[]

'''
t = step(torch)
print(*t)

t = step(*t)
print(*t)
'''
out = step(torch.nn)
print(out)