
from param_counter import get_params
from inspect import getdoc, cleandoc, ismodule, ismethod,isclass, getmembers
import pprint as pp
builtin_methods = []
def dir_with_members(obj):
    d = dir(obj)
    gm = getmembers(obj)
    for i in d:
        if i in gm:
            gm.pop(gm.index(i))
    return d + gm

        
def recursive_dir_list(obj, depth = 0, path = []):
    '''
    Converts a module and its attributes/members/methods into a recursive dictionary
    Arguments:
        obj: any module
        depth: Do NOT touch, it is used for recursion.
        path: Do NOT touch, it is used for recursion
    Returns: a list of attributes. Dictionary items are sub-modules and sub-classes inherited by `obj`
    '''
    out = []
    attributes = dir_with_members(obj)
    for i in attributes:
        try:#Try to get the attribute
            attr = getattr(obj, i)
            if (ismodule(attr) or isclass(attr)) and (i not in path) and ("__" not in i):
                path.append(i)
                out.append(recursive_dir_list(attr, depth+1, path))
            elif not (ismodule(attr) or isclass(attr)):
                out.append(i)
            else:
                pass
        except Exception:
            pass
        d = {}
        l = []
    for i in range(len(out)):
        if isinstance(out[i], dict):
            pass
        elif isinstance(out[i], dict):
            pass
    return out
class Ryven_Nodifier:
    def __init__(self):
        pass
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
        name = node_name if node_name != "" else ""#self._canon_name(func)
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
        if init_inputs=="":
            print(node_name, errcode)
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
        pass
c = "#e19d1e"#torch module color code
import torch
color = "#D06B34"
essentials = [
    "torch.tensor",
    "torch.Tensor.item",
    #tensor slice operations, ie [0:-1], [0][:]
    "torch.FloatTensor",
    "torch.Tensor.size",
    "torch.Tensor.ndimension",
    "torch.Tensor.view",
    "torch.Tensor.tolist",
    "torch.Tensor.numpy",
    "torch.as_tensor",
    "torch.autograd.backward",
    "torch.autograd.grad",
    "torch.device",
    "torch.detach",
    "torch.empty",
    "torch.enable_grad",
    "torch.is_grad_enabled",
    #torch.fft
    "torch.from_numpy",
    "torch.from_file",
    "torch.gru",
    "torch.gru_cell",
    "torch.is_grad_enabled",
    "torch.layer_norm",
    "torch.lstm",
    "torch.lstm_cell",
    "torch.empty_like",
    "torch.zeros_like",
    "torch.ones_like",
    "torch.optim.Adam",
    "torch.randn",
    "torch.randn_like",
    "torch.randint",
    "torch.randint_like",
    "torch.reshape"
    #pd.DataFrame(x.numpy()) #x Tensor --> pandas dataframe
]
'''
rn = Ryven_Nodifier()
code = ""
for i in essentials:
    f = eval(i)
    n = i.split(".")[-1].replace("_", " ")
    code +=rn.nodify(f,n,color)

with open("test.py","w") as o:
    o.write(code)
'''
p = get_params(torch.tensor)
print(p)