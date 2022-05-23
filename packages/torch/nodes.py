from ryven.NENV import *
import torch#from torch import *?
#"#D06B34"

class torch_tensor_Node(Node):
	'''
	tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False) -> Tensor

Constructs a tensor with :attr:`data`.

.. warning::

    :func:`torch.tensor` always copies :attr:`data`. If you have a Tensor
    ``data`` and want to avoid a copy, use :func:`torch.Tensor.requires_grad_`
    or :func:`torch.Tensor.detach`.
    If you have a NumPy ``ndarray`` and want to avoid a copy, use
    :func:`torch.as_tensor`.

.. warning::

    When data is a tensor `x`, :func:`torch.tensor` reads out 'the data' from whatever it is passed,
    and constructs a leaf variable. Therefore ``torch.tensor(x)`` is equivalent to ``x.clone().detach()``
    and ``torch.tensor(x, requires_grad=True)`` is equivalent to ``x.clone().detach().requires_grad_(True)``.
    The equivalents using ``clone()`` and ``detach()`` are recommended.

Args:
    data (array_like): Initial data for the tensor. Can be a list, tuple,
        NumPy ``ndarray``, scalar, and other types.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, infers data type from :attr:`data`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    pin_memory (bool, optional): If set, returned tensor would be allocated in
        the pinned memory. Works only for CPU tensors. Default: ``False``.


Example::

    >>> torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
    tensor([[ 0.1000,  1.2000],
            [ 2.2000,  3.1000],
            [ 4.9000,  5.2000]])

    >>> torch.tensor([0, 1])  # Type inference on data
    tensor([ 0,  1])

    >>> torch.tensor([[0.11111, 0.222222, 0.3333333]],
    ...              dtype=torch.float64,
    ...              device=torch.device('cuda:0'))  # creates a torch.cuda.DoubleTensor
    tensor([[ 0.1111,  0.2222,  0.3333]], dtype=torch.float64, device='cuda:0')

    >>> torch.tensor(3.14159)  # Create a scalar (zero-dimensional tensor)
    tensor(3.1416)

    >>> torch.tensor([])  # Create an empty tensor (of size (0,))
    tensor([])
	
	
	
	'''
	#torch.tensor(data,dtype,device,requires_grad)
	title = 'Make Tensor'
	tags = ['torch','make','tensor']
	init_inputs = [
		NodeInputBP(dtype=dtypes.Data(default=None),label="data"),
		NodeInputBP(dtype=dtypes.Data(default=None),label="dtype"),
		NodeInputBP(dtype=dtypes.Data(default=None),label="device"),
		NodeInputBP(dtype=dtypes.Data(default=False),label="requires_grad")
		]
	init_outputs = [NodeOutputBP()]
	color ="#D06B34"
	def update_event(self, inp=-1):
		self.set_output_val(0,torch.tensor(data=self.input(0),dtype=self.input(1),device = self.input(2), requires_grad=self.input(3)))
class torch_item_Node(Node):
    '''
    Returns the value of this tensor as a standard Python number. This only works for tensors with one element.
    '''
    title = "Get Tensor Item"
    tags = ['torch','item','tensor']
    init_inputs = [NodeInputBP(dtype=dtypes.Data(), label = "Tensor")]
    init_outputs = [NodeOutputBP()]
    color = "#D06B34"
    def update_event(self, inp=-1):
        self.set_output_val(0,torch.Tensor.item(self = self.input(0)))
class torch_tolist_Node(Node):
    '''Converts a tensor to a Python list'''
    title = 'torch.Tensor -> list'
    tags = ['tensor','list']
    init_inputs = [NodeInputBP(dtype=dtypes.Data(default=1), label="tensor")]
    init_outputs = [NodeOutputBP(label = 'list')]
    color ='#D06B34'
    def update_event(self, inp=-1):
        self.set_output_val(0,torch.Tensor.tolist(self = self.input(0)))
class torch_to_dtype(Node):
    '''Converts input tensor to the datatype provided'''
    
    title = 'tensor to dtype'
    tags = ['torch','tensor','dtype']
    init_inputs = [NodeInputBP(dtype=dtypes.Data(default=1), label = 'tensor'),
    NodeInputBP(dtype=dtypes.Data(default=1), label = 'dtype')]
    init_outputs = [NodeOutputBP()]
    color ='#D06B34'
    def update_event(self, inp=-1):
        datatypes = [{"float32":torch.float32},{"float16":torch.float16},{"float64":torch.float64},{"bfloat16":torch.bfloat16},{"complex32":torch.complex32},{"complex64":torch.complex64},{"complex128":torch.complex128},{"uint8":torch.uint8},{"int8":torch.int8},{"int16":torch.int16},{"int32":torch.int32},{"int64":torch.int64},{"bool":torch.bool},{"quint8":torch.quint8},{"quint4x2":torch.quint4x2},{"qint8":torch.qint8},{"qint32":torch.qint32}]
        if self.input(1) in [list(i.keys())[0] for i in datatypes]:
            i = [list(i.keys())[0] for i in datatypes].index(self.input(1))
            self.set_output_val(0,torch.Tensor.to(self = self.input(0),dtype = datatypes[i].get(self.input(1))))
        else:
            self.set_output_val(0,self.input(0))
torch_nodes =[torch_tensor_Node,torch_item_Node,torch_tolist_Node]
export_nodes(*torch_nodes)