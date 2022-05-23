from ryven.NENV import *
import torch#from torch import *?
torch.Tensor.item()
#"#D06B34"
class Torch_tensorNode(Node):
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
class Torch_Tensor_Item_Node(Node):
    '''
    '''
torch_nodes =[Torch_tensorNode]
export_nodes(*torch_nodes)