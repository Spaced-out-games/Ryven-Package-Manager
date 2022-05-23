

class tensor(Node):
    """tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False) -> Tensor

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
    tensor([])"""
    title = 'tensor'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(tensor(*self.inputs))


class item(Node):
    """item() -> number

Returns the value of this tensor as a standard Python number. This only works
for tensors with one element. For other cases, see :meth:`~Tensor.tolist`.

This operation is not differentiable.

Example::

    >>> x = torch.tensor([1.0])
    >>> x.item()
    1.0"""
    title = 'item'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(item(*self.inputs))


class FloatTensor(Node):
    """"""
    title = 'FloatTensor'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(FloatTensor(*self.inputs))


class size(Node):
    """size() -> torch.Size

Returns the size of the :attr:`self` tensor. The returned value is a subclass of
:class:`tuple`.

Example::

    >>> torch.empty(3, 4, 5).size()
    torch.Size([3, 4, 5])"""
    title = 'size'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(size(*self.inputs))


class ndimension(Node):
    """ndimension() -> int

Alias for :meth:`~Tensor.dim()`"""
    title = 'ndimension'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(ndimension(*self.inputs))


class view(Node):
    """view(*shape) -> Tensor

Returns a new tensor with the same data as the :attr:`self` tensor but of a
different :attr:`shape`.

The returned tensor shares the same data and must have the same number
of elements, but may have a different size. For a tensor to be viewed, the new
view size must be compatible with its original size and stride, i.e., each new
view dimension must either be a subspace of an original dimension, or only span
across original dimensions :math:`d, d+1, \dots, d+k` that satisfy the following
contiguity-like condition that :math:`\forall i = d, \dots, d+k-1`,

.. math::

  \text{stride}[i] = \text{stride}[i+1] \times \text{size}[i+1]

Otherwise, it will not be possible to view :attr:`self` tensor as :attr:`shape`
without copying it (e.g., via :meth:`contiguous`). When it is unclear whether a
:meth:`view` can be performed, it is advisable to use :meth:`reshape`, which
returns a view if the shapes are compatible, and copies (equivalent to calling
:meth:`contiguous`) otherwise.

Args:
    shape (torch.Size or int...): the desired size

Example::

    >>> x = torch.randn(4, 4)
    >>> x.size()
    torch.Size([4, 4])
    >>> y = x.view(16)
    >>> y.size()
    torch.Size([16])
    >>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
    >>> z.size()
    torch.Size([2, 8])

    >>> a = torch.randn(1, 2, 3, 4)
    >>> a.size()
    torch.Size([1, 2, 3, 4])
    >>> b = a.transpose(1, 2)  # Swaps 2nd and 3rd dimension
    >>> b.size()
    torch.Size([1, 3, 2, 4])
    >>> c = a.view(1, 3, 2, 4)  # Does not change tensor layout in memory
    >>> c.size()
    torch.Size([1, 3, 2, 4])
    >>> torch.equal(b, c)
    False


.. function:: view(dtype) -> Tensor

Returns a new tensor with the same data as the :attr:`self` tensor but of a
different :attr:`dtype`. :attr:`dtype` must have the same number of bytes per
element as :attr:`self`'s dtype.

.. warning::

    This overload is not supported by TorchScript, and using it in a Torchscript
    program will cause undefined behavior.


Args:
    dtype (:class:`torch.dtype`): the desired dtype

Example::

    >>> x = torch.randn(4, 4)
    >>> x
    tensor([[ 0.9482, -0.0310,  1.4999, -0.5316],
            [-0.1520,  0.7472,  0.5617, -0.8649],
            [-2.4724, -0.0334, -0.2976, -0.8499],
            [-0.2109,  1.9913, -0.9607, -0.6123]])
    >>> x.dtype
    torch.float32

    >>> y = x.view(torch.int32)
    >>> y
    tensor([[ 1064483442, -1124191867,  1069546515, -1089989247],
            [-1105482831,  1061112040,  1057999968, -1084397505],
            [-1071760287, -1123489973, -1097310419, -1084649136],
            [-1101533110,  1073668768, -1082790149, -1088634448]],
        dtype=torch.int32)
    >>> y[0, 0] = 1000000000
    >>> x
    tensor([[ 0.0047, -0.0310,  1.4999, -0.5316],
            [-0.1520,  0.7472,  0.5617, -0.8649],
            [-2.4724, -0.0334, -0.2976, -0.8499],
            [-0.2109,  1.9913, -0.9607, -0.6123]])

    >>> x.view(torch.int16)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    RuntimeError: Viewing a tensor as a new dtype with a different number of bytes per element is not supported."""
    title = 'view'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(view(*self.inputs))


class tolist(Node):
    """tolist() -> list or number

Returns the tensor as a (nested) list. For scalars, a standard
Python number is returned, just like with :meth:`~Tensor.item`.
Tensors are automatically moved to the CPU first if necessary.

This operation is not differentiable.

Examples::

    >>> a = torch.randn(2, 2)
    >>> a.tolist()
    [[0.012766935862600803, 0.5415473580360413],
     [-0.08909505605697632, 0.7729271650314331]]
    >>> a[0,0].tolist()
    0.012766935862600803"""
    title = 'tolist'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(tolist(*self.inputs))


class numpy(Node):
    """numpy() -> numpy.ndarray

Returns :attr:`self` tensor as a NumPy :class:`ndarray`. This tensor and the
returned :class:`ndarray` share the same underlying storage. Changes to
:attr:`self` tensor will be reflected in the :class:`ndarray` and vice versa."""
    title = 'numpy'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(numpy(*self.inputs))


class as_tensor(Node):
    """as_tensor(data, dtype=None, device=None) -> Tensor

Convert the data into a `torch.Tensor`. If the data is already a `Tensor` with the same `dtype` and `device`,
no copy will be performed, otherwise a new `Tensor` will be returned with computational graph retained if data
`Tensor` has ``requires_grad=True``. Similarly, if the data is an ``ndarray`` of the corresponding `dtype` and
the `device` is the cpu, no copy will be performed.

Args:
    data (array_like): Initial data for the tensor. Can be a list, tuple,
        NumPy ``ndarray``, scalar, and other types.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, infers data type from :attr:`data`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.

Example::

    >>> a = numpy.array([1, 2, 3])
    >>> t = torch.as_tensor(a)
    >>> t
    tensor([ 1,  2,  3])
    >>> t[0] = -1
    >>> a
    array([-1,  2,  3])

    >>> a = numpy.array([1, 2, 3])
    >>> t = torch.as_tensor(a, device=torch.device('cuda'))
    >>> t
    tensor([ 1,  2,  3])
    >>> t[0] = -1
    >>> a
    array([1,  2,  3])"""
    title = 'as tensor'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(as_tensor(*self.inputs))


class backward(Node):
    """Computes the sum of gradients of given tensors with respect to graph
leaves.

The graph is differentiated using the chain rule. If any of ``tensors``
are non-scalar (i.e. their data has more than one element) and require
gradient, then the Jacobian-vector product would be computed, in this
case the function additionally requires specifying ``grad_tensors``.
It should be a sequence of matching length, that contains the "vector"
in the Jacobian-vector product, usually the gradient of the differentiated
function w.r.t. corresponding tensors (``None`` is an acceptable value for
all tensors that don't need gradient tensors).

This function accumulates gradients in the leaves - you might need to zero
``.grad`` attributes or set them to ``None`` before calling it.
See :ref:`Default gradient layouts<default-grad-layouts>`
for details on the memory layout of accumulated gradients.

.. note::
    Using this method with ``create_graph=True`` will create a reference cycle
    between the parameter and its gradient which can cause a memory leak.
    We recommend using ``autograd.grad`` when creating the graph to avoid this.
    If you have to use this function, make sure to reset the ``.grad`` fields of your
    parameters to ``None`` after use to break the cycle and avoid the leak.

.. note::

    If you run any forward ops, create ``grad_tensors``, and/or call ``backward``
    in a user-specified CUDA stream context, see
    :ref:`Stream semantics of backward passes<bwd-cuda-stream-semantics>`.

Args:
    tensors (Sequence[Tensor] or Tensor): Tensors of which the derivative will be
        computed.
    grad_tensors (Sequence[Tensor or None] or Tensor, optional): The "vector" in
        the Jacobian-vector product, usually gradients w.r.t. each element of
        corresponding tensors. None values can be specified for scalar Tensors or
        ones that don't require grad. If a None value would be acceptable for all
        grad_tensors, then this argument is optional.
    retain_graph (bool, optional): If ``False``, the graph used to compute the grad
        will be freed. Note that in nearly all cases setting this option to ``True``
        is not needed and often can be worked around in a much more efficient
        way. Defaults to the value of ``create_graph``.
    create_graph (bool, optional): If ``True``, graph of the derivative will
        be constructed, allowing to compute higher order derivative products.
        Defaults to ``False``.
    inputs (Sequence[Tensor] or Tensor, optional): Inputs w.r.t. which the gradient
        be will accumulated into ``.grad``. All other Tensors will be ignored. If
        not provided, the gradient is accumulated into all the leaf Tensors that
        were used to compute the attr::tensors. All the provided inputs must be leaf
        Tensors."""
    title = 'backward'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(backward(*self.inputs))


class grad(Node):
    """Computes and returns the sum of gradients of outputs with respect to
the inputs.

``grad_outputs`` should be a sequence of length matching ``output``
containing the "vector" in Jacobian-vector product, usually the pre-computed
gradients w.r.t. each of the outputs. If an output doesn't require_grad,
then the gradient can be ``None``).

If ``only_inputs`` is ``True``, the function will only return a list of gradients
w.r.t the specified inputs. If it's ``False``, then gradient w.r.t. all remaining
leaves will still be computed, and will be accumulated into their ``.grad``
attribute.

.. note::

    If you run any forward ops, create ``grad_outputs``, and/or call ``grad``
    in a user-specified CUDA stream context, see
    :ref:`Stream semantics of backward passes<bwd-cuda-stream-semantics>`.

Args:
    outputs (sequence of Tensor): outputs of the differentiated function.
    inputs (sequence of Tensor): Inputs w.r.t. which the gradient will be
        returned (and not accumulated into ``.grad``).
    grad_outputs (sequence of Tensor): The "vector" in the Jacobian-vector product.
        Usually gradients w.r.t. each output. None values can be specified for scalar
        Tensors or ones that don't require grad. If a None value would be acceptable
        for all grad_tensors, then this argument is optional. Default: None.
    retain_graph (bool, optional): If ``False``, the graph used to compute the grad
        will be freed. Note that in nearly all cases setting this option to ``True``
        is not needed and often can be worked around in a much more efficient
        way. Defaults to the value of ``create_graph``.
    create_graph (bool, optional): If ``True``, graph of the derivative will
        be constructed, allowing to compute higher order derivative products.
        Default: ``False``.
    allow_unused (bool, optional): If ``False``, specifying inputs that were not
        used when computing outputs (and therefore their grad is always zero)
        is an error. Defaults to ``False``."""
    title = 'grad'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(grad(*self.inputs))


class device(Node):
    """"""
    title = 'device'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(device(*self.inputs))


class detach(Node):
    """"""
    title = 'detach'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(detach(*self.inputs))


class empty(Node):
    """empty(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False, memory_format=torch.contiguous_format) -> Tensor

Returns a tensor filled with uninitialized data. The shape of the tensor is
defined by the variable argument :attr:`size`.

Args:
    size (int...): a sequence of integers defining the shape of the output tensor.
        Can be a variable number of arguments or a collection like a list or tuple.

Keyword args:
    out (Tensor, optional): the output tensor.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
    layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
        Default: ``torch.strided``.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    pin_memory (bool, optional): If set, returned tensor would be allocated in
        the pinned memory. Works only for CPU tensors. Default: ``False``.
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.contiguous_format``.

Example::

    >>> a=torch.empty((2,3), dtype=torch.int32, device = 'cuda')
    >>> torch.empty_like(a)
    tensor([[0, 0, 0],
            [0, 0, 0]], device='cuda:0', dtype=torch.int32)"""
    title = 'empty'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(empty(*self.inputs))


class enable grad(Node):
    """Context-manager that enables gradient calculation.

Enables gradient calculation, if it has been disabled via :class:`~no_grad`
or :class:`~set_grad_enabled`.

This context manager is thread local; it will not affect computation
in other threads.

Also functions as a decorator. (Make sure to instantiate with parenthesis.)

.. note::
    enable_grad is one of several mechanisms that can enable or
    disable gradients locally see :ref:`locally-disable-grad-doc` for
    more information on how they compare.

Example::

    >>> x = torch.tensor([1.], requires_grad=True)
    >>> with torch.no_grad():
    ...   with torch.enable_grad():
    ...     y = x * 2
    >>> y.requires_grad
    True
    >>> y.backward()
    >>> x.grad
    >>> @torch.enable_grad()
    ... def doubler(x):
    ...     return x * 2
    >>> with torch.no_grad():
    ...     z = doubler(x)
    >>> z.requires_grad
    True"""
    title = 'enable grad'
    init_inputs = []
    init_outputs = [NodeOutputBP()]
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(enable_grad(*self.inputs))


class is_grad_enabled(Node):
    """is_grad_enabled() -> (bool)

Returns True if grad mode is currently enabled."""
    title = 'is grad enabled'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(is_grad_enabled(*self.inputs))


class from_numpy(Node):
    """from_numpy(ndarray) -> Tensor

Creates a :class:`Tensor` from a :class:`numpy.ndarray`.

The returned tensor and :attr:`ndarray` share the same memory. Modifications to
the tensor will be reflected in the :attr:`ndarray` and vice versa. The returned
tensor is not resizable.

It currently accepts :attr:`ndarray` with dtypes of ``numpy.float64``,
``numpy.float32``, ``numpy.float16``, ``numpy.complex64``, ``numpy.complex128``,
``numpy.int64``, ``numpy.int32``, ``numpy.int16``, ``numpy.int8``, ``numpy.uint8``,
and ``numpy.bool``.

Example::

    >>> a = numpy.array([1, 2, 3])
    >>> t = torch.from_numpy(a)
    >>> t
    tensor([ 1,  2,  3])
    >>> t[0] = -1
    >>> a
    array([-1,  2,  3])"""
    title = 'from numpy'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(from_numpy(*self.inputs))


class from_file(Node):
    """"""
    title = 'from file'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(from_file(*self.inputs))


class gru(Node):
    """"""
    title = 'gru'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(gru(*self.inputs))


class gru_cell(Node):
    """"""
    title = 'gru cell'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(gru_cell(*self.inputs))


class is_grad_enabled(Node):
    """is_grad_enabled() -> (bool)

Returns True if grad mode is currently enabled."""
    title = 'is grad enabled'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(is_grad_enabled(*self.inputs))


class layer_norm(Node):
    """"""
    title = 'layer norm'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(layer_norm(*self.inputs))


class lstm(Node):
    """"""
    title = 'lstm'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(lstm(*self.inputs))


class lstm_cell(Node):
    """"""
    title = 'lstm cell'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(lstm_cell(*self.inputs))


class empty like(Node):
    """empty_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor

Returns an uninitialized tensor with the same size as :attr:`input`.
``torch.empty_like(input)`` is equivalent to
``torch.empty(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.

Args:
    input (Tensor): the size of :attr:`input` will determine size of the output tensor.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned Tensor.
        Default: if ``None``, defaults to the dtype of :attr:`input`.
    layout (:class:`torch.layout`, optional): the desired layout of returned tensor.
        Default: if ``None``, defaults to the layout of :attr:`input`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, defaults to the device of :attr:`input`.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.preserve_format``.

Example::

    >>> torch.empty((2,3), dtype=torch.int64)
    tensor([[ 9.4064e+13,  2.8000e+01,  9.3493e+13],
            [ 7.5751e+18,  7.1428e+18,  7.5955e+18]])"""
    title = 'empty like'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(empty_like(*self.inputs))


class zeros like(Node):
    """zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor

Returns a tensor filled with the scalar value `0`, with the same size as
:attr:`input`. ``torch.zeros_like(input)`` is equivalent to
``torch.zeros(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.

.. warning::
    As of 0.4, this function does not support an :attr:`out` keyword. As an alternative,
    the old ``torch.zeros_like(input, out=output)`` is equivalent to
    ``torch.zeros(input.size(), out=output)``.

Args:
    input (Tensor): the size of :attr:`input` will determine size of the output tensor.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned Tensor.
        Default: if ``None``, defaults to the dtype of :attr:`input`.
    layout (:class:`torch.layout`, optional): the desired layout of returned tensor.
        Default: if ``None``, defaults to the layout of :attr:`input`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, defaults to the device of :attr:`input`.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.preserve_format``.

Example::

    >>> input = torch.empty(2, 3)
    >>> torch.zeros_like(input)
    tensor([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.]])"""
    title = 'zeros like'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(zeros_like(*self.inputs))


class ones like(Node):
    """ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor

Returns a tensor filled with the scalar value `1`, with the same size as
:attr:`input`. ``torch.ones_like(input)`` is equivalent to
``torch.ones(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.

.. warning::
    As of 0.4, this function does not support an :attr:`out` keyword. As an alternative,
    the old ``torch.ones_like(input, out=output)`` is equivalent to
    ``torch.ones(input.size(), out=output)``.

Args:
    input (Tensor): the size of :attr:`input` will determine size of the output tensor.

Keyword arguments:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned Tensor.
        Default: if ``None``, defaults to the dtype of :attr:`input`.
    layout (:class:`torch.layout`, optional): the desired layout of returned tensor.
        Default: if ``None``, defaults to the layout of :attr:`input`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, defaults to the device of :attr:`input`.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.preserve_format``.

Example::

    >>> input = torch.empty(2, 3)
    >>> torch.ones_like(input)
    tensor([[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]])"""
    title = 'ones like'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(ones_like(*self.inputs))


class Adam(Node):
    """Implements Adam algorithm.

It has been proposed in `Adam: A Method for Stochastic Optimization`_.
The implementation of the L2 penalty follows changes proposed in
`Decoupled Weight Decay Regularization`_.

Args:
    params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
    lr (float, optional): learning rate (default: 1e-3)
    betas (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.999))
    eps (float, optional): term added to the denominator to improve
        numerical stability (default: 1e-8)
    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    amsgrad (boolean, optional): whether to use the AMSGrad variant of this
        algorithm from the paper `On the Convergence of Adam and Beyond`_
        (default: False)

.. _Adam\: A Method for Stochastic Optimization:
    https://arxiv.org/abs/1412.6980
.. _Decoupled Weight Decay Regularization:
    https://arxiv.org/abs/1711.05101
.. _On the Convergence of Adam and Beyond:
    https://openreview.net/forum?id=ryQu7f-RZ"""
    title = 'Adam'
    init_inputs = [NodeInputBP(label = 'self'), NodeInputBP(label = 'params'), NodeInputBP(label = 'lr'), NodeInputBP(label = 'betas'), NodeInputBP(label = 'eps'), NodeInputBP(label = 'weight_decay'), NodeInputBP(label = 'amsgrad')]
    init_outputs = [NodeOutputBP()]
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(Adam(*self.inputs))


class randn(Node):
    """randn(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a tensor filled with random numbers from a normal distribution
with mean `0` and variance `1` (also called the standard normal
distribution).

.. math::
    \text{out}_{i} \sim \mathcal{N}(0, 1)

The shape of the tensor is defined by the variable argument :attr:`size`.

Args:
    size (int...): a sequence of integers defining the shape of the output tensor.
        Can be a variable number of arguments or a collection like a list or tuple.

Keyword args:
    generator (:class:`torch.Generator`, optional): a pseudorandom number generator for sampling
    out (Tensor, optional): the output tensor.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
    layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
        Default: ``torch.strided``.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.

Example::

    >>> torch.randn(4)
    tensor([-2.1436,  0.9966,  2.3426, -0.6366])
    >>> torch.randn(2, 3)
    tensor([[ 1.5954,  2.8929, -1.0923],
            [ 1.1719, -0.4709, -0.1996]])"""
    title = 'randn'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(randn(*self.inputs))


class randn like(Node):
    """randn_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor

Returns a tensor with the same size as :attr:`input` that is filled with
random numbers from a normal distribution with mean 0 and variance 1.
``torch.randn_like(input)`` is equivalent to
``torch.randn(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.

Args:
    input (Tensor): the size of :attr:`input` will determine size of the output tensor.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned Tensor.
        Default: if ``None``, defaults to the dtype of :attr:`input`.
    layout (:class:`torch.layout`, optional): the desired layout of returned tensor.
        Default: if ``None``, defaults to the layout of :attr:`input`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, defaults to the device of :attr:`input`.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.preserve_format``."""
    title = 'randn like'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(randn_like(*self.inputs))


class randint(Node):
    """randint(low=0, high, size, \*, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a tensor filled with random integers generated uniformly
between :attr:`low` (inclusive) and :attr:`high` (exclusive).

The shape of the tensor is defined by the variable argument :attr:`size`.

.. note::
    With the global dtype default (``torch.float32``), this function returns
    a tensor with dtype ``torch.int64``.

Args:
    low (int, optional): Lowest integer to be drawn from the distribution. Default: 0.
    high (int): One above the highest integer to be drawn from the distribution.
    size (tuple): a tuple defining the shape of the output tensor.

Keyword args:
    generator (:class:`torch.Generator`, optional): a pseudorandom number generator for sampling
    out (Tensor, optional): the output tensor.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
    layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
        Default: ``torch.strided``.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.

Example::

    >>> torch.randint(3, 5, (3,))
    tensor([4, 3, 4])


    >>> torch.randint(10, (2, 2))
    tensor([[0, 2],
            [5, 5]])


    >>> torch.randint(3, 10, (2, 2))
    tensor([[4, 5],
            [6, 7]])"""
    title = 'randint'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(randint(*self.inputs))


class randint like(Node):
    """randint_like(input, low=0, high, \*, dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor

Returns a tensor with the same shape as Tensor :attr:`input` filled with
random integers generated uniformly between :attr:`low` (inclusive) and
:attr:`high` (exclusive).

.. note:
    With the global dtype default (``torch.float32``), this function returns
    a tensor with dtype ``torch.int64``.

Args:
    input (Tensor): the size of :attr:`input` will determine size of the output tensor.
    low (int, optional): Lowest integer to be drawn from the distribution. Default: 0.
    high (int): One above the highest integer to be drawn from the distribution.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned Tensor.
        Default: if ``None``, defaults to the dtype of :attr:`input`.
    layout (:class:`torch.layout`, optional): the desired layout of returned tensor.
        Default: if ``None``, defaults to the layout of :attr:`input`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, defaults to the device of :attr:`input`.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.preserve_format``."""
    title = 'randint like'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(randint_like(*self.inputs))


class reshape(Node):
    """reshape(input, shape) -> Tensor

Returns a tensor with the same data and number of elements as :attr:`input`,
but with the specified shape. When possible, the returned tensor will be a view
of :attr:`input`. Otherwise, it will be a copy. Contiguous inputs and inputs
with compatible strides can be reshaped without copying, but you should not
depend on the copying vs. viewing behavior.

See :meth:`torch.Tensor.view` on when it is possible to return a view.

A single dimension may be -1, in which case it's inferred from the remaining
dimensions and the number of elements in :attr:`input`.

Args:
    input (Tensor): the tensor to be reshaped
    shape (tuple of ints): the new shape

Example::

    >>> a = torch.arange(4.)
    >>> torch.reshape(a, (2, 2))
    tensor([[ 0.,  1.],
            [ 2.,  3.]])
    >>> b = torch.tensor([[0, 1], [2, 3]])
    >>> torch.reshape(b, (-1,))
    tensor([ 0,  1,  2,  3])"""
    title = 'reshape'
    
    
    color = '#D06B34'
    
    def update_event(self, inp = -1):
        		self.set_output_value(reshape(*self.inputs))
