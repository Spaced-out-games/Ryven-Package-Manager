# coding=utf8
# the above tag defines encoding for this document and is for Python 2.x compatibility

import re

regex = r"    (([a-zA-Z]*)_?([a-zA-Z]*)?): .*"

test_str = ("Applies a multi-layer long short-term memory (LSTM) RNN to an input\n"
	"sequence.\n\n\n"
	"For each element in the input sequence, each layer computes the following\n"
	"function:\n\n"
	".. math::\n"
	"    \\begin{array}{ll} \\\\\n"
	"        i_t = \\sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\\\\n"
	"        f_t = \\sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\\\\n"
	"        g_t = \\tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\\\\n"
	"        o_t = \\sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\\\\n"
	"        c_t = f_t \\odot c_{t-1} + i_t \\odot g_t \\\\\n"
	"        h_t = o_t \\odot \\tanh(c_t) \\\\\n"
	"    \\end{array}\n\n"
	"where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the cell\n"
	"state at time `t`, :math:`x_t` is the input at time `t`, :math:`h_{t-1}`\n"
	"is the hidden state of the layer at time `t-1` or the initial hidden\n"
	"state at time `0`, and :math:`i_t`, :math:`f_t`, :math:`g_t`,\n"
	":math:`o_t` are the input, forget, cell, and output gates, respectively.\n"
	":math:`\\sigma` is the sigmoid function, and :math:`\\odot` is the Hadamard product.\n\n"
	"In a multilayer LSTM, the input :math:`x^{(l)}_t` of the :math:`l` -th layer\n"
	"(:math:`l >= 2`) is the hidden state :math:`h^{(l-1)}_t` of the previous layer multiplied by\n"
	"dropout :math:`\\delta^{(l-1)}_t` where each :math:`\\delta^{(l-1)}_t` is a Bernoulli random\n"
	"variable which is :math:`0` with probability :attr:`dropout`.\n\n"
	"If ``proj_size > 0`` is specified, LSTM with projections will be used. This changes\n"
	"the LSTM cell in the following way. First, the dimension of :math:`h_t` will be changed from\n"
	"``hidden_size`` to ``proj_size`` (dimensions of :math:`W_{hi}` will be changed accordingly).\n"
	"Second, the output hidden state of each layer will be multiplied by a learnable projection\n"
	"matrix: :math:`h_t = W_{hr}h_t`. Note that as a consequence of this, the output\n"
	"of LSTM network will be of different shape as well. See Inputs/Outputs sections below for exact\n"
	"dimensions of all variables. You can find more details in https://arxiv.org/abs/1402.1128.\n\n"
	"Args:\n"
	"    input_size: The number of expected features in the input `x`\n"
	"    hidden_size: The number of features in the hidden state `h`\n"
	"    num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``\n"
	"        would mean stacking two LSTMs together to form a `stacked LSTM`,\n"
	"        with the second LSTM taking in outputs of the first LSTM and\n"
	"        computing the final results. Default: 1\n"
	"    bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.\n"
	"        Default: ``True``\n"
	"    batch_first: If ``True``, then the input and output tensors are provided\n"
	"        as `(batch, seq, feature)` instead of `(seq, batch, feature)`.\n"
	"        Note that this does not apply to hidden or cell states. See the\n"
	"        Inputs/Outputs sections below for details.  Default: ``False``\n"
	"    dropout: If non-zero, introduces a `Dropout` layer on the outputs of each\n"
	"        LSTM layer except the last layer, with dropout probability equal to\n"
	"        :attr:`dropout`. Default: 0\n"
	"    bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``\n"
	"    proj_size: If ``> 0``, will use LSTM with projections of corresponding size. Default: 0\n\n"
	"Inputs: input, (h_0, c_0)\n"
	"    * **input**: tensor of shape :math:`(L, N, H_{in})` when ``batch_first=False`` or\n"
	"      :math:`(N, L, H_{in})` when ``batch_first=True`` containing the features of\n"
	"      the input sequence.  The input can also be a packed variable length sequence.\n"
	"      See :func:`torch.nn.utils.rnn.pack_padded_sequence` or\n"
	"      :func:`torch.nn.utils.rnn.pack_sequence` for details.\n"
	"    * **h_0**: tensor of shape :math:`(D * \\text{num\\_layers}, N, H_{out})` containing the\n"
	"      initial hidden state for each element in the batch.\n"
	"      Defaults to zeros if (h_0, c_0) is not provided.\n"
	"    * **c_0**: tensor of shape :math:`(D * \\text{num\\_layers}, N, H_{cell})` containing the\n"
	"      initial cell state for each element in the batch.\n"
	"      Defaults to zeros if (h_0, c_0) is not provided.\n\n"
	"    where:\n\n"
	"    .. math::\n"
	"        \\begin{aligned}\n"
	"            N ={} & \\text{batch size} \\\\\n"
	"            L ={} & \\text{sequence length} \\\\\n"
	"            D ={} & 2 \\text{ if bidirectional=True otherwise } 1 \\\\\n"
	"            H_{in} ={} & \\text{input\\_size} \\\\\n"
	"            H_{cell} ={} & \\text{hidden\\_size} \\\\\n"
	"            H_{out} ={} & \\text{proj\\_size if } \\text{proj\\_size}>0 \\text{ otherwise hidden\\_size} \\\\\n"
	"        \\end{aligned}\n\n"
	"Outputs: output, (h_n, c_n)\n"
	"    * **output**: tensor of shape :math:`(L, N, D * H_{out})` when ``batch_first=False`` or\n"
	"      :math:`(N, L, D * H_{out})` when ``batch_first=True`` containing the output features\n"
	"      `(h_t)` from the last layer of the LSTM, for each `t`. If a\n"
	"      :class:`torch.nn.utils.rnn.PackedSequence` has been given as the input, the output\n"
	"      will also be a packed sequence.\n"
	"    * **h_n**: tensor of shape :math:`(D * \\text{num\\_layers}, N, H_{out})` containing the\n"
	"      final hidden state for each element in the batch.\n"
	"    * **c_n**: tensor of shape :math:`(D * \\text{num\\_layers}, N, H_{cell})` containing the\n"
	"      final cell state for each element in the batch.\n\n"
	"Attributes:\n"
	"    weight_ih_l[k] : the learnable input-hidden weights of the :math:`\\text{k}^{th}` layer\n"
	"        `(W_ii|W_if|W_ig|W_io)`, of shape `(4*hidden_size, input_size)` for `k = 0`.\n"
	"        Otherwise, the shape is `(4*hidden_size, num_directions * hidden_size)`\n"
	"    weight_hh_l[k] : the learnable hidden-hidden weights of the :math:`\\text{k}^{th}` layer\n"
	"        `(W_hi|W_hf|W_hg|W_ho)`, of shape `(4*hidden_size, hidden_size)`. If ``proj_size > 0``\n"
	"        was specified, the shape will be `(4*hidden_size, proj_size)`.\n"
	"    bias_ih_l[k] : the learnable input-hidden bias of the :math:`\\text{k}^{th}` layer\n"
	"        `(b_ii|b_if|b_ig|b_io)`, of shape `(4*hidden_size)`\n"
	"    bias_hh_l[k] : the learnable hidden-hidden bias of the :math:`\\text{k}^{th}` layer\n"
	"        `(b_hi|b_hf|b_hg|b_ho)`, of shape `(4*hidden_size)`\n"
	"    weight_hr_l[k] : the learnable projection weights of the :math:`\\text{k}^{th}` layer\n"
	"        of shape `(proj_size, hidden_size)`. Only present when ``proj_size > 0`` was\n"
	"        specified.\n\n"
	".. note::\n"
	"    All the weights and biases are initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`\n"
	"    where :math:`k = \\frac{1}{\\text{hidden\\_size}}`\n\n"
	".. note::\n"
	"    For bidirectional LSTMs, forward and backward are directions 0 and 1 respectively.\n"
	"    Example of splitting the output layers when ``batch_first=False``:\n"
	"    ``output.view(seq_len, batch, num_directions, hidden_size)``.\n\n"
	".. include:: ../cudnn_rnn_determinism.rst\n\n"
	".. include:: ../cudnn_persistent_rnn.rst\n\n"
	"Examples::\n\n"
	"    >>> rnn = nn.LSTM(10, 20, 2)\n"
	"    >>> input = torch.randn(5, 3, 10)\n"
	"    >>> h0 = torch.randn(2, 3, 20)\n"
	"    >>> c0 = torch.randn(2, 3, 20)\n"
	"    >>> output, (hn, cn) = rnn(input, (h0, c0))")
doc = test_str
import torch

torch.nn.LSTM
start = r"([a-z]*: ?\n)    [a-z_]*:"