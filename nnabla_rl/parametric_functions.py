# Copyright 2021,2022 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional, Tuple

import numpy as np

import nnabla as nn
import nnabla.functions as NF
import nnabla.parametric_functions as NPF
import nnabla_rl.functions as RF
from nnabla.initializer import ConstantInitializer
from nnabla.parameter import get_parameter_or_create
from nnabla.parametric_functions import parametric_function_api
from nnabla_rl.initializers import HeUniform


def noisy_net(inp: nn.Variable,
              n_outmap: int,
              base_axis: int = 1,
              w_init: Optional[Callable[[Tuple[int, ...]], np.ndarray]] = None,
              b_init: Optional[Callable[[Tuple[int, ...]], np.ndarray]] = None,
              noisy_w_init: Optional[Callable[[Tuple[int, ...]], np.ndarray]] = None,
              noisy_b_init: Optional[Callable[[Tuple[int, ...]], np.ndarray]] = None,
              fix_parameters: bool = False,
              rng: Optional[np.random.RandomState] = None,
              with_bias: bool = True,
              with_noisy_bias: bool = True,
              apply_w: Optional[Callable[[nn.Variable], nn.Variable]] = None,
              apply_b: Optional[Callable[[nn.Variable], nn.Variable]] = None,
              apply_noisy_w: Optional[Callable[[nn.Variable], nn.Variable]] = None,
              apply_noisy_b: Optional[Callable[[nn.Variable], nn.Variable]] = None,
              seed: int = -1) -> nn.Variable:
    '''
    Noisy linear layer with factorized gaussian noise  proposed by Fortunato et al. in the paper
    "Noisy networks for exploration". See: https://arxiv.org/abs/1706.10295 for details.

    Args:
        inp (nn.Variable): Input of the layer n_outmaps (int): output dimension of the layer.
        n_outmap (int): Output dimension of the layer.
        base_axis (int): Axis of the input to treat as sample dimensions. Dimensions up to base_axis will be treated
            as sample dimensions. Defaults to 1.
        w_init (None or Callable[[Tuple[int, ...]], np.ndarray]): Initializer of weights used in deterministic stream.
            Defaults to None. If None, will be initialized with Uniform distribution
            :math:`(-\\frac{1}{\\sqrt{fanin}},\\frac{1}{\\sqrt{fanin}})`.
        b_init (None or Callable[[Tuple[int, ...]], np.ndarray]): Initializer of bias used in deterministic stream.
            Defaults to None. If None, will be initialized with Uniform distribution
            :math:`(-\\frac{1}{\\sqrt{fanin}},\\frac{1}{\\sqrt{fanin}})`.
        noisy_w_init (None or Callable[[Tuple[int, ...]], np.ndarray]): Initializer of weights used in noisy stream.
            Defaults to None. If None, will be initialized to a constant value of :math:`\\frac{0.5}{\\sqrt{fanin}}`.
        noisy_b_init (None or Callable[[Tuple[int, ...]], np.ndarray]): Initializer of bias used in noisy stream.
            Defaults to None. If None, will be initialized to a constant value of :math:`\\frac{0.5}{\\sqrt{fanin}}`.
        fix_parameters (bool): If True, underlying weight and bias parameters will Not be updated during training.
            Default to False.
        rng (None or np.random.RandomState): Random number generator for parameter initializer. Defaults to None.
        with_bias (bool): If True, deterministic bias term is included in the computation. Defaults to True.
        with_noisy_bias (bool): If True, noisy bias term is included in the computation. Defaults to True.
        apply_w (None or Callable[[nn.Variable], nn.Variable]): Callable object to apply to the weights on
            initialization. Defaults to None.
        apply_b (None or Callable[[nn.Variable], nn.Variable]): Callable object to apply to the bias on
            initialization. Defaults to None.
        apply_noisy_w (None or Callable[[nn.Variable], nn.Variable]):  Callable object to apply to the noisy weight on
            initialization. Defaults to None.
        apply_noisy_b (None or Callable[[nn.Variable], nn.Variable]):  Callable object to apply to the noisy bias on
            initialization. Defaults to None.
        seed (int): Random seed. If -1, seed will be sampled from global random number generator. Defaults to -1.

    Returns:
        nn.Variable: Linearly transformed input with noisy weights
    '''

    inmaps = int(np.prod(inp.shape[base_axis:]))
    if w_init is None:
        w_init = HeUniform(inmaps, n_outmap, factor=1.0/3.0, rng=rng)
    if noisy_w_init is None:
        noisy_w_init = ConstantInitializer(0.5 / np.sqrt(inmaps))
    w = get_parameter_or_create("W", (inmaps, n_outmap), w_init, True, not fix_parameters)
    if apply_w is not None:
        w = apply_w(w)

    noisy_w = get_parameter_or_create("noisy_W", (inmaps, n_outmap), noisy_w_init, True, not fix_parameters)
    if apply_noisy_w is not None:
        noisy_w = apply_noisy_w(noisy_w)

    b = None
    if with_bias:
        if b_init is None:
            b_init = HeUniform(inmaps, n_outmap, factor=1.0/3.0, rng=rng)
        b = get_parameter_or_create("b", (n_outmap, ), b_init, True, not fix_parameters)
        if apply_b is not None:
            b = apply_b(b)

    noisy_b = None
    if with_noisy_bias:
        if noisy_b_init is None:
            noisy_b_init = ConstantInitializer(0.5 / np.sqrt(inmaps))
        noisy_b = get_parameter_or_create("noisy_b", (n_outmap, ), noisy_b_init, True, not fix_parameters)
        if apply_noisy_b is not None:
            noisy_b = apply_noisy_b(noisy_b)

    def _f(x):
        return NF.sign(x) * RF.sqrt(NF.abs(x))

    e_i = _f(NF.randn(shape=(1, inmaps, 1), seed=seed))
    e_j = _f(NF.randn(shape=(1, 1, n_outmap), seed=seed))

    e_w = NF.reshape(NF.batch_matmul(e_i, e_j), shape=noisy_w.shape)
    e_w.need_grad = False
    noisy_w = noisy_w * e_w
    assert noisy_w.shape == w.shape

    if with_noisy_bias:
        assert isinstance(noisy_b, nn.Variable)
        e_b = NF.reshape(e_j, shape=noisy_b.shape)
        e_b.need_grad = False
        noisy_b = noisy_b * e_b
        assert noisy_b.shape == (n_outmap,)
    weight = w + noisy_w

    if with_bias and with_noisy_bias:
        assert isinstance(b, nn.Variable)
        assert isinstance(noisy_b, nn.Variable)
        bias = b + noisy_b
    elif with_bias:
        bias = b
    elif with_noisy_bias:
        bias = noisy_b
    else:
        bias = None
    return NF.affine(inp, weight, bias, base_axis)


def spatial_softmax(inp: nn.Variable, alpha_init: float = 1., fix_alpha: bool = False) -> nn.Variable:
    r''' Spatial softmax layer proposed in https://arxiv.org/abs/1509.06113. Computes

    .. math::
        s_{cij} &= \frac{\exp(x_{cij} / \alpha)}{\sum_{i'j'} \exp(x_{ci'j'} / \alpha)}

        f_{cx} &= \sum_{ij} s_{cij}px_{ij}, f_{cy} = \sum_{ij} s_{cij}py_{ij}

        y_{c} &= (f_{cx}, f_{cy})

    where :math:`x, y, \\alpha` are the input, output and parameter respectively,
    and :math:`c, i, j` are the number of channels, heights and widths respectively.
    :math:`(px_{ij}, py_{ij})` is the image-space position of the point (i, j) in the response map.

    Args:
        inp (nn.Variables): Input of the layer. Shape should be (batch_size, C, H, W)
        alpha_init (float): Initial temperature value. Defaults to 1.
        fix_alpha (bool): If True, underlying alpha will Not be updated during training.
            Defaults to False.

    Returns:
        nn.Variables: Feature points, Shape is (batch_size, C*2)
    '''
    assert len(inp.shape) == 4
    (batch_size, channel, height, width) = inp.shape
    alpha = get_parameter_or_create("alpha", shape=(1, 1), initializer=ConstantInitializer(alpha_init),
                                    need_grad=True, as_need_grad=not fix_alpha)

    features = NF.reshape(inp, (-1, height*width))
    softmax_attention = NF.softmax(features / alpha)

    # Image positions are normalized and defined by -1 to 1.
    # This normalization is referring to the original Guided Policy Search implementation.
    # See: https://github.com/cbfinn/gps/blob/master/python/gps/algorithm/policy_opt/tf_model_example.py#L238
    pos_x, pos_y = np.meshgrid(np.linspace(-1., 1., height), np.linspace(-1., 1., width))
    pos_x = nn.Variable.from_numpy_array(pos_x.reshape(-1, (height*width)))
    pos_y = nn.Variable.from_numpy_array(pos_y.reshape(-1, (height*width)))

    expected_x = NF.sum(pos_x*softmax_attention, axis=1, keepdims=True)
    expected_y = NF.sum(pos_y*softmax_attention, axis=1, keepdims=True)
    expected_xy = NF.concatenate(expected_x, expected_y, axis=1)

    feature_points = NF.reshape(expected_xy, (batch_size, channel*2))

    return feature_points


@parametric_function_api("lstm", [
    ('affine/W', 'Stacked weight matrixes of LSTM block',
     '(inmaps, 4, state_size)', True),
    ('affine/b', 'Stacked bias vectors of LSTM block', '(4, state_size,)', True),
])
def lstm_cell(x, h, c, state_size, w_init=None, b_init=None, fix_parameters=False, base_axis=1):
    """Long Short-Term Memory with base_axis.

    Long Short-Term Memory, or LSTM, is a building block for recurrent neural networks (RNN) layers.
    LSTM unit consists of a cell and input, output, forget gates whose functions are defined as following:

    .. math::
        f_t&&=\\sigma(W_fx_t+U_fh_{t-1}+b_f) \\\\
        i_t&&=\\sigma(W_ix_t+U_ih_{t-1}+b_i) \\\\
        o_t&&=\\sigma(W_ox_t+U_oh_{t-1}+b_o) \\\\
        c_t&&=f_t\\odot c_{t-1}+i_t\\odot\\tanh(W_cx_t+U_ch_{t-1}+b_c) \\\\
        h_t&&=o_t\\odot\\tanh(c_t).

    References:

        S. Hochreiter, and J. Schmidhuber. "Long Short-Term Memory."
        Neural Computation. 1997.

    Args:
        x (~nnabla.Variable): Input N-D array with shape (batch_size, input_size).
        h (~nnabla.Variable): Input N-D array with shape (batch_size, state_size).
        c (~nnabla.Variable): Input N-D array with shape (batch_size, state_size).
        state_size (int): Internal state size is set to `state_size`.
        w_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`, optional): Initializer for weight.
            By default, it is initialized with :obj:`nnabla.initializer.UniformInitializer`
            within the range determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`.
        b_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`, optional): Initializer for bias.
            By default, it is initialized with zeros if `with_bias` is `True`.
        fix_parameters (bool): When set to `True`, the weights and biases will not be updated.
        base_axis (int): Dimensions up to base_axis are treated as sample dimensions.

    Returns:
        :class:`~nnabla.Variable`

    """

    xh = NF.concatenate(*(x, h), axis=base_axis)
    iofc = NPF.affine(xh, (4, state_size), base_axis=base_axis, w_init=w_init,
                      b_init=b_init, fix_parameters=fix_parameters)
    i_t, o_t, f_t, gate = NF.split(iofc, axis=base_axis)
    c_t = NF.sigmoid(f_t) * c + NF.sigmoid(i_t) * NF.tanh(gate)
    h_t = NF.sigmoid(o_t) * NF.tanh(c_t)
    return h_t, c_t
