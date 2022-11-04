# Lint as: python3
"""Bottleneck ResNet v2 with GroupNorm and Weight Standardization."""

from collections import OrderedDict  # pylint: disable=g-importing-member
import mindspore as ms
import mindspore.nn as nn

from .padding import ConstantPad2d


class ReduceMean(nn.Cell):
  def __init__(self, axis):
    super(ReduceMean, self).__init__()
    self.axis = axis

  def construct(self, x):
    x = ms.ops.ReduceMean(keep_dims=True)(x, self.axis)
    return x


class StdConv2d(nn.Conv2d):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad_mode="same", padding=0, dilation=1, group=1, has_bias=False, weight_init="normal", bias_init="zeros", data_format="NCHW"):
    super(StdConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, pad_mode, padding, dilation, group, has_bias, weight_init, bias_init, data_format)
    self.moments = nn.Moments(axis=(1,2,3), keep_dims=True)

  def construct(self, x):
    mean, var = self.moments(self.weight)
    weight = (self.weight - mean) / (var + 1e-10) ** 0.5
    x = self.conv2d(x, weight)
    if self.has_bias:
      x = self.bias_add(x, self.bias)
    return x


def conv3x3(in_channels, out_channels, stride=1, has_bias=False):
  return StdConv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                   pad_mode="same", has_bias=has_bias)


def conv1x1(in_channels, out_channels, stride=1, has_bias=False):
  return StdConv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                   pad_mode="same", has_bias=has_bias)


def tf2ms(conv_weights):
  """Possibly convert HWIO to OIHW."""
  if conv_weights.ndim == 4:
    conv_weights = conv_weights.transpose([3, 2, 0, 1])
  return ms.Tensor(input_data=conv_weights, dtype=ms.dtype.float32)



class PreActBottleneck(nn.Cell):
  """Pre-activation (v2) bottleneck block.

  Follows the implementation of "Identity Mappings in Deep Residual Networks":
  https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

  Except it puts the stride on 3x3 conv when available.
  """

  def __init__(self, in_channels, out_channels=None, hidden_channels=None, stride=1):
    super(PreActBottleneck, self).__init__()
    out_channels = out_channels or in_channels
    hidden_channels = hidden_channels or out_channels // 4

    self.gn1 = nn.GroupNorm(32, in_channels, 1e-5)
    self.conv1 = conv1x1(in_channels, hidden_channels)
    self.gn2 = nn.GroupNorm(32, hidden_channels, 1e-5)
    self.conv2 = conv3x3(hidden_channels, hidden_channels, stride)  # Original code has it on conv1!!
    self.gn3 = nn.GroupNorm(32, hidden_channels, 1e-5)
    self.conv3 = conv1x1(hidden_channels, out_channels)
    self.relu = nn.ReLU()

    self.gn1.recompute()
    self.gn2.recompute()
    self.gn3.recompute()

    self.has_downsample = False

    if (stride != 1 or in_channels != out_channels):
      # Projection also with pre-activation according to paper.
      self.downsample = conv1x1(in_channels, out_channels, stride)
      self.has_downsample = True

  def construct(self, x):
    out = self.relu(self.gn1(x))

    # Residual branch
    residual = x
    if self.has_downsample:
      residual = self.downsample(out)

    # Unit's branch
    out = self.conv1(out)
    out = self.conv2(self.relu(self.gn2(out)))
    out = self.conv3(self.relu(self.gn3(out)))

    return out + residual

  def load_from(self, weights, prefix=''):
    convname = 'standardized_conv2d'
    self.conv1.weight.set_data(tf2ms(weights[f'{prefix}a/{convname}/kernel']))
    self.conv2.weight.set_data(tf2ms(weights[f'{prefix}b/{convname}/kernel']))
    self.conv3.weight.set_data(tf2ms(weights[f'{prefix}c/{convname}/kernel']))
    self.gn1.gamma.set_data(tf2ms(weights[f'{prefix}a/group_norm/gamma']))
    self.gn2.gamma.set_data(tf2ms(weights[f'{prefix}b/group_norm/gamma']))
    self.gn3.gamma.set_data(tf2ms(weights[f'{prefix}c/group_norm/gamma']))
    self.gn1.beta.set_data(tf2ms(weights[f'{prefix}a/group_norm/beta']))
    self.gn2.beta.set_data(tf2ms(weights[f'{prefix}b/group_norm/beta']))
    self.gn3.beta.set_data(tf2ms(weights[f'{prefix}c/group_norm/beta']))
    if self.has_downsample:
      self.downsample.weight.set_data(tf2ms(weights[f'{prefix}a/proj/{convname}/kernel']))


class ResNetV2(nn.Cell):
  """Implementation of Pre-activation (v2) ResNet mode."""

  def __init__(self, block_units, width_factor, head_size=21843, zero_head=False):
    super().__init__()
    wf = width_factor  # shortcut 'cause we'll use it a lot.

    # The following will be unreadable if we split lines.
    # pylint: disable=line-too-long
    self.root = nn.SequentialCell(OrderedDict([
        ('pad1', ConstantPad2d(3, 0)),
        ('conv', StdConv2d(3, 64*wf, kernel_size=7, stride=2, pad_mode="valid")),
        ('pad2', ConstantPad2d(1, 0)),
        ('pool', nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")),
    ]))

    self.body = nn.SequentialCell(OrderedDict([
        ('block1', nn.SequentialCell(OrderedDict(
            [('unit01', PreActBottleneck(in_channels=64*wf, out_channels=256*wf, hidden_channels=64*wf))] +
            [(f'unit{i:02d}', PreActBottleneck(in_channels=256*wf, out_channels=256*wf, hidden_channels=64*wf)) for i in range(2, block_units[0] + 1)],
        ))),
        ('block2', nn.SequentialCell(OrderedDict(
            [('unit01', PreActBottleneck(in_channels=256*wf, out_channels=512*wf, hidden_channels=128*wf, stride=2))] +
            [(f'unit{i:02d}', PreActBottleneck(in_channels=512*wf, out_channels=512*wf, hidden_channels=128*wf)) for i in range(2, block_units[1] + 1)],
        ))),
        ('block3', nn.SequentialCell(OrderedDict(
            [('unit01', PreActBottleneck(in_channels=512*wf, out_channels=1024*wf, hidden_channels=256*wf, stride=2))] +
            [(f'unit{i:02d}', PreActBottleneck(in_channels=1024*wf, out_channels=1024*wf, hidden_channels=256*wf)) for i in range(2, block_units[2] + 1)],
        ))),
        ('block4', nn.SequentialCell(OrderedDict(
            [('unit01', PreActBottleneck(in_channels=1024*wf, out_channels=2048*wf, hidden_channels=512*wf, stride=2))] +
            [(f'unit{i:02d}', PreActBottleneck(in_channels=2048*wf, out_channels=2048*wf, hidden_channels=512*wf)) for i in range(2, block_units[3] + 1)],
        ))),
    ]))
    # pylint: enable=line-too-long

    self.zero_head = zero_head
    self.head = nn.SequentialCell(OrderedDict([
        ('gn', nn.GroupNorm(32, 2048*wf, 1e-5)),
        ('relu', nn.ReLU()),
        ('avg', ReduceMean((2,3))),
        ('conv', nn.Conv2d(2048*wf, head_size, kernel_size=1, has_bias=True)),
    ]))

  def construct(self, x):
    x = self.head(self.body(self.root(x)))
    x = ms.ops.Squeeze((2,3))(x)
    return x

  def load_from(self, weights, prefix='resnet/'):
    self.root.conv.weight.set_data(tf2ms(weights[f'{prefix}root_block/standardized_conv2d/kernel']))  # pylint: disable=line-too-long
    self.head.gn.gamma.set_data(tf2ms(weights[f'{prefix}group_norm/gamma']))
    self.head.gn.beta.set_data(tf2ms(weights[f'{prefix}group_norm/beta']))
    if self.zero_head:
      self.head.conv.weight.set_data(ms.common.initializer.initializer("zeros", shape=self.head.conv.weight.shape, dtype=ms.dtype.float32))
      self.head.conv.bias.set_data(ms.common.initializer.initializer("zeros", shape=self.head.conv.bias.shape, dtype=ms.dtype.float32))
    else:
      self.head.conv.weight.set_data(tf2ms(weights[f'{prefix}head/conv2d/kernel']))  # pylint: disable=line-too-long
      self.head.conv.bias.set_data(tf2ms(weights[f'{prefix}head/conv2d/bias']))

    for bname, block in self.body.name_cells().items():
      # print("+++", bname, block)
      for uname, unit in block.name_cells().items():
        # print("***", uname, unit)
        unit.load_from(weights, prefix=f'{prefix}{bname}/{uname}/')


KNOWN_MODELS = OrderedDict([
    ('BiT-M-R50x1', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
    ('BiT-M-R50x3', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
    ('BiT-M-R101x1', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
    ('BiT-M-R101x3', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
    ('BiT-M-R152x2', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
    ('BiT-M-R152x4', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
    ('BiT-S-R50x1', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
    ('BiT-S-R50x3', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
    ('BiT-S-R101x1', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
    ('BiT-S-R101x3', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
    ('BiT-S-R152x2', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
    ('BiT-S-R152x4', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
])
