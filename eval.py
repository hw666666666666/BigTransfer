# Copyright 2020-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""evaluate BiT model on CIFAR-10"""


import mindspore as ms
import mindspore.nn as nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.dataset import create_dataset
from src.utils import do_keep_cell_fp16, context_device_init, count_params
from src.model_utils.moxing_adapter import moxing_wrapper, modelarts_process
from src.model_utils.device_adapter import get_device_id
from src.model_utils.config import config
from src.models import KNOWN_MODELS


def process_checkpoint(net, ckpt):
    prefix = "ema."
    len_prefix = len(prefix)
    if config.enable_ema:
        ema_ckpt = {}
        for name, param in ckpt.items():
            if name.startswith(prefix):
                ema_ckpt[name[len_prefix:]] = ms.Parameter(default_input=param.data, name=param.name[len_prefix:])
        ckpt = ema_ckpt

    net_param_dict = net.parameters_dict()
    ckpt = {k:v for k, v in ckpt.items() if k in net_param_dict}

    return ckpt


@moxing_wrapper(pre_process=modelarts_process)
def eval():
    config.batch_size = 100
    # config.pretrain_ckpt = config.load_path
    # config.dataset_path = os.path.join(config.dataset_path, 'val')

    # https://arxiv.org/abs/1906.06423, "Fixing the train-test resolution discrepancy"
    config.image_height = config.image_height + 32
    config.image_width = config.image_width + 32

    if not config.device_id:
        config.device_id = get_device_id()
    context_device_init(config)
    print('\nconfig: {} \n'.format(config))

    net = KNOWN_MODELS[config.model_name](head_size=10, zero_head=True)

    ckpt = load_checkpoint(config.load_path)

    ckpt = process_checkpoint(net, ckpt)

    load_param_into_net(net, ckpt)

    net.to_float(ms.dtype.float32)
    # do_keep_cell_fp16(net, cell_types=(nn.Conv2d))

    dataset = create_dataset(dataset_path=config.dataset_path, do_train=False, config=config, drop_remainder=False)
    step_size = dataset.get_dataset_size()
    if step_size == 0:
        raise ValueError("The step_size of dataset is zero. Check if the images count of eval dataset is more \
            than batch_size in config.py")
    print("step_size = ", step_size)
    net.set_train(False)

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    metrics = {'Validation-Loss': nn.Loss(),
               'Top1-Acc': nn.Top1CategoricalAccuracy()}
    model = ms.Model(net, loss_fn=loss, metrics=metrics)

    res = model.eval(dataset)
    print("result:{}\npretrain_ckpt={}".format(res, config.load_path))


if __name__ == '__main__':
    eval()
