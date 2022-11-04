import numpy as np

import mindspore as ms
import mindspore.dataset as ds

from .rand_augment import RandAugment


class OneHotOpFp32(ds.transforms.py_transforms.OneHotOp):
    def __init__(self, num_classes, smoothing_rate=0.0):
        super(OneHotOpFp32, self).__init__(num_classes, smoothing_rate)

    def __call__(self, label):
        return super(OneHotOpFp32, self).__call__(label).astype(np.float32)


class MixUpFp32(ds.vision.py_transforms.MixUp):
    def __init__(self, batch_size, alpha, is_single=True):
        super(MixUpFp32, self).__init__(batch_size, alpha, is_single)

    def __call__(self, image, label):
        mix_image, mix_label = super(MixUpFp32, self).__call__(image, label)
        return mix_image.astype(np.float32), mix_label.astype(np.float32)


def create_dataset(dataset_path, do_train, config, drop_remainder=True, enable_cache=False, cache_session_id=None):
    """
    create a train or eval dataset

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        config(struct): the config of train and eval in diffirent platform.
        enable_cache(bool): whether tensor caching service is used for dataset on nfs. Default: False
        cache_session_id(string): If enable_cache, cache session_id need to be provided. Default: None

    Returns:
        dataset
    """
    if enable_cache:
        if not cache_session_id:
            raise ValueError("A cache session_id must be provided to use cache.")
        nfs_dataset_cache = ds.DatasetCache(session_id=int(cache_session_id), size=0)
    else:
        nfs_dataset_cache = None

    num_parallel_workers = config.num_parallel_workers
    image_height = config.image_height
    image_width = config.image_width
    num_layers = config.rand_augment_num_layers
    magnitude = config.rand_augment_magnitude
    num_classes = config.num_classes
    batch_size = config.batch_size
    alpha = config.mixup_alpha
    smoothing_rate = config.label_smoothing

    if do_train:
        usage = "train"
    else:
        usage = "test"

    data_set = ds.Cifar10Dataset(dataset_path, usage=usage, num_parallel_workers=num_parallel_workers, shuffle=do_train,
                                 num_shards=config.rank_size, shard_id=config.rank_id, cache=nfs_dataset_cache)

    # define map operation
    resize_op = ds.vision.py_transforms.Resize((int(image_height + 32), int(image_width + 32)),
                                               interpolation=ds.vision.Inter.BICUBIC)
    # rand_resized_crop_op = ds.vision.py_transforms.RandomResizedCrop((image_height, image_width),
    #                                                                  scale=(0.08, 1.0),
    #                                                                  ratio=(3.0 / 4.0, 4.0 / 3.0),
    #                                                                  interpolation=ds.vision.Inter.BICUBIC)

    rand_horizontal_flip_op = ds.vision.py_transforms.RandomHorizontalFlip()

    rand_crop_op = ds.vision.py_transforms.RandomCrop((image_height, image_width))

    to_pil_op = ds.vision.py_transforms.ToPIL()

    rand_augment_op = RandAugment(num_layers=num_layers, magnitude=magnitude)

    # ToTenso() op also changes the format from HWC to CHW
    to_tensor_op = ds.vision.py_transforms.ToTensor()

    resize_to_image_size_op = ds.vision.py_transforms.Resize((image_height, image_width),
                                                             interpolation=ds.vision.Inter.BICUBIC)

    # input data must be in range [0.0, 1.0]; applied after ToTensor() op
    # normalize_op = ds.vision.c_transforms.Normalize(mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
    #                                                 std=[0.5 * 255, 0.5 * 255, 0.5 * 255])
    normalize_op = ds.vision.py_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                     std=[0.5, 0.5, 0.5])

    # hwc2chw_op = ds.vision.c_transforms.HWC2CHW()

    if do_train:
        trans = [to_pil_op, resize_op, rand_crop_op, rand_horizontal_flip_op, rand_augment_op, to_tensor_op, normalize_op]
        # trans = [to_pil_op, rand_resized_crop_op, rand_horizontal_flip_op, rand_augment_op, to_tensor_op, normalize_op]
    else:
        trans = [to_pil_op, resize_to_image_size_op, to_tensor_op, normalize_op]

    # type_cast_int32_op = ds.transforms.c_transforms.TypeCast(ms.dtype.int32)
    type_cast_int32_op = ds.vision.py_transforms.ToType(np.int32)

    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=num_parallel_workers)
    data_set = data_set.map(operations=type_cast_int32_op, input_columns="label", num_parallel_workers=num_parallel_workers)

    if do_train:
        one_hot_op = OneHotOpFp32(num_classes=num_classes, smoothing_rate=smoothing_rate)
        data_set = data_set.map(operations=one_hot_op, input_columns="label", num_parallel_workers=num_parallel_workers)

    # apply batch operations
    data_set = data_set.batch(batch_size=batch_size, drop_remainder=drop_remainder)

    if do_train:
        mixup_op = MixUpFp32(batch_size=batch_size, alpha=alpha)
        data_set = data_set.map(operations=mixup_op, input_columns=["image", "label"], num_parallel_workers=num_parallel_workers)

    return data_set
