"""
@author: Junguang Jiang, Baixu Chen
@contact: JiangJunguang1123@outlook.com, cbx_99_hasta@outlook.com
"""
import sys
import os.path as osp
import time
from copy import deepcopy
from enum import Enum
from pathlib import Path

import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import ConcatDataset
import wilds

from examples.domain_adaptation.video_classification import video_transform
from examples.domain_adaptation.video_classification.dataset_access import DatasetAccess
from examples.domain_adaptation.video_classification.videos import VideoFrameDataset

sys.path.append('../../..')
import common.vision.datasets as datasets
import common.vision.models as models
# from common.vision.transforms import ResizeImage
# from common.utils.metric import accuracy, ConfusionMatrix
# from common.utils.meter import AverageMeter, ProgressMeter


def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()


def get_model(model_name, pretrain=True):
    if model_name in models.__dict__:
        # load models from common.vision.models
        backbone = models.__dict__[model_name](pretrained=pretrain)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=pretrain)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    return backbone


def convert_from_wilds_dataset(wild_dataset):
    class Dataset:
        def __init__(self):
            self.dataset = wild_dataset

        def __getitem__(self, idx):
            x, y, metadata = self.dataset[idx]
            return x, y

        def __len__(self):
            return len(self.dataset)

    return Dataset()


def get_dataset_names():
    return sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    ) + wilds.supported_datasets + ['Digits']


# def get_dataset(dataset_name, root, source, target, train_source_transform, val_transform, train_target_transform=None):
#     if dataset_name == "EPIC100":
#         src_data_path, src_tr_listpath, src_te_listpath = generate_list(data_src_name, data_params_local, domain="src")
#         data_tgt_name = data_params_local["dataset_tgt_name"].upper()
#         tgt_data_path, tgt_tr_listpath, tgt_te_listpath = generate_list(data_tgt_name, data_params_local, domain="tgt")
#         image_modality = data_params_local["dataset_image_modality"]
#         input_type = data_params_local["dataset_input_type"]
#         class_type = data_params_local["dataset_class_type"]
#         num_segments = data_params_local["dataset_num_segments"]
#         frames_per_segment = data_params_local["frames_per_segment"]
#
#         source = EPIC100DatasetAccess(
#                     domain="source",
#                     data_path=src_data_path,
#                     train_list=src_tr_listpath,
#                     test_list=src_te_listpath,
#                     image_modality="rgb",
#                     num_segments=num_segments,
#                     frames_per_segment=frames_per_segment,
#                     n_classes=num_verb_classes,
#                     transform=None,
#                     seed=seed,
#                     input_type=input_type,
#         )
#
#         target = EPIC100DatasetAccess(
#             domain="target",
#             data_path=tgt_data_path,
#             train_list=tgt_tr_listpath,
#             test_list=tgt_te_listpath,
#             image_modality="rgb",
#             num_segments=num_segments,
#             frames_per_segment=frames_per_segment,
#             n_classes=num_verb_classes,
#             transform=target_tf,
#             seed=seed,
#             input_type=input_type,
#         )
#
#         train_source_dataset = source.get_train()
#         train_target_dataset = source.get_train()
#         val_dataset = target.get_test()
#         test_dataset = target.get_test()
#         num_classes = 97
#         class_names = None

# if train_target_transform is None:
#     train_target_transform = train_source_transform
# if dataset_name == "Digits":
#     train_source_dataset = datasets.__dict__[source[0]](osp.join(root, source[0]), download=True,
#                                                         transform=train_source_transform)
#     train_target_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), download=True,
#                                                         transform=train_target_transform)
#     val_dataset = test_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), split='test',
#                                                               download=True, transform=val_transform)
#     class_names = datasets.MNIST.get_classes()
#     num_classes = len(class_names)
# elif dataset_name in datasets.__dict__:
#     # load datasets from common.vision.datasets
#     dataset = datasets.__dict__[dataset_name]
#
#     def concat_dataset(tasks, **kwargs):
#         return ConcatDataset([dataset(task=task, **kwargs) for task in tasks])
#
#     train_source_dataset = concat_dataset(root=root, tasks=source, download=True, transform=train_source_transform)
#     train_target_dataset = concat_dataset(root=root, tasks=target, download=True, transform=train_target_transform)
#     val_dataset = concat_dataset(root=root, tasks=target, download=True, transform=val_transform)
#     if dataset_name == 'DomainNet':
#         test_dataset = concat_dataset(root=root, tasks=target, split='test', download=True, transform=val_transform)
#     else:
#         test_dataset = val_dataset
#     class_names = train_source_dataset.datasets[0].classes
#     num_classes = len(class_names)
# else:
#     # load datasets from wilds
#     dataset = wilds.get_dataset(dataset_name, root_dir=root, download=True)
#     num_classes = dataset.n_classes
#     class_names = None
#     train_source_dataset = convert_from_wilds_dataset(dataset.get_subset('train', transform=train_source_transform))
#     train_target_dataset = convert_from_wilds_dataset(dataset.get_subset('test', transform=train_target_transform))
#     val_dataset = test_dataset = convert_from_wilds_dataset(dataset.get_subset('test', transform=val_transform))
# return train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, class_names





# def get_train_transform(resizing='default', random_horizontal_flip=True, random_color_jitter=False,
#                         resize_size=224, norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
#     """
#     resizing mode:
#         - default: resize the image to 256 and take a random resized crop of size 224;
#         - cen.crop: resize the image to 256 and take the center crop of size 224;
#         - res: resize the image to 224;
#     """
#     if resizing == 'default':
#         transform = T.Compose([
#             ResizeImage(256),
#             T.RandomResizedCrop(224)
#         ])
#     elif resizing == 'cen.crop':
#         transform = T.Compose([
#             ResizeImage(256),
#             T.CenterCrop(224)
#         ])
#     elif resizing == 'ran.crop':
#         transform = T.Compose([
#             ResizeImage(256),
#             T.RandomCrop(224)
#         ])
#     elif resizing == 'res.':
#         transform = ResizeImage(resize_size)
#     else:
#         raise NotImplementedError(resizing)
#     transforms = [transform]
#     if random_horizontal_flip:
#         transforms.append(T.RandomHorizontalFlip())
#     if random_color_jitter:
#         transforms.append(T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
#     transforms.extend([
#         T.ToTensor(),
#         T.Normalize(mean=norm_mean, std=norm_std)
#     ])
#     return T.Compose(transforms)


# def get_val_transform(resizing='default', resize_size=224,
#                       norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
#     """
#     resizing mode:
#         - default: resize the image to 256 and take the center crop of size 224;
#         – res.: resize the image to 224
#     """
#     if resizing == 'default':
#         transform = T.Compose([
#             ResizeImage(256),
#             T.CenterCrop(224),
#         ])
#     elif resizing == 'res.':
#         transform = ResizeImage(resize_size)
#     else:
#         raise NotImplementedError(resizing)
#     return T.Compose([
#         transform,
#         T.ToTensor(),
#         T.Normalize(mean=norm_mean, std=norm_std)
#     ])


# def pretrain(train_source_iter, model, optimizer, lr_scheduler, epoch, cfg, device):
#     batch_time = AverageMeter('Time', ':3.1f')
#     data_time = AverageMeter('Data', ':3.1f')
#     losses = AverageMeter('Loss', ':3.2f')
#     cls_accs = AverageMeter('Cls Acc', ':3.1f')
#
#     progress = ProgressMeter(
#         cfg.SOLVER.ITER_PER_EPOCH,
#         [batch_time, data_time, losses, cls_accs],
#         prefix="Epoch: [{}]".format(epoch))
#
#     # switch to train mode
#     model.train()
#
#     end = time.time()
#     for i in range(cfg.SOLVER.ITER_PER_EPOCH):
#         x_s, labels_s = next(train_source_iter)
#         x_s = x_s.to(device)
#         labels_s = labels_s.to(device)
#
#         # measure data loading time
#         data_time.update(time.time() - end)
#
#         # compute output
#         y_s, f_s = model(x_s)
#
#         cls_loss = F.cross_entropy(y_s, labels_s)
#         loss = cls_loss
#
#         cls_acc = accuracy(y_s, labels_s)[0]
#
#         losses.update(loss.item(), x_s.size(0))
#         cls_accs.update(cls_acc.item(), x_s.size(0))
#
#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         lr_scheduler.step()
#
#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         if i % cfg.SOLVER.PRINT_FREQ == 0:
#             progress.display(i)


def get_image_modality(image_modality):
    """Change image_modality (string) to rgb (bool), flow (bool) and audio (bool) for efficiency"""

    if image_modality.lower() == "all":
        rgb = flow = audio = True
    elif image_modality.lower() == "joint":
        rgb = flow = True
        audio = False
    elif image_modality.lower() in ["rgb", "flow", "audio"]:
        rgb = image_modality == "rgb"
        flow = image_modality == "flow"
        audio = image_modality == "audio"
    else:
        raise Exception("Invalid modality option: {}".format(image_modality))
    return rgb, flow, audio


def get_class_type(class_type):
    """Change class_type (string) to verb (bool) and noun (bool) for efficiency. Only noun is NA because we
    work on action recognition."""

    verb = True
    if class_type.lower() == "verb":
        noun = False
    elif class_type.lower() == "verb+noun":
        noun = True
    else:
        raise ValueError("Invalid class type option: {}".format(class_type))
    return verb, noun


def get_domain_adapt_config(cfg):
    """Get the configure parameters for video data for action recognition domain adaptation from the cfg files"""

    config_params = {
        "data_params": {
            "dataset_root": cfg.DATASET.ROOT,
            "dataset_src_name": cfg.DATASET.SOURCE,
            "dataset_src_trainlist": cfg.DATASET.SRC_TRAINLIST,
            "dataset_src_testlist": cfg.DATASET.SRC_TESTLIST,
            "dataset_tgt_name": cfg.DATASET.TARGET,
            "dataset_tgt_trainlist": cfg.DATASET.TGT_TRAINLIST,
            "dataset_tgt_testlist": cfg.DATASET.TGT_TESTLIST,
            "dataset_image_modality": cfg.DATASET.IMAGE_MODALITY,
            "dataset_input_type": cfg.DATASET.INPUT_TYPE,
            "dataset_class_type": cfg.DATASET.CLASS_TYPE,
            "dataset_num_segments": cfg.DATASET.NUM_SEGMENTS,
            "frames_per_segment": cfg.DATASET.FRAMES_PER_SEGMENT,
        }
    }
    return config_params


def generate_list(data_name, data_params_local, domain=None):
    """
    Args:
        data_name (string): name of dataset
        data_params_local (dict): hyperparameters from configure file
        domain (string, optional): domain type (source or target)

    Returns:
        data_path (string): image directory of dataset
        train_listpath (string): training list file directory of dataset
        test_listpath (string): test list file directory of dataset
    """

    if data_name == "EPIC":
        dataset_path = Path(data_params_local["dataset_root"]).joinpath(data_name, "EPIC_KITCHENS_2018")
    elif data_name in ["ADL", "GTEA", "KITCHEN", "EPIC100"]:
        dataset_path = Path(data_params_local["dataset_root"]).joinpath(data_name)
    else:
        raise ValueError("Wrong dataset name. Select from [EPIC, ADL, GTEA, KITCHEN, EPIC100]")

    data_path = Path.joinpath(dataset_path, "frames_rgb_flow")

    if domain is None:
        train_listpath = Path.joinpath(
            dataset_path, "annotations", "labels_train_test", data_params_local["dataset_trainlist"]
        )
        test_listpath = Path.joinpath(
            dataset_path, "annotations", "labels_train_test", data_params_local["dataset_testlist"]
        )
    else:
        train_listpath = Path.joinpath(
            dataset_path, "annotations", "labels_train_test", data_params_local["dataset_{}_trainlist".format(domain)]
        )
        test_listpath = Path.joinpath(
            dataset_path, "annotations", "labels_train_test", data_params_local["dataset_{}_testlist".format(domain)]
        )

    return data_path, train_listpath, test_listpath


class VideoDataset(Enum):
    # EPIC = "EPIC"
    # ADL = "ADL"
    # GTEA = "GTEA"
    # KITCHEN = "KITCHEN"
    EPIC100 = "EPIC100"

    @staticmethod
    def get_source_target(source: "VideoDataset", target: "VideoDataset", seed, params):
        """
        Gets data loaders for source and target datasets
        Sets channel_number as 3 for RGB, 2 for flow.
        Sets class_number as 8 for EPIC, 7 for ADL, 6 for both GTEA and KITCHEN.

        Args:
            source: (VideoDataset): source dataset name
            target: (VideoDataset): target dataset name
            seed: (int): seed value set manually.
            params: (CfgNode): hyper parameters from configure file

        Examples::
            >>> source, target, num_classes = get_source_target(source, target, seed, params)
        """
        config_params = get_domain_adapt_config(params)
        data_params = config_params["data_params"]
        data_params_local = deepcopy(data_params)
        data_src_name = data_params_local["dataset_src_name"].upper()
        src_data_path, src_tr_listpath, src_te_listpath = generate_list(data_src_name, data_params_local, domain="src")
        data_tgt_name = data_params_local["dataset_tgt_name"].upper()
        tgt_data_path, tgt_tr_listpath, tgt_te_listpath = generate_list(data_tgt_name, data_params_local, domain="tgt")
        image_modality = data_params_local["dataset_image_modality"]
        input_type = data_params_local["dataset_input_type"]
        class_type = data_params_local["dataset_class_type"]
        num_segments = data_params_local["dataset_num_segments"]
        frames_per_segment = data_params_local["frames_per_segment"]

        rgb, flow, audio = get_image_modality(image_modality)
        verb, noun = get_class_type(class_type)

        transform_names = {
            # VideoDataset.EPIC: "epic",
            # VideoDataset.GTEA: "gtea",
            # VideoDataset.ADL: "adl",
            # VideoDataset.KITCHEN: "kitchen",
            VideoDataset.EPIC100: None,
        }

        verb_class_numbers = {
            # VideoDataset.EPIC: 8,
            # VideoDataset.GTEA: 6,
            # VideoDataset.ADL: 7,
            # VideoDataset.KITCHEN: 6,
            VideoDataset.EPIC100: 97,
        }

        noun_class_numbers = {
            VideoDataset.EPIC100: 300,
        }

        factories = {
            # VideoDataset.EPIC: EPICDatasetAccess,
            # VideoDataset.GTEA: GTEADatasetAccess,
            # VideoDataset.ADL: ADLDatasetAccess,
            # VideoDataset.KITCHEN: KITCHENDatasetAccess,
            VideoDataset.EPIC100: EPIC100DatasetAccess,
        }

        rgb_source = rgb_target = flow_source = flow_target = audio_source = audio_target = None
        num_verb_classes = num_noun_classes = None

        if verb:
            num_verb_classes = min(verb_class_numbers[source], verb_class_numbers[target])
        if noun:
            num_noun_classes = min(noun_class_numbers[source], noun_class_numbers[target])

        source_tf = transform_names[source]
        target_tf = transform_names[target]

        if input_type == "image":

            if rgb:
                rgb_source = factories[source](
                    data_path=src_data_path,
                    train_list=src_tr_listpath,
                    test_list=src_te_listpath,
                    image_modality="rgb",
                    num_segments=num_segments,
                    frames_per_segment=frames_per_segment,
                    n_classes=num_verb_classes,
                    transform=source_tf,
                    seed=seed,
                )
                rgb_target = factories[target](
                    data_path=tgt_data_path,
                    train_list=tgt_tr_listpath,
                    test_list=tgt_te_listpath,
                    image_modality="rgb",
                    num_segments=num_segments,
                    frames_per_segment=frames_per_segment,
                    n_classes=num_verb_classes,
                    transform=target_tf,
                    seed=seed,
                )

            if flow:
                flow_source = factories[source](
                    data_path=src_data_path,
                    train_list=src_tr_listpath,
                    test_list=src_te_listpath,
                    image_modality="flow",
                    num_segments=num_segments,
                    frames_per_segment=frames_per_segment,
                    n_classes=num_verb_classes,
                    transform=source_tf,
                    seed=seed,
                )
                flow_target = factories[target](
                    data_path=tgt_data_path,
                    train_list=tgt_tr_listpath,
                    test_list=tgt_te_listpath,
                    image_modality="flow",
                    num_segments=num_segments,
                    frames_per_segment=frames_per_segment,
                    n_classes=num_verb_classes,
                    transform=target_tf,
                    seed=seed,
                )
            if audio:
                raise ValueError("Not support {} for input_type {}.".format(image_modality, input_type))

        elif input_type == "feature":
            # Input is feature vector, no need to use transform.
            if rgb:
                rgb_source = factories[source](
                    domain="source",
                    data_path=src_data_path,
                    train_list=src_tr_listpath,
                    test_list=src_te_listpath,
                    image_modality="rgb",
                    num_segments=num_segments,
                    frames_per_segment=frames_per_segment,
                    n_classes=num_verb_classes,
                    transform=source_tf,
                    seed=seed,
                    input_type=input_type,
                )

                rgb_target = factories[source](
                    domain="target",
                    data_path=tgt_data_path,
                    train_list=tgt_tr_listpath,
                    test_list=tgt_te_listpath,
                    image_modality="rgb",
                    num_segments=num_segments,
                    frames_per_segment=frames_per_segment,
                    n_classes=num_verb_classes,
                    transform=target_tf,
                    seed=seed,
                    input_type=input_type,
                )
            if flow:
                flow_source = factories[source](
                    domain="source",
                    data_path=src_data_path,
                    train_list=src_tr_listpath,
                    test_list=src_te_listpath,
                    image_modality="flow",
                    num_segments=num_segments,
                    frames_per_segment=frames_per_segment,
                    n_classes=num_verb_classes,
                    transform=source_tf,
                    seed=seed,
                    input_type=input_type,
                )

                flow_target = factories[source](
                    domain="target",
                    data_path=tgt_data_path,
                    train_list=tgt_tr_listpath,
                    test_list=tgt_te_listpath,
                    image_modality="flow",
                    num_segments=num_segments,
                    frames_per_segment=frames_per_segment,
                    n_classes=num_verb_classes,
                    transform=target_tf,
                    seed=seed,
                    input_type=input_type,
                )
            if audio:
                audio_source = factories[source](
                    domain="source",
                    data_path=src_data_path,
                    train_list=src_tr_listpath,
                    test_list=src_te_listpath,
                    image_modality="audio",
                    num_segments=num_segments,
                    frames_per_segment=frames_per_segment,
                    n_classes=num_verb_classes,
                    transform=source_tf,
                    seed=seed,
                    input_type=input_type,
                )

                audio_target = factories[source](
                    domain="target",
                    data_path=tgt_data_path,
                    train_list=tgt_tr_listpath,
                    test_list=tgt_te_listpath,
                    image_modality="audio",
                    num_segments=num_segments,
                    frames_per_segment=frames_per_segment,
                    n_classes=num_verb_classes,
                    transform=target_tf,
                    seed=seed,
                    input_type=input_type,
                )

        else:
            raise Exception("Invalid input type option: {}".format(input_type))

        # return (
        #     {"rgb": rgb_source, "flow": flow_source, "audio": audio_source},
        #     {"rgb": rgb_target, "flow": flow_target, "audio": audio_target},
        #     {"verb": num_verb_classes, "noun": num_noun_classes},
        # )

        train_source_dataset = rgb_source.get_train()
        train_target_dataset = rgb_target.get_train()
        val_dataset = rgb_target.get_test()
        test_dataset = rgb_target.get_test()
        num_classes = num_verb_classes
        class_names = None

        return train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, class_names


class VideoDatasetAccess(DatasetAccess):
    """
    Common API for video dataset access

    Args:
        data_path (string): image directory of dataset
        train_list (string): training list file directory of dataset
        test_list (string): test list file directory of dataset
        image_modality (string): image type (RGB or Optical Flow)
        num_segments (int): number of segments the video should be divided into to sample frames from.
        frames_per_segment (int): length of each action sample (the unit is number of frame)
        n_classes (int): number of class
        transform (string, None): types of video transforms
        seed: (int): seed value set manually.
    """

    def __init__(
            self,
            data_path,
            train_list,
            test_list,
            image_modality,
            num_segments,
            frames_per_segment,
            n_classes,
            transform,
            seed,
    ):
        super().__init__(n_classes)
        self._data_path = data_path
        self._train_list = train_list
        self._test_list = test_list
        self._image_modality = image_modality
        self._num_segments = num_segments
        self._frames_per_segment = frames_per_segment
        self._transform = video_transform.get_transform(transform, self._image_modality)
        self._seed = seed

    def get_train_valid(self, valid_ratio):
        """Get the train and validation dataset with the fixed random split. This is used for joint input like RGB and
        optical flow, which will call `get_train_valid` twice. Fixing the random seed here can keep the seeds for twice
        the same."""
        train_dataset = self.get_train()
        ntotal = len(train_dataset)
        ntrain = int((1 - valid_ratio) * ntotal)
        return torch.utils.data.random_split(
            train_dataset, [ntrain, ntotal - ntrain], generator=torch.Generator().manual_seed(self._seed)
        )


class EPIC100DatasetAccess(VideoDatasetAccess):
    """EPIC-100 video feature data loader"""

    def __init__(
            self,
            domain,
            data_path,
            train_list,
            test_list,
            image_modality,
            num_segments,
            frames_per_segment,
            n_classes,
            transform,
            seed,
            input_type,
    ):
        super(EPIC100DatasetAccess, self).__init__(
            data_path,
            train_list,
            test_list,
            image_modality,
            num_segments,
            frames_per_segment,
            n_classes,
            transform,
            seed,
        )
        self._input_type = input_type
        self._domain = domain
        self._num_train_dataload = len(pd.read_pickle(self._train_list).index)
        self._num_test_dataload = len(pd.read_pickle(self._test_list).index)

    def get_train(self):
        return VideoFrameDataset(
            root_path=Path(self._data_path, self._input_type, "{}_val.pkl".format(self._domain)),
            # Uncomment to run on train subset for EPIC UDA 2021 challenge
            # root_path=Path(self._data_path, self._input_type, "{}_train.pkl".format(self._domain)),
            annotationfile_path=self._train_list,
            num_segments=self._num_segments,  # 5
            frames_per_segment=self._frames_per_segment,  # 1
            image_modality=self._image_modality,
            imagefile_template="img_{:05d}.t7"
            if self._image_modality in ["RGB", "RGBDiff", "RGBDiff2", "RGBDiffplus"]
            else self._input_type + "{}_{:05d}.t7",
            random_shift=False,
            test_mode=False,
            input_type="feature",
            num_data_load=self._num_train_dataload,
        )

    def get_train_valid(self, valid_ratio):
        train_dataset = self.get_train()
        valid_dataset = self.get_test()
        return train_dataset, valid_dataset

    def get_test(self):
        return VideoFrameDataset(
            root_path=Path(self._data_path, self._input_type, "{}_val.pkl".format(self._domain)),
            # Uncomment to run on test subset for EPIC UDA 2021 challenge
            # root_path=Path(self._data_path, self._input_type, "{}_test.pkl".format(self._domain)),
            annotationfile_path=self._test_list,
            num_segments=self._num_segments,  # 5
            frames_per_segment=self._frames_per_segment,  # 1
            image_modality=self._image_modality,
            imagefile_template="img_{:05d}.t7"
            if self._image_modality in ["RGB", "RGBDiff", "RGBDiff2", "RGBDiffplus"]
            else self._input_type + "{}_{:05d}.t7",
            random_shift=False,
            test_mode=True,
            input_type="feature",
            num_data_load=self._num_test_dataload,
        )


class LinearNet(nn.Module):
    def __init__(self, in_feature=1024, hidden_size=256, out_feature=1024):
        super(LinearNet, self).__init__()
        self._out_feature = out_feature
        self.net = nn.Sequential(
            nn.Linear(in_feature, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, out_feature),
        )

    def forward(self, x):
        x = self.net(x)
        return x

    @property
    def out_features(self):
        """The dimension of output features"""
        return self._out_feature
