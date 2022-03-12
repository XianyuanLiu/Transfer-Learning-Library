"""
@author: Junguang Jiang, Baixu Chen
@contact: JiangJunguang1123@outlook.com, cbx_99_hasta@outlook.com
"""
import math
import sys
import os.path as osp
import comet_ml
import time
from collections import OrderedDict
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
# import common.vision.datasets as datasets
# import common.vision.models as models
# from common.vision.transforms import ResizeImage
from common.utils.metric import accuracy, ConfusionMatrix
from common.utils.meter import AverageMeter, ProgressMeter


def to_numpy(x):
    return x.detach().cpu().numpy()


def nested_children(m: torch.nn.Module):
    children = dict(m.named_children())
    output = {}
    if children == {}:
        # if module has no children; m is last child! :O
        return m
    else:
        # look for children from children... to the last child!
        for name, child in children.items():
            try:
                output[name] = nested_children(child)
            except TypeError:
                output[name] = nested_children(child)
    return output


def update_gradient_map(model, gradmap):
    for name, layer in model.named_modules():
        if "activ" in name:
            continue

        if not hasattr(layer, "weight"):
            continue

        wname = "%s/%s.%s" % ("gradient", name, "weight")
        bname = "%s/%s.%s" % ("gradient", name, "bias")

        gradmap.setdefault(wname, 0)
        gradmap.setdefault(bname, 0)

        gradmap[wname] += layer.weight.grad
        gradmap[bname] += layer.bias.grad

    return gradmap


def log_gradients(gradmap, step, experiment):
    for k, v in gradmap.items():
        experiment.log_histogram_3d(to_numpy(v), name=k, step=step)


def log_weights(model, step, experiment):
    for name, layer in zip(model._modules, model.children()):
        if "activ" in name:
            continue

        if not hasattr(layer, "weight"):
            continue

        wname = "%s.%s" % (name, "weight")
        bname = "%s.%s" % (name, "bias")

        experiment.log_histogram_3d(to_numpy(layer.weight), name=wname, step=step)
        experiment.log_histogram_3d(to_numpy(layer.bias), name=bname, step=step)


def validate(val_loader, model, cfg, device, class_names, experiment, epoch=None, name=None) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    # cm = comet_ml.ConfusionMatrix()
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if cfg.SOLVER.PER_CLASS_EVAL:
        confmat = ConfusionMatrix(len(class_names))
    else:
        confmat = None

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target[0]
            target = target.to(device)

            # compute output
            output = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, = accuracy(output, target, topk=(1,))
            if confmat:
                confmat.update(target, output.argmax(1))

            if cfg.COMET.ENABLE:
                experiment.log_metric("train_loss", loss.item(), epoch=epoch)
                experiment.log_metric('val_loss', loss.item(), epoch=epoch)
                experiment.log_metric('val_acc', acc1.item(), epoch=epoch)

            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg.SOLVER.PRINT_FREQ == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        if confmat:
            # print(confmat.format(class_names))
            if cfg.COMET.ENABLE and name == "valid":
                experiment.log_confusion_matrix(
                    matrix=confmat.mat.tolist(),
                    title="Confusion Matrix, Epoch {}".format(epoch + 1),
                    file_name="confusion_matrix_{}.json".format(epoch + 1),
                    overwrite=True
                )

    return top1.avg


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
        class_names = [i for i in range(num_classes)]

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


class Pooling(nn.Module):
    def __init__(self):
        super(Pooling, self).__init__()
        self.net = nn.Sequential(
            OrderedDict(
                [("avgpool", nn.AdaptiveAvgPool1d(1)), ]
            )
        )

    def forward(self, x):
        x = self.net(x.transpose(1, 2)).squeeze(2)
        return x


class LinearNet(nn.Module):
    def __init__(self, in_feature=1024, hidden_size=256, out_feature=1024):
        super(LinearNet, self).__init__()
        self._out_feature = out_feature
        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("linear0", nn.Linear(in_feature, hidden_size)),
                    # nn.BatchNorm1d(hidden_size),
                    ("activ0", nn.ReLU()),
                    ("dp0", nn.Dropout(0.5)),
                    ("linear1", nn.Linear(hidden_size, hidden_size)),
                    # nn.BatchNorm1d(hidden_size),
                    ("activ1", nn.ReLU()),
                    ("dp1", nn.Dropout(0.5)),
                    ("output", nn.Linear(hidden_size, out_feature)),
                ]
            )
        )

    def forward(self, x):
        x = self.net(x)
        return x

    @property
    def out_features(self):
        """The dimension of output features"""
        return self._out_feature


class SelfAttention(nn.Module):
    """A vanilla multi-head attention layer with a projection at the end. Can be set to causal or not causal."""

    def __init__(
        self, emb_dim, num_heads, att_dropout, final_dropout, causal=False, max_seq_len=10000, use_performer_att=False
    ):
        super().__init__()
        assert emb_dim % num_heads == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(emb_dim, emb_dim)
        self.query = nn.Linear(emb_dim, emb_dim)
        self.value = nn.Linear(emb_dim, emb_dim)
        # regularization
        self.att_dropout = nn.Dropout(att_dropout)
        self.final_dropout = nn.Dropout(final_dropout)
        # output projection
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.causal = causal
        if causal:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "mask", torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
            )

        self.num_heads = num_heads

        self.use_performer_att = use_performer_att
        # if self.use_performer_att:
        #     self.performer_att = FastAttention(dim_heads=emb_dim//num_heads, nb_features=emb_dim, causal=False)

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)

        if not self.use_performer_att:
            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.causal:
                att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.att_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        else:
            y = self.performer_att(q, k, v)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.final_dropout(self.proj(y))
        return y


class TransformerBlock(nn.Module):
    """
    Standard transformer block consisting of multi-head attention and two-layer MLP.
    """

    def __init__(
        self, emb_dim, num_heads, att_dropout, att_resid_dropout, final_dropout, max_seq_len, ff_dim, causal=False,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.attn = SelfAttention(emb_dim, num_heads, att_dropout, att_resid_dropout, causal, max_seq_len)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, emb_dim), nn.Dropout(final_dropout),
        )

    def forward(self, x):
        # BATCH, TIME, CHANNELS = x.size()
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerSENet(nn.Module):
    """Regular simple network for video input.
    Args:
        in_feature (int, optional): the dimension of the final feature vector.
        hidden_size (int, optional): the number of channel for Linear and BN layers.
        out_feature (int, optional): the dimension of output.
    """

    def __init__(self, in_feature=1024, hidden_size=512, out_feature=1024):
        super(TransformerSENet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 4
        self.num_heads = 8

        self.transformer = nn.ModuleList(
            [
                TransformerBlock(
                    emb_dim=in_feature,
                    num_heads=self.num_heads,
                    att_dropout=0.1,
                    att_resid_dropout=0.1,
                    final_dropout=0.1,
                    max_seq_len=9,
                    ff_dim=self.hidden_size,
                    causal=False,
                )
                for _ in range(self.num_layers)
            ]
        )

        # self.fc1 = nn.Linear(input_size, n_channel)
        # self.relu1 = nn.ReLU()
        # self.dp1 = nn.Dropout(dropout_keep_prob)
        # self.fc2 = nn.Linear(n_channel, output_size)
        # self.fc3 = nn.Linear(input_size, output_size)
        # self.selayer = SELayerFeat(channel=16, reduction=4)

    def forward(self, x):
        for layer in self.transformer:
            x = layer(x)
        # x = self.fc2(self.dp1(self.relu1(self.fc1(x))))
        # x = self.fc3(x)
        # x = self.selayer(x)
        return x

    def out_features(self):
        """The dimension of output features"""
        return self._out_feature
