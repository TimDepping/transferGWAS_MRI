from os.path import join
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

from tqdm import tqdm

from sklearn.decomposition import PCA

import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from torchvision import transforms, models

TRANSFORMATION_MEAN = [0.485, 0.456, 0.406]
TRANSFORMATION_STD = [0.229, 0.224, 0.225]


'''
This script to go from trained model to low-dimensional condensed features.
'''


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'img_csv',
        type=str,
        help='input image .csv-file. Needs to have columns "IID" and "path". '
             'Can contain additional column "instance" for multiple images per '
             'IID (e.g. left and right eye).',
    )
    parser.add_argument(
        'out_dir',
        type=str,
        default='results',
        help='Directory to save the results',
    )
    parser.add_argument(
        '--base_img_dir',
        type=str,
        help='Path to image directory.'
             'If specified, all paths in `img_csv` will be preprended by this')
    parser.add_argument('--save_str', type=str,
                        help='Optional name of file to save to. Needs to contain two `%s` substrings (for layer and explained variance).')
    parser.add_argument('--img_size', type=int,
                        default=448, help='Input image size')
    parser.add_argument(
        '--tfms',
        type=str,
        default='basic',
        help='What kind of image transformations to use.',
    )
    parser.add_argument('--dev', default='cuda:0',
                        type=str, help='cuda device to use')
    parser.add_argument('--n_pcs', type=int, default=50,
                        help='How many PCs to export')
    parser.add_argument('--num_threads', type=int, default=1,
                        help='How many threads to use')
    parser.add_argument(
        '--model',
        type=str,
        default='resnet50',
        choices=['resnet18', 'resnet34', 'resnet50'],
        help='Model architecture.',
    )
    parser.add_argument(
        '--pretraining',
        type=str,
        default='imagenet',
        help='What weights to load into the model. If `imagenet`, load the default'
             ' pytorch weights; otherwise, specify path to `.pt` with state dict',
    )
    parser.add_argument(
        '--layer',
        type=str,
        default=['L4'],
        nargs='+',
        help='At what layer to extract the embeddings from the network. '
             'Can be either `L1`, `L2`, `L3`, `L4` for the default locations in layer 1-4, '
             'or can name a specific module in the architecture such as `layer4.2.conv3`. '
             'Multiple layers can be specified and will be exported to corresponding files.',
    )
    parser.add_argument(
        "--use_mc",
        dest="use_mc",
        action="store_true",
        default=False,
        help="Change Dataset class to tensor input .pt (50 Channels) and adapt model to comply with new input shape. By default 3 Channel RGB images are handled.",
    )
    args = parser.parse_args()

    dsets = load_data_from_csv(
        args.img_csv,
        tfms=args.tfms,
        img_size=args.img_size,
        base_img_dir=args.base_img_dir,
        use_mc=args.use_mc,
    )
    model = load_model(
        args.model,
        args.pretraining,
        args.dev,
        args.use_mc,
    )
    layer_funcs = load_layers(
        args.model,
        args.layer,
    )

    embeddings = []
    for dset in dsets:
        embeddings_i = compute_embeddings(
            model=model,
            layer_funcs=layer_funcs,
            dset=dset,
            dev=args.dev,
            num_threads=args.num_threads,
        )
        embeddings.append(embeddings_i)
    embeddings = join_embeddings(embeddings, dsets)

    os.makedirs(args.out_dir, exist_ok=True)
    for layer_name, layer_embedding in zip(args.layer, embeddings):
        pca = PCA(n_components=args.n_pcs)
        pca_embedding = pca.fit_transform(layer_embedding)
        explained_var = pca.explained_variance_ratio_

        pret_part = args.pretraining.split('/')[-1].split('.')[0]
        if not args.save_str:
            save_str = join(
                args.out_dir,
                f'{args.model}_{pret_part}_{layer_name}.txt',
            )
            save_str_evr = join(
                args.out_dir,
                f'{args.model}_{pret_part}_{layer_name}_explained_variance.txt',
            )
        else:
            save_str = join(
                args.out_dir,
                args.save_str % (layer_name, ''),
            )
            save_str_evr = join(
                args.out_dir,
                args.save_str % (layer_name, '_explained_variance'),
            )

        to_file(
            pca_embedding,
            explained_var,
            dsets[0].ids,
            save_str=save_str,
            save_str_evr=save_str_evr,
        )


def load_data_from_csv(fn, tfms='basic', img_size=448, base_img_dir='', use_mc=False):
    """Load csv into ImageData instance(s)"""
    dsets = []

    # Create dataframe from csv file.
    df = pd.read_csv(fn)
    if base_img_dir:
        # Construct path with base_img_dir for each file if base_img_dir is provided.
        df.path = [join(base_img_dir, p) for p in df.path]

    if use_mc:
        dsets = [TensorData(df)]
    else:
        instance_grouping = "instance"
        if instance_grouping in df.columns:
            # Column instance_grouping exists in df (in the csv file).
            iid = np.unique(df.IID)
            for _, sub_df in df.groupby(instance_grouping):
                iid = np.intersect1d(iid, sub_df.IID)
            # At this point, we only have unique iids of entries which are available in all groups.
            for _, sub_df in df.groupby(instance_grouping):
                sub_df.index = sub_df.IID
                dset = ImageData(sub_df.loc[iid], tfms=tfms, img_size=img_size)
                dsets.append(dset)
        else:
            dsets = [ImageData(df, tfms=tfms, img_size=img_size)]

    return dsets


def get_tfms_basic(size=224, min_val=0, max_val=1):
    '''Resize and Min-Max Scaling'''

    class MinMaxScaling(object):

        def percentile_scaling(self, tensor, min_val=0, max_val=1, lower_percentile=0, upper_percentile=98):
            array = tensor.numpy()
            lower_bound = np.percentile(array, lower_percentile)
            upper_bound = np.percentile(array, upper_percentile)
            array = (array - lower_bound) / (upper_bound - lower_bound)
            array = array * (max_val - min_val) + min_val
            # rearrange shape off array to fit ToTensor (224,224,3)
            array = np.transpose(array, (1, 2, 0))
            tensor = transforms.ToTensor()(array)
            return tensor

        def __call__(self, tensor):
            return self.percentile_scaling(tensor, min_val, max_val)

    tfms = transforms.Compose([
        transforms.Resize(size=size),
        transforms.ToTensor(),
        MinMaxScaling(),
    ])
    return tfms


def get_tfms_augmented(size=224):
    '''get test-time augmentation tfms'''
    resize = transforms.Resize(size=size)

    tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=TRANSFORMATION_MEAN, std=TRANSFORMATION_STD),
    ])

    def tfm(x):
        x = resize(x)
        flipped = x.transpose(Image.FLIP_LEFT_RIGHT)

        x90 = x.transpose(Image.ROTATE_90)
        flipped90 = flipped.transpose(Image.ROTATE_90)
        x180 = x.transpose(Image.ROTATE_180)
        flipped180 = flipped.transpose(Image.ROTATE_180)
        x270 = x.transpose(Image.ROTATE_270)
        flipped270 = flipped.transpose(Image.ROTATE_270)

        imgs = [x, flipped, x90, flipped90, x180, flipped180, x270, flipped270]
        imgs = torch.stack([tfms(img) for img in imgs])
        return imgs

    return tfm


def load_model(model='resnet50', pretraining='imagenet', dev='cuda:0', use_mc=False):
    """Prepare model and load pretrained weights."""
    # Default model is a ResNet50.
    m_func = models.resnet50
    if model == 'resnet18':
        m_func = models.resnet18
    elif model == 'resnet34':
        m_func = models.resnet34

    model = m_func(pretrained=True)
    if pretraining != 'imagenet':
        if use_mc:
            model.conv1 = nn.Conv2d(
                50, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        model.load_state_dict(torch.load(pretraining, map_location="cpu"))
    return prep_model(model, dev)


def prep_model(model, dev):
    """Set to eval, move to device and remove `inplace` operations."""
    model = model.eval().to(dev)
    for mod in model.modules():
        if hasattr(mod, 'inplace'):
            # TODO: Why do we not do it inplace?
            mod.inplace = False
    return model


def load_layers(model, layers):
    """Get functions to return layers, for hooks."""

    LAYERS_RES18 = [
        lambda m: m.layer1[-1].conv2,
        lambda m: m.layer2[-1].conv2,
        lambda m: m.layer3[-1].conv2,
        lambda m: m.layer4[-1].conv2,
    ]
    LAYERS_RES50 = [
        lambda m: m.layer1[-1].conv3,
        lambda m: m.layer2[-1].conv3,
        lambda m: m.layer3[-1].conv3,
        lambda m: m.layer4[-1].conv3,
    ]

    # Default case is 'resnet50'.
    lf = LAYERS_RES50
    if model in ['resnet18', 'resnet34']:
        lf = LAYERS_RES18
    lfs = []
    for layer in layers:
        def layer_func(m, l_str=layer): return dict(m.named_modules())[l_str]
        if layer in ['L1', 'L2', 'L3', 'L4']:
            layer_func = lf[int(layer[1]) - 1]
        lfs.append(layer_func)
    return lfs


def compute_embeddings(
        model,
        layer_funcs,
        dset,
        dev='cuda:0',
        num_threads=1,
):
    '''compute all embeddings in dset at layer_funcs in model

    # Parameters:
    model (nn.Module): pretrained pytorch model
    layer_funcs (list of functions): each element is a function that takes model
                as argument and returns the corresponding submodule (for registering hooks)
    dset (ImageData): dataset for which to extract the embeddings
    dev (torch.device or str): where to perform the computations
    num_threads (int): how many torch CPU-threads
    '''
    torch.set_num_threads(num_threads)
    output = [[] for _ in layer_funcs]
    hooks = []
    for idx, layer_func in enumerate(layer_funcs):
        hook = layer_func(model).register_forward_hook(
            lambda m, i, o, idx=idx: output[idx].append(o.detach()))
        hooks.append(hook)

    tmp_img = dset[0][0]
    if len(tmp_img.shape) == 3:
        tmp_img.unsqueeze_(0)
    n_tfm = tmp_img.shape[0]
    _ = model(tmp_img.to(dev))
    shapes = [out[0].shape[1] * n_tfm for out in output]
    # TODO: Why clear output array items?
    for i in range(len(output)):
        output[i] = []

    embeddings = [np.empty((len(dset), shape)) for shape in shapes]
    for sample_idx, (img, iid) in tqdm(enumerate(dset), total=len(dset)):
        with torch.no_grad():
            if len(img.shape) == 3:
                img = img.view(1, *img.shape)
            _ = model(img.to(dev))
        for layer_idx, out in enumerate(output):
            # conv layer
            if len(out[0].shape) > 2:
                embedding = out[0].mean([-1, -2]).flatten()
            # non-conv layer
            else:
                embedding = out[0].flatten()
            embeddings[layer_idx][sample_idx, :] = embedding.cpu().numpy()
        for i in range(len(output)):
            output[i] = []

    # clean up model afterwards again
    for hook in hooks:
        hook.remove()
    return embeddings


def join_embeddings(embeddings, dsets):
    '''take list of list of embeddings and return joined embeddings

    # Parameters:
    embeddings (list of list of np.array): nested list of embeddings with structure:
                [[dset1-layer1, dset1-layer2, ...], [dset2-layer1, dset2-layer2, ...], ...]
    dsets (list of ImageData): corresponding to embeddings, only used for checking IIDs
    '''
    all_ids = np.array([d.ids for d in dsets])
    # make sure all dsets have the same ids and ordering
    assert np.all(np.repeat(np.expand_dims(
        all_ids[0], 0), len(dsets), axis=0) == all_ids)

    out = []
    for i in range(len(embeddings[0])):
        out.append(np.concatenate([emb[i] for emb in embeddings], axis=1))
    return out


def to_file(
        embeddings,
        explained_var,
        iid,
        save_str,
        save_str_evr,
        pheno_name='PC_%d',
):
    '''save embeddings and explained variance to files'''
    evr = pd.DataFrame(
        [(pheno_name % d, e) for d, e in enumerate(explained_var)],
        columns=['PC', 'explained_variance_ratio'],
    )
    evr.to_csv(save_str_evr, index=False, header=True)

    data = pd.DataFrame(
        dict(
            [('FID', iid), ('IID', iid)] +
            [(pheno_name % d, embeddings[:, d])
             for d in range(embeddings.shape[1])]
        )
    )
    data.to_csv(save_str, sep=' ', index=False, header=True)


class ImageData(Dataset):
    def __init__(self, df, tfms='basic', img_size=448):
        self.ids = df.IID.values
        self.path = df.path.values
        self.img_size = img_size
        self.tfms = get_tfms_basic(
            img_size) if tfms == 'basic' else get_tfms_augmented(img_size)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = self._load_item(idx)
        id = self.ids[idx]
        return img, id

    def _load_item(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        p = self.path[idx]
        img = Image.open(p).convert('RGB')
        if self.tfms:
            img = self.tfms(img)
        return img


class TensorData(Dataset):
    '''! Not possible to change img size yet. No transformations, except minMaxScaling + ToTensor'''

    def __init__(self, df):
        self.ids = df.IID.values
        self.path = df.path.values
        self.img_size = 224

    def percentile_scaling_array(
        self, array, lower_percentile=0, upper_percentile=98, min_val=0, max_val=1
    ):
        lower_bound = np.percentile(array, lower_percentile)
        upper_bound = np.percentile(array, upper_percentile)
        array = (array - lower_bound) / (upper_bound - lower_bound)
        array = array * (max_val - min_val) + min_val
        tensor = transforms.ToTensor()(array).float()
        return tensor

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = self._load_item(idx)
        id = self.ids[idx]
        return img, id

    def _load_item(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        mcTensor = torch.load(self.path[idx])

        # ToTensor requires a numpy.ndarray (H x W x C)
        # 1) transform mcTensor to npArrayList (50x 1x1x224x224)
        npArrays = mcTensor.numpy()
        npArrayList = np.split(npArrays, 50, axis=0)
        # 2) remove dimensions of size 1 infront of the array 1,1,224,244 -> 224,244
        npArrayList = np.squeeze(npArrayList)
        # 3) add dimentions of size 1 at the end of the array 224,244 -> 224,244,1
        npArrayList = [array[:, :, np.newaxis] for array in npArrayList]
        # 4) scale tensor list + toTensor
        scaledTensorList = [
            self.percentile_scaling_array(array, 0, 98, 0, 1) for array in npArrayList
        ]
        # 5) Get rid of the first dimension: [1,224,224] -> [224,224]
        squeezed_tensors = [tensor.squeeze(0) for tensor in scaledTensorList]
        # 6) Stack all the transformed images back together to a create a 50 channel tensor
        stacked_tensor = torch.stack(squeezed_tensors, dim=0)

        return stacked_tensor


if __name__ == "__main__":
    main()
