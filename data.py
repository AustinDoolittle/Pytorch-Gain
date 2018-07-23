import cv2
import numpy as np
import os
import torch
import random

def write_image(path, image):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    cv2.imwrite(path, image)

def load_image(image_path, resize_dims, expected_channels):
    # decode and resize
    image = cv2.imread(image_path, -1)
    if image is None:
        raise IOError('There was an error reading image %s'%image_path)

    image = cv2.resize(image, resize_dims)
    return cv_to_mx(image, expected_channels)

def scale_to_range(arr, min_range, max_range):
    min_val = arr.min()
    max_val = arr.max()
    return ((max_range-min_range) * (arr - min_val)) / (max_val - min_val + 1e-5) + min_range

def mx_to_cv(image):
    image = scale_to_range(image, 0, 255)
    image = image.transpose((1, 2, 0)).astype(np.uint8)
    return image


def cv_to_mx(image, expected_channels):
    if len(image.shape) == 2:
        image = np.expand_dims(image , axis=2)
    # convert to expected channel count
    if image.shape[2] == 1 and expected_channels == 3:
        image = cv2.cvtColor(image , cv2.COLOR_GRAY2RGB)

    elif image.shape[2] == 3 and expected_channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = np.expand_dims(image, axis=2)

    # pytorch is [C, H, W]
    image = image.transpose((2, 0, 1))
    image = image.astype(np.float32)
    image = scale_to_range(image, 0.0, 1.0)
    return image


class RawDataset:
    def __init__(self, root_dir, ds_split=0.8, include_exts=['.jpg', '.png', '.jpeg'], transformer=None, output_dims=(224, 224), output_channels=3, num_workers=1, batch_size_dict=None):
        self.name = os.path.basename(os.path.normpath(root_dir))
        self._ds_split = ds_split
        self.root_dir = root_dir
        self.output_channels = output_channels
        self.include_exts = include_exts
        self.output_dims = output_dims
        self.num_workers = num_workers
        if not batch_size_dict:
            batch_size_dict = {
                'train': 1,
                'test': 1
            }
        self.batch_size_dict = batch_size_dict

        # TODO implement transformers
        self.transformer = transformer

        self.datasets = self._load_datasets()

    def _load_datasets(self):
        # iterate over the immediate child directories and store paths to images
        image_dict = {}
        for p in os.listdir(self.root_dir):
            full_dir = os.path.join(self.root_dir, p)
            if not os.path.isdir(full_dir):
                continue

            p_split = p.rsplit('___', 1)
            p = p_split[0]
            if len(p_split) > 1:
                p_tag = p_split[1]
                if p_tag == 'ignore':
                    continue

            for f in os.listdir(full_dir):
                if os.path.splitext(f)[1] in self.include_exts:
                    if not p in image_dict:
                        image_dict[p] = []

                    image_dict[p].append(os.path.join(full_dir, f))
        self.labels = list(image_dict.keys())

        # split the train and test datasets equally among labels
        filesets = {'test': [], 'train': []}
        for k, filenames in image_dict.items():
            random.shuffle(filenames)
            split_index = int(len(filenames) * self._ds_split)
            filesets['train'] += [(f, k) for f in filenames[:split_index]]
            filesets['test'] += [(f, k) for f in filenames[split_index:]]

        ret_dict = {}

        # create image datasets for train and test
        for k in filesets:
            ds_args = {
                'output_dims': self.output_dims,
                'output_channels': self.output_channels
            }
            if k == 'train':
                ds_args['transformer'] = self.transformer

            ds = ImageDataset(filesets[k], self.labels, **ds_args)
            ret_dict[k] = torch.utils.data.DataLoader(ds, shuffle=True, num_workers=self.num_workers, batch_size=self.batch_size_dict[k])

        return ret_dict


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, fileset, labels, transformer=None, output_dims=(256,256), output_channels=3):
        self.output_dims = output_dims
        self.output_channels=output_channels
        self.transformer = transformer
        self._fileset = fileset
        self.labels = labels

    def __len__(self):
        return len(self._fileset)

    def __getitem__(self, idx):
        decoded_img = None
        label = None
        while decoded_img is None:
            try:
                decoded_img = load_image(self._fileset[idx][0], self.output_dims, self.output_channels)
                label = self._fileset[idx][1]
            except IOError as e:
                print('WARNING error reading file at index %i, returning random data instance: %s'%(idx, str(e)))
                idx = random.randint(0, len(self._fileset) - 1)

        label_index = self.labels.index(label)

        # NOTE I'd much prefer this be done in actual model
        # using pytorch, but they don't have great onehot support
        one_hot = np.zeros((len(self.labels),), dtype=np.float32)
        one_hot[label_index] = 1

        ret_dict = {
            'image': decoded_img,
            'label/idx': label_index,
            'label/onehot': one_hot,
            'label/name': self._fileset[idx][1]
        }

        # transform 'em if we got 'em
        if self.transformer:
            ret_dict = self.transformer(ret_dict)

        return ret_dict

