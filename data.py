import cv2
import numpy as np
import os
import torch
import random

def load_image(image_path, resize_dims, expected_channels):
    # decode and resize
    decoded_img = cv2.imread(image_path, -1)
    if decoded_img is None:
        raise IOError('There was an error reading image %s'%image_path)

    decoded_img = cv2.resize(decoded_img, resize_dims)
    if len(decoded_img.shape) == 2:
        decoded_img = np.expand_dims(decoded_img, axis=2)
    # convert to expected channel count
    if decoded_img.shape[2] == 1 and expected_channels == 3:
        decoded_img = cv2.cvtColor(decoded_img, cv2.COLOR_GRAY2RGB)

    elif decoded_img.shape[2] == 3 and expected_channels == 1:
        decoded_img = cv2.cvtColor(decoded_img, cv2.COLOR_RGB2GRAY)
        decoded_img = np.expand_dims(decoded_img, axis=2)

    # pytorch is [C, H, W]
    decoded_img = decoded_img.transpose((2, 0, 1))
    decoded_img = decoded_img.astype(np.float32)
    decoded_img = decoded_img / 255.0
    return decoded_img


class RawDataset:
    def __init__(self, root_dir, ds_split=0.8, include_exts=['.jpg', '.png', '.jpeg'], transformer=None, output_dims=(224, 224), output_channels=3, num_workers=1):
        self.name = os.path.split(root_dir)[-1]
        self._ds_split = ds_split
        self.root_dir = root_dir
        self.output_channels = output_channels
        self.include_exts = include_exts
        self.output_dims = output_dims
        self.num_workers = num_workers

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

                    image_dict[p].append(os.path.join(self.root_dir, p, f))
        self.labels = image_dict.keys()

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

            ds = ImageDataset(filesets['train'], self.labels, **ds_args)
            ret_dict[k] = torch.utils.data.DataLoader(ds, shuffle=True, num_workers=self.num_workers)

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

