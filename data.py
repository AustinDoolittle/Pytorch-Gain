import cv2
import os
import torch
import random

class RawDataset:
    def __init__(self, root_dir, ds_split=0.8, include_exts=['.jpg', '.png', '.jpeg'], transformer=None, out_dims=(224, 224)):
        self._ds_split = ds_split
        self.root_dir = root_dir
        self.include_exts = include_exts
        self.out_dims = out_dims
        self.transformer = transformer

        self.datasets = self._load_datasets()

    def _load_datasets(self):
        # iterate over the immediate child directories and store paths to images
        image_dict = {}
        for p in os.listdir(self.root_dir):
            full_dir = os.path.join(self.root_dir, p)
            if not os.path.isdir(full_dir):
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
                'out_dims': self.out_dims,
            }
            if k == 'train':
                ds_args['transformer'] = self.transformer

            ds = ImageDataset(filesets['train'], self.labels, **ds_args)
            ret_dict[k] = torch.utils.data.DataLoader(ds, shuffle=True, num_workers=2)

        return ret_dict


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, fileset, labels, transformer=None, out_dims=(256,256)):
        self.out_dims = out_dims
        self.transformer = transformer
        self._fileset = fileset
        self.labels = labels

    def __len__(self):
        return len(self._fileset)

    def __getitem__(self, idx):
        # decode and resize
        decoded_img = cv2.imread(self._fileset[idx][0])
        decoded_img = cv2.resize(decoded_img, self.out_dims)

        # pytorch is [C, H, W]
        decoded_img = decoded_img.transpose((2, 0, 1))
        decoded_img = decoded_img.astype(np.float32)
        decoded_img = decoded_img / 255.0

        label_index = self.labels.index(self._fileset[idx][1])

        ret_dict = {
            'image': decoded_img,
            'label/idx': label_index,
            'label/name': self._fileset[idx][1]
        }

        # transform 'em if we got 'em
        if self.transformer:
            ret_dict = self.transformer(ret_dict)

        return ret_dict
