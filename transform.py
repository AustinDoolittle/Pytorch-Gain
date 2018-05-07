import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np

available_transformers = ['Dropout', 'Affine', 'DropoutAndAffine', 'Translate']


class TransformerBase:
    def __init__(self, transform_labels=None, **kwargs):
        self.augmenter = self._build_augmenter(**kwargs)
        self.transform_labels = transform_labels

    def _build_augmenter(**kwargs):
        raise NotImplementedError()

    def __call__(self, data_dict):
        if not self.transform_labels or data_dict['label/name'] in self.transform_labels:
            image = data_dict['image'] * 255.0
            image = image.astype(np.uint8)
            image = self.augmenter.augment_images(image)
            data_dict['image'] = image.astype(np.float32) / 255.0

        return data_dict


class DropoutAndAffine(TransformerBase):
    def _build_augmenter(self, **kwargs):
        affine_xform = Affine()
        dropout_xform = Dropout()
        aug = iaa.Sequential([
           affine_xform.augmenter,
           dropout_xform.augmenter],
           random_order=True)
        return aug


class Translate(TransformerBase):
    def _build_augmenter(self, **kwargs):
        aug = iaa.SomeOf((0, None), [
            iaa.Affine(translate_percent=(0, 0.1)),
            iaa.CropAndPad(percent=(0, -0.1), keep_size=True, sample_independently=True)],
            random_order=True)
        return aug


class Affine(TransformerBase):
    def _build_augmenter(self, **kwargs):
        aug = iaa.SomeOf((0, None), [
            iaa.Affine(translate_percent=(0, 0.1)),
            iaa.Affine(rotate=(0, 45)),
            iaa.Fliplr(1),
            iaa.Flipud(1)],
            random_order=True)
        return aug


class Dropout(TransformerBase):
    def _build_augmenter(self, **kwargs):
        aug = iaa.Sometimes(0.5,
            iaa.CoarseDropout((0.01, 0.05), size_percent=(0.05, 0.25)))
        return aug
