import imgaug as ia
import imgaug.augmenters as iaa

available_transformers = ['DropoutAndAffine']


class TransformerBase:
    def __init__(self, transform_labels=None, **kwargs):
        self.augmenter = self._build_augmenter(**kwargs)
        self.transform_labels = transform_labels

    def _build_augmenter(**kwargs):
        raise NotImplementedError()

    def __call__(self, data_dict):
        if not self.transform_labels or data_dict['label/name'] in self.transform_labels:
            data_dict['image'] = self.augmenter.augment_images(data_dict['image'])

        return data_dict


class DropoutAndAffine(TransformerBase):
    def _build_augmenter(self, **kwargs):
        aug = iaa.SomeOf((0, None), [
            iaa.Affine(rotate=45),
            iaa.CoarseDropout((0.01, 0.05), size_percent=(0.05, 0.33)),
            iaa.Fliplr(1),
            iaa.Flipud(1),
            iaa.CropAndPad(percent=(-0.25, 0.25))
        ])
        return aug
