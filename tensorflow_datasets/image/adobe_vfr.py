"""TODO(adobe_vfr): Add a description here."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os

import numpy as np
import tensorflow as tf
from PIL import Image
import tensorflow_datasets as tfds
# TODO(adobe_vfr): BibTeX citation
from tensorflow_datasets.core import api_utils

_CITATION = """
"""

# TODO(adobe_vfr):
_DESCRIPTION = """
"""

# Shared constants
_ADOBEVFR_IMAGE_SIZE = 105
_ADOBEVFR_IMAGE_SHAPE = (_ADOBEVFR_IMAGE_SIZE, _ADOBEVFR_IMAGE_SIZE, 1)

_LABELS_FILENAME = 'fontlist.txt'

class bcf: #TODO Switch to tf.io.gfile
    def __init__(self, filename):
        self._filename = filename
        self._file = open(filename, 'rb')
        print(type(self._file))
        size = np.uint32(np.frombuffer(self._file.read(8), dtype=np.uint32)[0])
        file_sizes = np.frombuffer(self._file.read(8 * size),
                                   dtype=np.uint64)
        # print("File sizes", file_sizes)
        self._offsets = np.append(np.uint64(0),
                                  np.add.accumulate(file_sizes))
        # print('BCF initialised')

    def get(self, i):
        # print("get image", i)
        self._file.seek(np.int32(len(self._offsets) * 8 + self._offsets[i]))
        # print(self._file)
        image = self._file.read(self._offsets[i + 1] - self._offsets[i])
        # print(image, type(image))
        return image
        # return self._file.read(self._offsets[i + 1] - self._offsets[i])

    def size(self):
        return len(self._offsets) - 1


def read_bcf(path, mode):
    base = {'train': 'train', 'test': 'vfr_large'}
    bcf_path = '%s/%s.bcf' % (path, base[mode])
    label_path = '%s/%s.label' % (path, base[mode])
    labels = read_label(label_path)
    images = bcf(bcf_path)
    print("BCF labels and images successfully loaded.", images, labels)
    return images, labels


def read_label(path):
    with open(path, 'rb') as f:
        labels = np.frombuffer(f.read(), dtype=np.uint32)
    return labels


class AdobeVFRConfig(tfds.core.BuilderConfig):
    """BuilderConfig for AdobeVFR."""

    @api_utils.disallow_positional_args
    def __init__(self, mode, **kwargs):
        """BuilderConfig for AdobeVFR Dataset.

        Args:
          mode: raw or bcf
          **kwargs: keyword arguments forwarded to super.
        """
        super(AdobeVFRConfig, self).__init__(**kwargs)
        self.mode = mode


class AdobeVFR(tfds.core.GeneratorBasedBuilder):
    """TODO(adobe_vfr): Short description of my dataset."""

    VERSION = tfds.core.Version('0.1.0')

    print("Loading VFR")

    BUILDER_CONFIGS = [
        AdobeVFRConfig(
            name='AdobeVFR Synthetic',
            description="Synthetic (computer generated) images with texts for font recognition.",
            version="0.1.0",
            mode="syn",
        ),
        AdobeVFRConfig(
            name='AdobeVFR Real-world',
            description="Real-world images with texts for font recognition.",
            version="0.1.0",
            mode="real",
        )
    ]

    def _info(self):
        # print(os.path.join(dl_manager.manual_dir, _LABELS_FILENAME))
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(shape=_ADOBEVFR_IMAGE_SHAPE),
                "label": tfds.features.ClassLabel(num_classes=100)
                # "label": tfds.features.ClassLabel(num_classes=100, names_file=os.path.join(dl_manager.manual_dir, _LABELS_FILENAME))
            }),
            supervised_keys=("image", "label"),
            urls=[],
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """
        Returns SplitGenerators.

        Throws assertion error if required files were not found.
        """

        print("Splitting VFR for config", self.mode)
        # # extracted_path = dl_manager.download("https://www.dropbox.com/sh/o320sowg790cxpe/AAAJr15UkGQ7vEG9YitS2ETma/BCF%20format?dl=1")
        # # print("Downloaded to", extracted_path)
        # return [
        #     tfds.core.SplitGenerator(
        #         name=tfds.Split.TRAIN,
        #         num_shards=5,
        #         gen_kwargs={
        #             "path": extracted_path, "mode": "train"
        #         },
        #     ),
        #     tfds.core.SplitGenerator(
        #         name=tfds.Split.TEST,
        #         num_shards=1,
        #         gen_kwargs={
        #             "path": extracted_path, "mode": "test"
        #         },
        #     )
        # ]

        path = dl_manager.manual_dir
        if not tf.io.gfile.exists(path):
            self.throw_download_error(dl_manager.manual_dir)
        if self.builder_config.mode == "syn":
            path = os.path.join(path, "BCF Format")
            train = os.path.join(path, "VFR_syn_train")
            # elif self.builder_config.mode == 'raw':
        else:
            path = os.path.join(path, "Raw Images")
            train = os.path.join(path, "VFR_real_u")

        # Original dataset has wrong label for VFR_syn_test
        test = os.path.join(path, "VFR_real_test")
        return self.get_split_data(train, test)

    def _generate_examples(self, path, mode):
        """Yields examples."""
        print("Generating examples")
        images, labels = read_bcf(path, mode)
        for i in range(images.size()):
            # print("iteration", i)
            # print(images.get(i), labels[i])
            # with tf.io.gfile.GFile(image_filepath, "rb") as f:
            #     f.read(16)  # header
            #     buf = f.read(_MNIST_IMAGE_SIZE * _MNIST_IMAGE_SIZE * num_images)
            #     data = np.frombuffer(
            #         buf,
            #         dtype=np.uint8,
            #     ).reshape(num_images, _MNIST_IMAGE_SIZE, _MNIST_IMAGE_SIZE, 1)
            #     # return data
            yield {
                "image":
                    # np.frombuffer(
                    # Image.open(
                    #     io.BytesIO(
                    #         images.get(i)
                    #     )
                    # )).reshape(_ADOBEVFR_IMAGE_SHAPE),
                    np.asarray(tfds.core.lazy_imports.PIL_Image.open(
                        io.BytesIO(images.get(i))).resize((
                            _ADOBEVFR_IMAGE_SIZE, _ADOBEVFR_IMAGE_SIZE),
                            resample=tfds.core.lazy_imports.PIL_Image.NEAREST))
                    .reshape(_ADOBEVFR_IMAGE_SHAPE),
                "label": labels[i]
            }

    def throw_download_error(self, path):
        message = "You must download the dataset files manually and place them in: " + path
        raise AssertionError(message)

    def get_split_data(self, train_path, test_path):
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                num_shards=5,
                gen_kwargs={
                    "path": train_path, "mode": "train"
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                num_shards=1,
                gen_kwargs={
                    "path": test_path, "mode": "test"
                },
            )
        ]