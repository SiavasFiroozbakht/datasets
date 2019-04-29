"""TODO(adobe_vfr): Add a description here."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os

import numpy as np
import tensorflow as tf

import tensorflow_datasets.public_api as tfds
# TODO(adobe_vfr): BibTeX citation
from tensorflow_datasets.core import api_utils

_CITATION = """
"""

# TODO(adobe_vfr):
_DESCRIPTION = """
"""


class bcf: #TODO Switch to tf.io.gfile
    def __init__(self, filename):
        self._filename = filename
        self._file = open(filename, 'rb')
        size = np.uint32(np.frombuffer(self._file.read(8), dtype=np.uint32)[0])
        file_sizes = np.frombuffer(self._file.read(8 * size),
                                   dtype=np.uint64)
        self._offsets = np.append(np.uint64(0),
                                  np.add.accumulate(file_sizes))

    def get(self, i):
        self._file.seek(np.int32(len(self._offsets) * 8 + self._offsets[i]))
        return self._file.read(self._offsets[i + 1] - self._offsets[i])

    def size(self):
        return len(self._offsets) - 1


def read_bcf(path, mode):
    base = {'train': 'train', 'test': 'vfr_large'}
    bcf_path = '%s/%s.bcf' % (path, base[mode])
    label_path = '%s/%s.label' % (path, base[mode])
    labels = read_label(label_path)
    images = bcf(bcf_path)
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

    BUILDER_CONFIGS = [
        AdobeVFRConfig(
            name='AdobeVFR BCF',
            description="A collection of images",
            version="0.1.0",
            mode="bcf",
        ),
        AdobeVFRConfig(
            name='AdobeVFR RAW',
            description="A collection of images",
            version="0.1.0",
            mode="raw",
        )
    ]

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(),
                "label": tfds.features.ClassLabel()
            }),
            supervised_keys=("image", "label"),
            urls=[],
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        path = '/data2/VFRForWeb/'
        if self.builder_config.mode == "bcf":
            if not tf.io.gfile.exists(path):
                message = "You must download the dataset files manually and place them in: " + dl_manager.manual_dir
                raise AssertionError(message)
            path = os.path.join(path, "BCF format")
            train = os.path.join(path, "VFR_syn_train")
            test = os.path.join(path, "VFR_real_test")
            return [
                tfds.core.SplitGenerator(
                    name=tfds.Split.TRAIN,
                    num_shards=5,
                    gen_kwargs={
                        "path": train, "mode": "train"
                    },
                ),
                tfds.core.SplitGenerator(
                    name=tfds.Split.TEST,
                    num_shards=1,
                    gen_kwargs={
                        "path": test, "mode": "test"
                    },
                )
            ]

    def _generate_examples(self, path, mode):
        """Yields examples."""
        images, labels = read_bcf(path, mode)
        for i in range(images.size()):
            yield {
                "image": tfds.core.lazy_imports.PIL_Image.open(
                    io.BytesIO(images.get(i))),
                "label": labels[i]
            }
