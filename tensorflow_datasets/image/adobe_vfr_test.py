"""TODO(adobe_vfr): Add a description here."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_datasets import testing
from tensorflow_datasets.image import adobe_vfr


class AdobeVfrTest(testing.DatasetBuilderTestCase):
  # TODO(adobe_vfr):
  DATASET_CLASS = adobe_vfr.AdobeVFR
  SPLITS = {
      "train": 3,  # Number of fake train example
      "test": 1,  # Number of fake test example
  }

if __name__ == "__main__":
  testing.test_main()

