import argparse
import math
from pathlib import Path
import random
import sys

import numpy as np
import tensorflow as tf

# Default data paths.
DEFAULT_TRAIN_INPUT_DIR_STR = './cars_images'
DEFAULT_TEST_INPUT_DIR_STR = './test_images'
DEFAULT_OUTPUT_DIR_STR = './tfrecords-output'
DEFAULT_NUM_SHARDS_TRAIN = 1
DEFAULT_NUM_SHARDS_TEST = 0


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class TFRecordsConverter(object):
    """Class that handles converting images to TFRecords."""

    def __init__(self, train_input_dir, test_input_dir, output_dir,
                 num_shards_train, num_shards_test):

        self.train_input_path = Path(train_input_dir)
        self.test_input_path = Path(test_input_dir)
        self.output_path = Path(output_dir)
        self.num_shards_train = num_shards_train
        self.num_shards_test = num_shards_test

        self.output_path.mkdir(exist_ok = True)

        # Get lists of images and labels.
        self.train_filenames = \
            self.process_image(self.train_input_path)
        '''
        self.test_filenames, self.test_labels = \
            self.process_image_labels(self.test_input_path)
        '''

        '''
        try:
            self.labels_file = (self.train_input_path / 'label.txt').read_text(encoding = 'utf-8').splitlines()
        except:
            print("label.txt could not be read.")
            sys.exit()
        '''

        # Counter for total number of images processed.
        self.counter = 0

    def process_image(self, input_path):

        # Build the lists.
        images = []
        im_files = list(input_path.glob('*.jpg'))
        for im_path in im_files:
            images.append(im_path)

        # Randomize the order of all the images/labels.
        shuffled_indices = list(range(len(images)))
        random.seed(12121)
        random.shuffle(shuffled_indices)
        filenames = [images[i] for i in shuffled_indices]

        return filenames

    def write_tfrecords_file(self, output_path, filenames, indices):
        """Writes out TFRecords file."""
        writer = tf.io.TFRecordWriter(str(output_path))
        for i in indices:
            filename = filenames[i]
            try:
                im_data = filename.read_bytes()
            except:
                print("The image file cannot be read. ")
                sys.exit()

            # Example is a data format that contains a key-value store, where
            # each key maps to a Feature message. In this case, each Example
            # contains two features. One will be a ByteList for the raw image
            # data and the other will be an Int64List containing the index of
            # the corresponding label in the labels list from the file.
            example = tf.train.Example(features=tf.train.Features(feature={
                'filename': _bytes_feature(bytes(filename.name, encoding = "utf-8")),
                'image/encoded': _bytes_feature(im_data)}))
            writer.write(example.SerializeToString())
            self.counter += 1
            if not self.counter % 1000:
                print('Processed {} images...'.format(self.counter))
        writer.close()

    def convert(self):
        """This function will drive the conversion to TFRecords.

        Here, we partition the data into a training and testing set, then
        divide each data set into the specified number of TFRecords shards.
        """

        print('Processing training set TFRecords...')
        num_files_train = len(self.train_filenames)
        files_per_shard = int(math.ceil(num_files_train /
                                        self.num_shards_train))
        start = 0
        for i in range(0, self.num_shards_train):
            shard_path = self.output_path / f'train-{i}.tfrecords'
            # Get a subset of indices to get only a subset of images/labels for
            # the current shard file.
            file_indices = np.arange(start, start+files_per_shard, dtype=int)
            start = start + files_per_shard
            self.write_tfrecords_file(shard_path, self.train_filenames, file_indices)

        # The remaining images will go in the final shard.
        file_indices = np.arange(start, num_files_train, dtype=int)
        if len(file_indices) > 0:
            final_shard_path = self.output_path / f'train-{self.num_shards_train}.tfrecords'
            self.write_tfrecords_file(final_shard_path, self.train_filenames, file_indices)

        '''
        print('Processing testing set TFRecords...')
        num_files_test = len(self.test_filenames)
        files_per_shard = math.ceil(num_files_test / self.num_shards_test)
        start = 0
        for i in range(0, self.num_shards_test):
            shard_path = self.output_path / f'test-{i}.tfrecords'
            # Get a subset of indices to get only a subset of images/labels for
            # the current shard file.
            file_indices = np.arange(start, start+files_per_shard, dtype=int)
            start = start + files_per_shard
            self.write_tfrecords_file(shard_path, self.test_filenames, self.test_labels, file_indices)

        # The remaining images will go in the final shard.
        file_indices = np.arange(start, num_files_test, dtype=int)
        if len(file_indices) > 0:
            final_shard_path = self.output_path / f'test-{self.num_shards_test}.tfrecords'
            self.write_tfrecords_file(final_shard_path, self.test_filenames, self.test_labels, file_indices)
        '''

        print(f'\nProcessed {self.counter} total images...')
        print(f'Number of training examples: {num_files_train}')
        #print(f'Number of testing examples: {num_files_test}')
        print(f'TFRecords files saved to {str(self.output_path)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-input-dir', type=str, dest='train_input_dir',
                        default=DEFAULT_TRAIN_INPUT_DIR_STR,
                        help='Train input directory to convert image files.')
    parser.add_argument('--test-input-dir', type=str, dest='test_input_dir',
                        default=DEFAULT_TEST_INPUT_DIR_STR,
                        help='Test input directory to convert image files.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR_STR,
                        help='Output directory to store TFRecords files.')
    parser.add_argument('--num-shards-train', type=int,
                        dest='num_shards_train',
                        default=DEFAULT_NUM_SHARDS_TRAIN,
                        help='Number of shards to divide training set '
                             'TFRecords into.')
    parser.add_argument('--num-shards-test', type=int,
                        dest='num_shards_test',
                        default=DEFAULT_NUM_SHARDS_TEST,
                        help='Number of shards to divide test set '
                             'TFRecords into.')
    args = parser.parse_args()
    converter = TFRecordsConverter(args.train_input_dir, args.test_input_dir, args.output_dir,
                                   args.num_shards_train, args.num_shards_test)
    converter.convert()
