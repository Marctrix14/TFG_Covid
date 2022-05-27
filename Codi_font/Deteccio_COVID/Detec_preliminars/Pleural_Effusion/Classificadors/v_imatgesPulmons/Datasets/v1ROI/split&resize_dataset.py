# Original code is from https://github.com/cs230-stanford/cs230-code-examples/blob/master/tensorflow/vision/build_dataset.py
# Tutorial about spliitting a dataset: https://cs230.stanford.edu/blog/split/

"""Split the original dataset into train/valid/test.

The dataset comes in the following format:
    train_test/
        normal/
            normal1.png
            ...
        effusion/
            effusion1.png
            ...
    validation/
        normal/
            normal5.png
            ...
        effusion/
            effusion5.png
            ...
"""

# IMPORTANT!!!
# This script splits the original dataset's folder "train_test" into "train"/ "valid".
# 90% of "train_test" images will be saved inside train_val/train folder of the new dataset.
# 10% of "train_test" images will be saved inside train_val/valid folder of the new dataset.
# All oimages of "validation" folder from the original dataset will be saved inside "test" folder of the new dataset.

import argparse
from cgi import test
import random
import os
import shutil


from PIL import Image
from torch import normal
from tqdm import tqdm


SIZE = 300

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='ds_original', help="Directory with the original dataset")
parser.add_argument('--output_dir', default='ds_resized', help="Where to write the new data")


def resize_and_save(filename, output_dir, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    image.save(os.path.join(output_dir, filename.split('\\')[-1]))


def save(filename, output_dir):
    """Save the image contained in `filename` to the `output_dir`"""
    image = Image.open(filename)
    image.save(os.path.join(output_dir, filename.split('\\')[-1]))

if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    train_data_dir = os.path.join(args.data_dir, 'train_test')

    test_data_dir = os.path.join(args.data_dir, 'validation') 

    # Get the filenames in each directory (train and test)
    effusion_dir = os.path.join(train_data_dir, 'effusion_p')
    effusionFilenames = os.listdir(effusion_dir)
    effusionFilenames = [os.path.join(effusion_dir, f) for f in effusionFilenames if f.endswith('.png')]

    normal_dir = os.path.join(train_data_dir, 'normal_p')
    normalFilenames = os.listdir(normal_dir)
    normalFilenames = [os.path.join(normal_dir, f) for f in normalFilenames if f.endswith('.png')]
 
    # Test filenames
    effusion_test_dir = os.path.join(test_data_dir, 'effusion_p')
    testEffusionFilenames = os.listdir(effusion_test_dir)
    testEffusionFilenames = [os.path.join(effusion_test_dir, f) for f in testEffusionFilenames if f.endswith('.png')]
 
    normal_test_dir = os.path.join(test_data_dir, 'normal_p')
    testNormalFilenames = os.listdir(normal_test_dir)
    testNormalFilenames = [os.path.join(normal_test_dir, f) for f in testNormalFilenames if f.endswith('.png')]

    # Split the images in into 90% train and 10% valid
    # To make sure to have the same split each time this code is run, 
    # we need to fix the random seed before shuffling the filenames
    random.seed(230)
    effusionFilenames.sort() # make sure that the filenames have a fixed order before shuffling
    random.shuffle(effusionFilenames) # shuffles the ordering of filenames (deterministic given the chosen seed)

    random.seed(230)
    normalFilenames.sort() # make sure that the filenames have a fixed order before shuffling
    random.shuffle(normalFilenames) # shuffles the ordering of filenames (deterministic given the chosen seed)

    split = int((0.9 * (len(effusionFilenames)+len(normalFilenames))) / 2) # we divide 2 because each set (train/val/test) has 2 subsets (normal/effusion) 
    train_effusion_filenames = effusionFilenames[:split] 
    train_normal_filenames = normalFilenames[:split]
    valid_effusion_filenames = effusionFilenames[split:]
    valid_normal_filenames = normalFilenames[split:]

    filenames = {'train': (train_effusion_filenames, train_normal_filenames),
                 'valid': (valid_effusion_filenames, valid_normal_filenames),
                 'test': (testEffusionFilenames, testNormalFilenames)}

    # Create the new dataset folder named "ds_splitted"  
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))


    output_train_val_dir = os.path.join(args.output_dir, "train_val")

    # Process train, valid and test
    for split in ['train', 'valid', 'test']:
        if split == 'test':
            output_dir_split = os.path.join(args.output_dir, split) # create 'test' folder inside the new dataset
        else:
            output_dir_split = os.path.join(output_train_val_dir, split) # create 'train' / 'valid' folder inside the folder "train_val" of the new dataset

        normal_dir = os.path.join(output_dir_split, 'normal')
        effusion_dir = os.path.join(output_dir_split, 'effusion')
        # Remove all previous content inside 'normal' and 'effusion' folders
        shutil.rmtree(normal_dir, ignore_errors=True)
        shutil.rmtree(effusion_dir, ignore_errors=True)

        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(effusion_dir, exist_ok=True)

        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
        for i in range(len(filenames[split][0])):
            resize_and_save(filenames[split][0][i], effusion_dir)
            resize_and_save(filenames[split][1][i], normal_dir)

    print("Done building dataset")
