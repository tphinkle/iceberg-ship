import sys
import os




# Constants related to iceberg-ship detection challenge


num_img_rows = 75
num_img_cols = 75
num_features = 75*75*2 + 1

inputs = ['inc_angle'] + ['band_1_' + str(i) for i in range(num_img_rows*num_img_cols)] + \
            ['band_2_' + str(i) for i in range(num_img_rows*num_img_cols)]
output = 'is_iceberg'


# File locations

base_file_path = os.getcwd()[:os.getcwd().find('iceberg-ship')] + 'iceberg-ship'

train_raw_file_path = base_file_path + '/data/train/raw/train_reformatted.csv'
train_aug_file_path = base_file_path + '/data/train/augmented/train_rotref.csv'

test_raw_file_path = base_file_path + '/data/test/raw/test_reformatted.csv'
