import numpy as np
import random
import os
import csv
import math
import argparse
from PIL import Image


action_list = ['1_BrowLift', '2_EyeClose', '3_CheekBlow', '4_TeethShow']

def data_file_csv(data_root, save_root, target_action_id):
    if not os.path.exists(data_root) or not os.path.exists(save_root):
        raise FileNotFoundError(f"Data / Save Root Not Found!")
    id_list = os.listdir(data_root)
    id_list.sort()
    
    # Read Original Label File
    label_path = '/media/data2/ganwei/Xiehe_Project/Xiehe_Data/Xiehe_labels.csv'
    id_label_pairs = {}
    with open(label_path, 'rt') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            id_label_pairs[int(row[0])] = int(row[target_action_id+1]) - 1  # label in [1,6], level-1

    target_action = action_list[target_action_id]
    target_files_save = []
    for id in id_list:
        label = id_label_pairs[int(id)]  # pick the corresponding id label
        action_file = os.path.join(data_root, id, target_action)
        reap_files = os.listdir(action_file)
        reap_files = [os.path.join(action_file, folder) for folder in reap_files if folder.isdigit()]
        target_files_save.extend([file, label] for file in reap_files)  #! [file]: ensure saving format correct
    
    save_file = os.path.join(save_root, '{}.csv'.format(target_action))
    print('Save {} files into {}'.format(target_action, save_file))
    with open(save_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(target_files_save)


def my_parser():
    parser = argparse.ArgumentParser(description='Generate One Branch Data File')
    parser.add_argument('--data_root', type=str, default='/media/data2/ganwei/Xiehe_Project/Xiehe_Data/processed_data')
    parser.add_argument('--save_root', type=str, default='/media/data2/ganwei/Xiehe_Project/Xiehe_Data')
    opts = parser.parse_args()
    return opts


if __name__=='__main__':
    opts = my_parser()
    
    for target_action_id in range(4):
        data_file_csv(opts.data_root, opts.save_root, target_action_id)
