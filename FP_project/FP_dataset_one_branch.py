import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.utils.data as data
import numpy as np
import random
import os
import csv
from PIL import Image
import random
from Landmark_Detect import symm_diff 


class FP_Dataset_one_branch(Dataset):
    def __init__(self, frame_num, k_in_5, transform=None, mode='train'):
        self.mode = mode
        self.transform = transform if transform is not None \
            else transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])  #! need to update
        self.frame_num = frame_num  # better in [5, 20]

        self.id_path_train = []
        self.id_label_train = []
        self.geo_train = []

        self.id_path_test = []
        self.id_label_test = []
        self.geo_test = []

        self.id_path = []  # all the path info of id
        self.id_label = []  # all the label info of id

        # self.weights = [1.778, 7.724, 14.933, 11.2, 8.615, 28.0]
        self.k_in_5 = k_in_5
        self.load_csv()
        self.split_random = True
        self.train_test_split()

    def load_csv(self):
        label_path = '/media/data2/ganwei/Xiehe_Project/Xiehe_Data/1_BrowLift.csv'
        with open(label_path, 'rt') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                self.id_path.append(row[0])
                self.id_label.append(int(row[1]))

    def train_test_split(self):
        fold_len = len(self.id_label) // 5
        if self.split_random:  # random split
            random.seed(42)
            test_ids = random.sample(range(len(self.id_label)), fold_len)
        else:  # normal split
            test_ids = np.arange(fold_len*self.k_in_5, fold_len*self.k_in_5+fold_len, 1)

        for i in range(len(self.id_label)):
            if i in test_ids:
                self.id_path_test.append(self.id_path[i])
                self.id_label_test.append(self.id_label[i])
            else:
                self.id_path_train.append(self.id_path[i])
                self.id_label_train.append(self.id_label[i])
        print('Loading the CSV file of Dataset...', self.__len__())

    def __len__(self):
        if self.mode == 'train':
            return len(self.id_label_train)
        else:
            return len(self.id_label_test)
    
    def file_filter(self, f, is_folder=False):
        if is_folder:  # only need 'int' id for folder
            try: 
                int(f)
                return True
            except ValueError: return False
        else:  # check if imgs
            if f[-4:] in ['.jpg', '.png']:
                return True
            else:
                return False

    def read_images(self, id_path, frame_num, use_transform):
        random.seed(0)
        act_id = {1:'1_BrowLift', 2:'2_EyeClose', 3:'3_CheekBlow', 4:'4_TeethShow'}

        X = []  # for 4 actions' frames
        Geo = []  # for 4 actions' landmark attributes
        stat = []  # for 4 actions' static frame 

        frame_list = os.listdir(id_path)
        frame_list = list(filter(lambda x: self.file_filter(x, is_folder=False), frame_list))  # filter for image files
        frame_list.sort(key = lambda x:int(x[:-4].split('_')[-1]))  # os.listdir获取文件名无序，自行按frame_x尾标排序
        
        start_rand = random.randint(0, 5)  #! 想要随机均匀抽取 N 帧，暂且如此
        frame_pick = np.linspace(start=start_rand, stop=len(frame_list)-1, num=frame_num)
            
        action_X = []
        action_Geo = []

        for f_i in range(len(frame_pick)):
            img_path = os.path.join(id_path+'_roi', frame_list[int(frame_pick[f_i])])  # frames in ROI folders
            img = Image.open(img_path)  # .convert('L')
            if use_transform is not None:
                img = use_transform(img)
            img = img.unsqueeze(0)  # Add batch dimension
            action_X.append(img)  # Concatenate tensors

            if f_i == len(frame_pick)//2:
                mid_img = Image.open(os.path.join(id_path, frame_list[int(frame_pick[f_i])]))
                if use_transform is not None:
                    mid_img = use_transform(mid_img)
                stat.append(mid_img)

            land_path = os.path.join(id_path, frame_list[int(frame_pick[f_i])])[:-4] + '_land.npy'
            land = np.load(land_path)
            # geo_info = symm_diff(img, land).combine()  #! Not sure to use land or symm_diff; 不确定用8/9/2个？
            geo_info = symm_diff(img, land).get_dist_diff()  
            
            action_Geo.append(geo_info[2*0:2*0+2])  #! ONLY use landmark diffs, NEED to UPDATE  # geo_info[2*i:2*i+2]

        X = torch.stack(action_X, dim=1)  #.squeeze()  #! Need to Check the DIMs later!
        Geo = np.stack(action_Geo)
        stat = np.stack(stat).squeeze()
        return X, Geo, stat

    def __getitem__(self, index):
        if self.mode == 'train':
            X, Geo, stat = self.read_images(self.id_path_train[index], self.frame_num, self.transform)
            y = torch.from_numpy(np.array(self.id_label_train[index], dtype = int))
        else:
            X, Geo, stat = self.read_images(self.id_path_test[index], self.frame_num, self.transform)
            y = torch.from_numpy(np.array(self.id_label_test[index], dtype = int))
        return X, Geo, stat, y





if __name__ == '__main__':
    ## Just for Test...
    my_dataset = FP_Dataset_one_branch(frame_num=5, k_in_5=1, transform=None, mode='train')
    X, Geo, stat, y = my_dataset.__getitem__(1)
    print(my_dataset.__len__())
