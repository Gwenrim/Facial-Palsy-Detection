import dlib
import cv2
import numpy as np
import random
import os
import math
from multiprocessing import Process
from PIL import Image

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Landmark_Detect import symm_diff


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/media/data2/ganwei/Xiehe_Project/FP_project/preprocess/shape_predictor_68_face_landmarks.dat')


class FaceFunctions:
    def __init__(self, face_root, save_face_root = None):
        self.FACE_MARGIN = 5
        self.face_root = face_root
        self.face_ids = list(filter(self.id_filer, os.listdir(face_root)))
        self.save_face_root = save_face_root if save_face_root else face_root

    def id_filer(self, id):
        return id[-4:] in ['.jpg', '.png', '.bmp'] 

    def crop_align_face(self, save = False, debug = False):
        self.face_ids.sort(key = lambda x:int(x[:-4].split('_')[1])) 
        ids = [random.choice(self.face_ids)] if debug else self.face_ids
        
        for id in ids:
            img = cv2.imread(os.path.join(self.face_root, id), cv2.COLOR_BGR2RGB)
            if img is None: 
                print("Empty Image!", os.path.join(self.face_root, id))
                continue
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            rects = detector(gray_img, 0)
            if len(rects) == 0:
                print(os.path.join(self.face_root, id), "NO FACE!")
                continue
            landmarks = np.array([[pt.x, pt.y] for pt in predictor(img, rects[0]).parts()]).reshape(-1,2)

            h, w = img.shape[:2]
            eye_center = ((landmarks[36, 0] + landmarks[45, 0]) * 1. / 2,  # 计算左右外眼角中心坐标
                          (landmarks[36, 1] + landmarks[45, 1]) * 1. / 2)
            dx = (landmarks[45, 0] - landmarks[36, 0])
            dy = (landmarks[45, 1] - landmarks[36, 1])

            angle = math.atan2(dy, dx) * 180. / math.pi  # 计算角度
            RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)  # 计算仿射矩阵
            aligned_img = cv2.warpAffine(img, RotateMatrix, (w, h))  # 进行放射变换，即旋转

            lm_trans = landmarks - eye_center
            lm_rotate = np.matmul(lm_trans, RotateMatrix[:,:2].T) + eye_center

            
            center = lm_rotate[29]  # 鼻头处
            xy_min = lm_rotate.min(axis = 0) 
            xy_max = lm_rotate.max(axis = 0)
            
            width = max(np.max(xy_max - center), np.max(center - xy_min))
            width += width // 20
            
            xy_min = (center - width).astype(np.int32)
            xy_max = (center + width).astype(np.int32)

            if xy_min[0] < 0: 
                xy_min[0] = 0
                # xy_max[0, 0] = 2 * width 
                
            if xy_min[1] < 0: 
                xy_min[1] = 0
                # xy_max[0, 1] = 2 * width 
                
            if xy_max[0] > aligned_img.shape[1]: 
                xy_max[0] = aligned_img.shape[1] - 1
                # xy_min[0, 0] = xy_max[0, 0] - 2 * width 
                
            if xy_max[1] > aligned_img.shape[0]: 
                xy_max[1] = aligned_img.shape[0] - 1
                # xy_min[0, 1] = xy_max[0, 1] - 2 * width 
            
            crop_aligned_img = aligned_img[xy_min[1]: xy_max[1], xy_min[0]: xy_max[0]]
            
            lm_rotate[:,0] = lm_rotate[:,0] - xy_min[0]
            lm_rotate[:,1] = lm_rotate[:,1] - xy_min[1]
            if debug:
                cv2.imshow('landmarks',crop_aligned_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                break
            if save:
                if not os.path.exists(self.save_face_root):
                    os.makedirs(self.save_face_root)
                if crop_aligned_img.size == 0:
                    print(self.save_face_root, id, 'crop_aligned_img.size ==  0')
                    continue
                # if not crop_aligned_img.shape[0] == crop_aligned_img.shape[1]:
                #     print(self.save_face_root, id, 'crop_aligned_img.shape[0] == crop_aligned_img.shape[1]')
                #     continue
                cv2.imwrite(os.path.join(self.save_face_root, '{}'.format(id)), crop_aligned_img)
                np.save(os.path.join(self.save_face_root, '{}_land'.format(id[:-4])), lm_rotate)
                # print(self.save_face_root, '{}'.format(id))
                # print(self.save_face_root, 'ac{}---saved'.format(id))


def multiprocessing_align_crop_face():
    source_root = '/media/ljy/ubuntu_disk1/gw/Xiehe/Picked_Data/new_data'
    save_root = '/media/ljy/ubuntu_disk1/gw/Xiehe/Picked_Data/processed_new_data'
    #root = '/media/data1/huangbin/AU/relabeled_DISFA/all_data'
    folders = os.listdir(source_root)
    folders.sort(key = lambda x:int(x)) 
    iter = 6
    for folder in folders:
        print("processing", folder)
        subfolders = os.listdir(os.path.join(source_root, folder))
        subfolders.sort(key = lambda x:int(x.split('_')[0])) 
        for subfolder in subfolders:  # for each repetition
            print("processing", folder, subfolder)
            subrepts = os.listdir(os.path.join(source_root, folder, subfolder))
            for rept in subrepts:
                print("processing", folder, subfolder, rept)
                ff = FaceFunctions(face_root = os.path.join(source_root, folder, subfolder, rept), save_face_root = os.path.join(save_root, folder, subfolder, rept))
                p = Process(target = ff.crop_align_face, args=(True, False))
                p.start()
                iter -= 1
                if iter == 0: 
                    iter = 1
                    p.join()


# RENAME FOLDER
# if act_folder == '3_TeethShow':
#     os.rename(os.path.join(id_folder_path,act_folder), os.path.join(id_folder_path,'4_TeethShow'))
# elif act_folder == '4_CheekBlow':
#     os.rename(os.path.join(id_folder_path,act_folder), os.path.join(id_folder_path,'3_CheekBlow'))


def file_filter(f, is_folder=False):
    if is_folder:  # only need 'int' id for folder
        try: 
            int(f)
            return True
        except ValueError: return False
    else:  # check imgs
        if f[-4:] in ['.jpg', '.png']:
            return True
        else:
            return False
            
def region_crop():
    source_root = '/media/data2/ganwei/Xiehe_Project/Xiehe_Data/processed_data'
    id_folders = os.listdir(source_root)
    id_folders.sort(key = lambda x:int(x)) 

    iter = 6
    for id_folder in id_folders:
        print("processing", id_folder)
        act_folders = os.listdir(os.path.join(source_root, id_folder))
        act_folders.sort(key = lambda x:int(x.split('_')[0])) 
        for act_folder in act_folders:  # for each repetition
            act_id = int(act_folder.split('_')[0])
            print("processing", id_folder, act_folder)
            rept_folders = os.listdir(os.path.join(source_root, id_folder, act_folder))
            rept_folders = list(filter(lambda x: file_filter(x, is_folder=True), rept_folders))
            rept_folders.sort(key = lambda x:int(x))
            for rept in rept_folders:
                print("processing", id_folder, act_folder, rept)
                rept_path = os.path.join(source_root, id_folder, act_folder, rept)
                if not os.path.exists(rept_path+'_roi'):
                    os.makedirs(rept_path+'_roi')
                frame_list = os.listdir(rept_path)
                frame_list = list(filter(lambda x: file_filter(x, is_folder=False), frame_list))
                frame_list.sort(key = lambda x:int(x[:-4].split('_')[1]))
                for frame in frame_list:
                    img = Image.open(os.path.join(rept_path, frame))
                    lm = np.load(os.path.join(rept_path, frame[:-4]+'_land.npy'))
                    BB = symm_diff(img, lm).get_area()[act_id-1]
                    img.crop((BB[0,0], BB[0,1], BB[1,0], BB[1,1])).save(os.path.join(rept_path+'_roi', frame))
                

if __name__=='__main__':
    # multiprocessing_align_crop_face()
    region_crop()

    ### Test by One
    # source_root = '/home/ganwei/codes/Face_Alignment/Others/Face_Synthesis/photoes_result'
    # save_root = '/home/ganwei/codes/Face_Alignment/Others/Face_Synthesis/processed_results'
    # ff = FaceFunctions(face_root=source_root, save_face_root=save_root)
    # ff.crop_align_face(True, False)
    # print('0')