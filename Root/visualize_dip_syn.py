import sys
sys.path.insert(0, '/data/Guha/GR/code/GR19/smpl/smpl_webuser')
sys.path.insert(0, '/data/Guha/GR/code/GR19/smpl/models')

import shutil
import os
import numpy as np
import  myUtil
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from serialization import load_model
import cv2

## Load SMPL model (here we load the female model)
m1 = load_model('/data/Guha/GR/code/GR19/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl')
m1.betas[:] = np.random.rand(m1.betas.size) * .03

m2 = load_model('/data/Guha/GR/code/GR19/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl')
m2.betas[:] = np.random.rand(m2.betas.size) * .03
## Create OpenDR renderer
rn1 = ColoredRenderer()
rn2 = ColoredRenderer()

## Assign attributes to renderer
w, h = (640, 480)

rn1.camera = ProjectPoints(v=m1, rt=np.zeros(3), t=np.array([0, 0, 2.]), f=np.array([w, w]) / 2.,
                          c=np.array([w, h]) / 2., k=np.zeros(5))
rn1.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
rn1.set(v=m1, f=m1.f, bgcolor=np.zeros(3))

rn2.camera = ProjectPoints(v=m2, rt=np.zeros(3), t=np.array([0, 0, 2.]), f=np.array([w, w]) / 2.,
                          c=np.array([w, h]) / 2., k=np.zeros(5))
rn2.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
rn2.set(v=m2, f=m2.f, bgcolor=np.zeros(3))

## Construct point light source
rn1.vc = LambertianPointLight(
    f=m1.f,
    v=rn1.v,
    num_verts=len(m1),
    light_pos=np.array([-1000, -1000, -2000]),
    vc=np.ones_like(m1) * .9,
    light_color=np.array([1., 1., 1.]))

rn2.vc = LambertianPointLight(
    f=m2.f,
    v=rn2.v,
    num_verts=len(m2),
    light_pos=np.array([-1000, -1000, -2000]),
    vc=np.ones_like(m2) * .9,
    light_color=np.array([1., 1., 1.]))



Result_path = '/data/Guha/GR/Output/DIP_IMU/'
fileList = os.listdir(Result_path)
print (fileList)
shutil.rmtree('/data/Guha/GR/Output/GT/')
shutil.rmtree('/data/Guha/GR/Output/Prediction/')
os.mkdir('/data/Guha/GR/Output/GT/')
os.mkdir('/data/Guha/GR/Output/Prediction/')

path1 = '/data/Guha/GR/DIPIMUandOthers/DIP_IMU_and_Others/DIP_IMU/s_10/01_b.pkl'
path2 = '/data/Guha/GR/synthetic60FPS/H36/S8_WalkDog.pkl'
with open(path1, 'rb') as file1, open(path2, 'rb') as file2:
    print (path1)
    data_dict1 = np.load(file1, encoding='latin1')
    sample_pose1 = data_dict1['poses'].reshape(-1,135)
    sample_ori1 = data_dict1['ori'].reshape(-1, 5, 3,3)
    sample_acc1= data_dict1['acc'].reshape(-1, 5, 3)

    print(path2)
    data_dict2 = np.load(file2, encoding='latin1')
    sample_pose2 = data_dict2['poses'].reshape(-1,135)
    sample_ori2 = data_dict2['ori'].reshape(-1, 5, 3,3)
    sample_acc2 = data_dict2['acc'].reshape(-1, 5, 3)

    # for SMPL 15 to 24 joints
    dip = myUtil.smpl_reduced_to_full(sample_pose1)
    syn = myUtil.smpl_reduced_to_full(sample_pose2)

    #from quat to axis-angle
    dip_aa = myUtil.rot_matrix_to_aa(dip)
    syn_aa =  myUtil.rot_matrix_to_aa(syn)

    seq_len = dip_aa.shape[0]
    print('seq len:', seq_len)
    for seq_num in range(seq_len):

        pose1 = dip_aa[seq_num]
        pose2 = syn_aa[seq_num]

        m1.pose[:] = pose1
        m1.pose[0] = np.pi

        m2.pose[:] = pose2
        m2.pose[0] = np.pi


        #to visualize demo
        cv2.imshow('Dip', rn1.r)
        cv2.imshow('synthetic', rn2.r)
        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # # while saving frames to create video
        # gt_img = rn1.r * 255
        # pred_img = rn2.r * 255
        # cv2.imwrite('/data/Guha/GR/Output/GT/' + str(seq_num) + '.png', gt_img)
        # cv2.imwrite('/data/Guha/GR/Output/Prediction/' + str(seq_num) + '.png', pred_img)

