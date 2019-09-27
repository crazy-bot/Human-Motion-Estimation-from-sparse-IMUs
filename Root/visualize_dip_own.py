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

############################ Use SMPL python library packages to instantiate SMPL body model ############
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
###################### Finish of Initialization of SMPL body model #############

############ below code is required if we want to create video - it saves each frame ina folder ######
# shutil.rmtree('/data/Guha/GR/Output/GT/')
# shutil.rmtree('/data/Guha/GR/Output/Prediction/')
# os.mkdir('/data/Guha/GR/Output/GT/')
# os.mkdir('/data/Guha/GR/Output/Prediction/')

file_path = '/data/Guha/GR/Output/TestSet/dip_own/test_our_data_all_frames.npz'
with open(file_path, 'rb') as file:
    data_dict = dict(np.load(file))
    gt = data_dict['gt']
    pred = data_dict['prediction']

    for act in range(18):
        act = 16
        gt = gt[act].reshape(-1,24*3)
        pred = pred[act].reshape(-1, 24*3)
        print('activity no: ',act)
        seq_len = gt.shape[0]
        print('seq len:',seq_len)
        for seq_num in range(seq_len):
            pose1 = gt[seq_num]
            pose2 = pred[seq_num]

            m1.pose[:] = pose1
            m1.pose[0] = np.pi

            m2.pose[:] = pose2
            m2.pose[0] = np.pi

            ####################to visualize runtime demo########################
            cv2.imshow('GT', rn1.r)
            cv2.imshow('Prediction', rn2.r)
            # Press Q on keyboard to  exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            print('gt values--', pose1)
            print('pred values--', pose2)

            ##################  while saving frames to create video ##################
            # gt_img = rn1.r * 255
            # pred_img = rn2.r * 255
            # cv2.imwrite('/data/Guha/GR/Output/GT/' + str(seq_num) + '.png', gt_img)
            # cv2.imwrite('/data/Guha/GR/Output/Prediction/' + str(seq_num) + '.png', pred_img)