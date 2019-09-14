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

## Create OpenDR renderer
rn1 = ColoredRenderer()

## Assign attributes to renderer
w, h = (640, 480)

rn1.camera = ProjectPoints(v=m1, rt=np.zeros(3), t=np.array([0, 0, 2.]), f=np.array([w, w]) / 2.,
                          c=np.array([w, h]) / 2., k=np.zeros(5))
rn1.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
rn1.set(v=m1, f=m1.f, bgcolor=np.zeros(3))

## Construct point light source
rn1.vc = LambertianPointLight(
    f=m1.f,
    v=rn1.v,
    num_verts=len(m1),
    light_pos=np.array([-1000, -1000, -2000]),
    vc=np.ones_like(m1) * .9,
    light_color=np.array([1., 1., 1.]))

Result_path = '/data/Guha/GR/Output/ValidationSet/18/'
fileList = os.listdir(Result_path)
print (fileList)
shutil.rmtree('/data/Guha/GR/Output/GT/')
shutil.rmtree('/data/Guha/GR/Output/Prediction/')
os.mkdir('/data/Guha/GR/Output/GT/')
os.mkdir('/data/Guha/GR/Output/Prediction/')

path = os.path.join(Result_path,'walking_1.npz')
with open(path, 'rb') as file:
    print (path)
    data_dict = dict(np.load(file))
    pred = data_dict['predictions']

    # for SMPL 15 to 24 joints
    pred_full = myUtil.smpl_reduced_to_full(pred.reshape(-1,15*4))

    #from quat to axis-angle
    pred_aa = myUtil.quat_to_aa_representation(pred_full,24)

    seq_len = pred_aa.shape[0]
    print('seq len:', seq_len)
    for seq_num in range(seq_len):

        pose1 = pred_aa[seq_num]

        m1.pose[:] = pose1
        m1.pose[0] = np.pi

        #to visualize demo
        cv2.imshow('Prediction', rn1.r)
        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # # while saving frames to create video
        # pred_img = rn1.r * 255
        # cv2.imwrite('/data/Guha/GR/Output/Prediction/' + str(seq_num) + '.png', pred_img)

