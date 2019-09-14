import cv2
import  numpy as np
import glob
import scipy.io as sio

import numpy as np
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from GR19.smpl.smpl_webuser.serialization import load_model
import os
import transforms3d
import cv2
import pickle as pkl
import matplotlib.pyplot as plt
import copy
import pdb

## Load SMPL model (here we load the female model)
m1 = load_model('../models/basicModel_m_lbs_10_207_0_v1.0.0.pkl')
m1.betas[:] = np.random.rand(m1.betas.size) * .03

m2 = load_model('../models/basicModel_m_lbs_10_207_0_v1.0.0.pkl')
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


file_path = '/data/Guha/GR//code/dip18/train_and_eval/evaluation_results/validate_our_data_all_frames.npz'
with open(file_path, 'rb') as file:
    data_dict = dict(np.load(file))
    gt = data_dict['gt']
    pred = data_dict['prediction']
    gt_arr = []
    pred_arr = []
    height, width, layers = (640, 480, 3)
    size = (640, 480)

    act = 1
    gt = gt[act].reshape(-1, 24, 3)
    pred = pred[act].reshape(-1, 24, 3)
    print('activity no: ', act)
    seq_len = gt.shape[0]
    print('seq len:', seq_len)
    for seq_num in range(seq_len):
        pose1 = gt[seq_num]
        pose2 = pred[seq_num]

        m1.pose[:] = (pose1).reshape(72)
        m1.pose[0] = np.pi

        m2.pose[:] = (pose2).reshape(72)
        m2.pose[0] = np.pi

        # cv2.imshow('GT', rn1.r)
        # cv2.imshow('Prediction', rn2.r)
        # Press Q on keyboard to  exit
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        gt_img = rn1.r * 255
        pred_img = rn2.r * 255
        cv2.imwrite('/data/Guha/GR/Output/GT/'+str(seq_num)+'.png', gt_img)
        cv2.imwrite('/data/Guha/GR/Output/Prediction/'+str(seq_num)+'.png', pred_img)
        # gt_arr.append(gt_img)
        # pred_arr.append(pred_img)


# #out = cv2.VideoWriter('AW7.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, size))
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# gt_out = cv2.VideoWriter('Act_GT:'+act,fourcc, 20.0, size)
# pred_out = cv2.VideoWriter('Act_Pred:'+act,fourcc, 20.0, size)
#
# for i in range(len(gt_arr)):
#     gt_out.write(gt_arr[i])
# print ('gt finished')
# for i in range(len(pred_arr)):
#     pred_out.write(gt_arr[i])
# print ('pred finished')
# gt_out.release()
# pred_out.release()
