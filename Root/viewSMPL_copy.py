import sys
sys.path.insert(0, '/data/Guha/GR/code/GR19/smpl/smpl_webuser')
sys.path.insert(0, '/data/Guha/GR/code/GR19/smpl/models')

import numpy as np
# from opendr.renderer import ColoredRenderer
# from opendr.lighting import LambertianPointLight
# from opendr.camera import ProjectPoints
# from serialization import load_model
import os
import transforms3d
import cv2
import pickle as pkl
import matplotlib.pyplot as plt
import copy
import pdb
import itertools

## Load SMPL model (here we load the female model)
# m1 = load_model('/data/Guha/GR/code/GR19/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl')
# m1.betas[:] = np.random.rand(m1.betas.size) * .03
#
# m2 = load_model('/data/Guha/GR/code/GR19/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl')
# m2.betas[:] = np.random.rand(m2.betas.size) * .03
# ## Create OpenDR renderer
# rn1 = ColoredRenderer()
# rn2 = ColoredRenderer()
#
# ## Assign attributes to renderer
# w, h = (640, 480)
#
# rn1.camera = ProjectPoints(v=m1, rt=np.zeros(3), t=np.array([0, 0, 2.]), f=np.array([w, w]) / 2.,
#                           c=np.array([w, h]) / 2., k=np.zeros(5))
# rn1.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
# rn1.set(v=m1, f=m1.f, bgcolor=np.zeros(3))
#
# rn2.camera = ProjectPoints(v=m2, rt=np.zeros(3), t=np.array([0, 0, 2.]), f=np.array([w, w]) / 2.,
#                           c=np.array([w, h]) / 2., k=np.zeros(5))
# rn2.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
# rn2.set(v=m2, f=m2.f, bgcolor=np.zeros(3))
#
# ## Construct point light source
# rn1.vc = LambertianPointLight(
#     f=m1.f,
#     v=rn1.v,
#     num_verts=len(m1),
#     light_pos=np.array([-1000, -1000, -2000]),
#     vc=np.ones_like(m1) * .9,
#     light_color=np.array([1., 1., 1.]))
#
# rn2.vc = LambertianPointLight(
#     f=m2.f,
#     v=rn2.v,
#     num_verts=len(m2),
#     light_pos=np.array([-1000, -1000, -2000]),
#     vc=np.ones_like(m2) * .9,
#     light_color=np.array([1., 1., 1.]))



######################## read DataFile - DIP_IMU ###########################
# DIPPath = '/data/Guha/GR/DIPIMUandOthers/DIP_IMU_and_Others/DIP_IMU/s_10'
# imu_order = ['head', 'spine2', 'belly', 'lchest', 'rchest', 'lshoulder', 'rshoulder', 'lelbow', 'relbow', 'lhip', 'rhip', 'lknee', 'rknee', 'lwrist', 'lwrist', 'lankle', 'rankle']
# SENSORS = [ 'lknee', 'rknee','lelbow', 'relbow', 'head','belly']
# sensor_idx = [11,12,7,8,0,2]
# fileList = os.listdir(DIPPath)
# print (fileList)
#
# for file_num in range(len(fileList)):
#     path = os.path.join(DIPPath,fileList[4])
#     with open(path, 'rb') as f:
#         print (path)
#         data_dict = pkl.load(f)
#         imu_ori = data_dict['imu'][:,:,0:9]
#         imu_acc = data_dict['imu'][:,:,9:12]
#         imu_acc = np.asarray([imu_acc[:,k,:] for k in sensor_idx])
#         imu_acc = imu_acc.reshape(-1,6,3)
#         gt = data_dict['gt'].reshape(-1,24,3)
#         seq_len = gt.shape[0]
#         print('seq len:', seq_len)
#
#         head_0_frame = imu_ori[0, 0, :].reshape(3, 3)
#         norm_ori_1 = np.asarray(
#             [np.dot(np.linalg.inv(head_0_frame), imu_ori[k, j, :].reshape(3, 3)) for k, j in itertools.product(range(seq_len), sensor_idx)])
#         norm_ori_1 = norm_ori_1.reshape(-1,6,3,3)
#
#         R_TS_0 = norm_ori_1[0,0,:,:]
#         R_TS_1 = norm_ori_1[0, 1, :, :]
#         #inv_R_TB_0 = np.asarray([[0,-1,0],[1,0,0],[0,0,1]])
#         inv_R_TB_0 = np.asarray([[-0.0254474,  -0.98612685 , 0.16403132],
#  [ 0.99638575, -0.01171777 , 0.08413162],
#  [-0.08104237 , 0.1655794 ,  0.98286092]])
#         #inv_R_TB_1 = np.asarray([[0,1,0],[-1,0,0],[0,0,1]])
#         inv_R_TB_1 = np.asarray([[-0.01307566,  0.97978612, - 0.1996201],
#          [-0.97720228 , 0.02978678 , 0.21021048],
#         [0.21190735,
#         0.19781786,
#         0.95705975]])
#
#         R_BS_0 = np.dot(inv_R_TB_0,R_TS_0)
#         R_BS_1 = np.dot(inv_R_TB_1,R_TS_1)
#
#
#         bone2S =norm_ori_1[0,:,:,:]
#
#         bone2S[0,:,:] = R_BS_0
#         bone2S[ 1, :, :] = R_BS_1
#
#         # bone_ori = np.asarray([np.dot(norm_ori_1[k, j, :, :], np.linalg.inv(bone2S[j])) for k,j in itertools.product(range(seq_len),range(6))])
#
#         bone_ori = np.asarray([np.dot(bone2S[j],norm_ori_1[k, j, :, :]) for k, j in
#                                itertools.product(range(seq_len), range(6))])
#
#
#         bone_ori = bone_ori.reshape(-1,6,3,3)
#
#         norm_ori_2 = np.asarray(
#             [np.dot(np.linalg.inv(bone_ori[i,5,:,:]), bone_ori[i, j, :,:]) for i, j in itertools.product(range(seq_len), range(6))])
#
#         norm_ori_2 = norm_ori_2.reshape(-1,6,3,3)
#
#         #acceleration
#         norm_acc_1 = np.asarray(
#             [np.dot(norm_ori_1[k, j, :].reshape(3, 3),imu_acc[k,j,:]) for k, j in itertools.product(range(seq_len), range(6))])
#         norm_acc_1 = norm_acc_1.reshape(-1,6,3)
#
#         norm_acc = np.asarray([np.dot(np.linalg.inv(norm_ori_2[k,5,:,:]), (norm_acc_1[k, j, :] - norm_acc_1[k, 5, :])) for k, j in itertools.product(range(seq_len), range(6))])
#         norm_acc_1 = norm_acc_1.reshape(-1, 6, 3)
#         for i in range(seq_len):
#
#             bone_ori = np.asarray([np.dot(bone2S[j],norm_ori_1[i,j,:,:]) for j in range(6)])
#             root_ori = imu_ori[i,2,:].reshape(3,3)
#             norm_ori = np.asarray([np.dot(np.linalg.inv(root_ori) , imu_ori[i,j,:].reshape(3,3)) for j in sensor_idx])
#
#             root_acc = imu_acc[i, 2, :].reshape(3)
#             norm_acc = np.asarray([np.dot(np.linalg.inv(root_ori) , (imu_acc[i, j, :] - root_acc)) for j in sensor_idx])
#
#             pose = gt[i]
#
#             # ########## NormalizeZeroMeanUnitVariance #############
#             # pose[:,0] = (pose[:,0 ] - np.mean(pose[:,0],axis=0)) / np.std(pose[:,0],axis=0)
#             # pose[:,1] = (pose[:,1] - np.mean(pose[:,1], axis=0)) / np.std(pose[:,1], axis=0)
#             # pose[:,2] = (pose[:,2] - np.mean(pose[:,2], axis=0)) / np.std(pose[:,2], axis=0)
#
#             Ipose = np.asarray([0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  2.40190000e-02,
#                                 -5.65980000e-02, -6.00000000e-02,  4.01920000e-03,  1.07109000e-01,
#                                 4.00000000e-02, -6.85148000e-02,  2.97456624e-02,  0.00000000e+00,
#                                 -1.50640000e-02,  1.22855000e-01, -2.80000000e-03, -2.51200000e-04,
#                                 -7.49550000e-02,  2.80000000e-03, -1.97083023e-02, -5.90451714e-02,
#                                 0.00000000e+00, -3.69410000e-02, -1.39870000e-02,  1.09700000e-03,
#                                 3.08240000e-02,  1.10824000e-01,  5.58300000e-02,  3.68217919e-02,
#                                 -9.79798425e-03,  0.00000000e+00,  7.38820000e-02,  8.71628260e-02,
#                                 1.15933226e-01, -1.36454340e-02,  7.27977859e-02, -2.04008074e-01,
#                                 2.75226449e-02,  3.74526146e-02, -3.26716395e-02,  7.95110800e-02,
#                                 1.55932400e-02, -3.61916400e-01,  7.95110800e-02, -1.55932400e-02,
#                                 3.61916400e-01,  4.14048214e-02, -5.75496269e-03,  6.12744933e-02,
#                                 -1.08706800e-01, -1.39227600e-02, -1.10823788e+00,  7.96932000e-02,
#                                 2.02324166e-01,  1.06021472e+00,  1.14999360e-01, -1.25600000e-01,
#                                 -1.25600000e-01,  5.21993600e-02,  1.25600000e-01,  1.25600000e-01,
#                                 1.34247560e-01, -9.28749200e-02, -8.79514000e-02,  1.31183097e-02,
#                                 4.85928009e-02,  6.31077200e-02, -2.00966541e-01, -3.42684870e-02,
#                                 -1.76926440e-01, -1.28807464e-01,  1.02772092e-01,  2.61631080e-01 ])
#             Ipose[50]=0.0
#             Ipose[53]=0.0
#             Tpose = np.zeros(72)
#             Tpose[3] = -3.14
#
#             ########## NormalizeZ Min-Max #############
#             # pose[:, 0] = (pose[:, 0] - np.min(pose[:, 0], axis=0)) / (np.max(pose[:, 0], axis=0) - np.min(pose[:, 0]))
#             # pose[:, 1] = (pose[:, 1] - np.min(pose[:, 1], axis=0)) / (np.max(pose[:, 1], axis=0) - np.min(pose[:, 1]))
#             # pose[:, 2] = (pose[:, 2] - np.min(pose[:, 2], axis=0)) / (np.max(pose[:, 2], axis=0) - np.min(pose[:, 2]))
#
#
#             #m1.pose[:] = (pose *.2).reshape(72)
#             m1.pose[:] = pose.reshape(72)
#             m1.pose[0] = np.pi
#
#             cv2.imshow('GT', rn1.r)
#             cv2.waitKey(1)


######################## read DataFile -DIP_IMU_nn - imu_own_test.npz ###########################
file_path = '/data/Guha/GR//code/dip18/train_and_eval/data/dipIMU/imu_own_test.npz'
with open(file_path, 'rb') as file:
    data_dict = dict(np.load(file))
    gt = data_dict['gt']
    pred = data_dict['prediction']

    for act in range(18):
        act = 1
        gt = gt[act].reshape(-1,24,3)
        pred = pred[act].reshape(-1, 24, 3)
        print('activity no: ',act)
        seq_len = gt.shape[0]
        print('seq len:',seq_len)
        for seq_num in range(seq_len):
            pose1 = gt[seq_num]
            pose2 = pred[seq_num]

            # print ('GT-- min:', np.min(pose1[:,0]),np.min(pose1[:,1]),np.min(pose1[:,2]), 'max:', np.min(pose1[:,0]),np.min(pose1[:,1]),np.min(pose1[:,2]))
            # print ('Pred-- min:', np.min(pose2[:,0]),np.min(pose2[:,1]),np.min(pose2[:,2]), 'max:', np.max(pose2[:,0]),np.max(pose2[:,1]),np.max(pose2[:,2]))

            # ########## NormalizeZeroMeanUnitVariance #############
            # pose1[:, 0] = (pose1[:, 0] - np.mean(pose1[:, 0], axis=0)) / np.std(pose1[:, 0], axis=0)
            # pose1[:, 1] = (pose1[:, 1] - np.mean(pose1[:, 1], axis=0)) / np.std(pose1[:, 1], axis=0)
            # pose1[:, 2] = (pose1[:, 2] - np.mean(pose1[:, 2], axis=0)) / np.std(pose1[:, 2], axis=0)
            #
            # # ########## NormalizeZeroMeanUnitVariance #############
            # pose2[:, 0] = (pose2[:, 0] - np.mean(pose2[:, 0], axis=0)) / np.std(pose2[:, 0], axis=0)
            # pose2[:, 1] = (pose2[:, 1] - np.mean(pose2[:, 1], axis=0)) / np.std(pose2[:, 1], axis=0)
            # pose2[:, 2] = (pose2[:, 2] - np.mean(pose2[:, 2], axis=0)) / np.std(pose2[:, 2], axis=0)


            # m1.pose[:] = (pose1 ).reshape(72)
            # m1.pose[0] = np.pi
            #
            # m2.pose[:] = (pose2 ).reshape(72)
            # m2.pose[0] = np.pi
            #
            # cv2.imshow('GT', rn1.r)
            # cv2.imshow('Prediction', rn2.r)
            # # Press Q on keyboard to  exit
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

##################### synthetic data - ###########################

# DIPPath = '/data/Guha/GR/synthetic60FPS/H36/'
# fileList = os.listdir(DIPPath)
# print (fileList)
# for file_num in range(len(fileList)):
#     path = os.path.join(DIPPath,fileList[22])
#     with open(path, 'rb') as f:
#         print (path)
#         data_dict = pkl.load(f)
#         smpl = np.asarray(data_dict['poses'])
#         smpl_full = smpl_reduced_to_full(smpl).reshape(-1,216)
#         aa = rot_matrix_to_aa(smpl_full)
#
#         seq_len = smpl_full.shape[0]
#         print('seq len:', seq_len)
#         for seq_num in range(seq_len):
#         #     joints = smpl_full[seq_num]
#         #     aa = []
#         #     for i in range(0, 24):
#         #         ax, angle = transforms3d.axangles.mat2axangle(joints[i])
#         #         aa.append(ax * angle)
#         #     aa = np.asarray(aa)
#
#             pose = aa[seq_num]
#             # ########## NormalizeZeroMeanUnitVariance #############
#             # pose[:,0] = (pose[:,0 ] - np.mean(pose[:,0],axis=0)) / np.std(pose[:,0],axis=0)
#             # pose[:,1] = (pose[:,1] - np.mean(pose[:,1], axis=0)) / np.std(pose[:,1], axis=0)
#             # pose[:,2] = (pose[:,2] - np.mean(pose[:,2], axis=0)) / np.std(pose[:,2], axis=0)
#
#             m1.pose[:] = (pose).reshape(72)
#             m1.pose[0] = np.pi
#
#             cv2.imshow('GT', rn1.r)
#             cv2.waitKey(1)
#
#             # plt.ion()
#             # plt.clf()
#             # plt.imshow(rn1.r)
#             # plt.pause(1e-27)
#             # plt.show()