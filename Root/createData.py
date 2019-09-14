import numpy as np
import os
import transforms3d
import cv2
import pickle as pkl
import matplotlib.pyplot as plt
import copy
import pdb
import quaternion
import  myUtil
import  itertools
import  h5py


def calibrateRawIMU(acc,ori,pose):

    ######### **********the order of IMUs are: [left_lower_wrist, right_lower_wrist, left_lower_leg, right_loewr_leg, head, back]
    #SMPL_SENSOR = ['L_Shoulder', 'R_Shoulder', 'L_Knee', 'R_Knee', 'Head', 'Pelvis']
    SMPL_SENSOR = ['L_Elbow', 'R_Elbow', 'L_Knee', 'R_Knee', 'Head', 'Pelvis']

    # safety check if any frame has NAN Values
    frames2del = np.unique(np.where(np.isnan(ori) == True)[0])
    ori = np.delete(ori, frames2del, 0)
    acc = np.delete(acc, frames2del, 0)
    pose = np.delete(pose, frames2del, 0)
    pose = pose.reshape(-1,24,3)
    seq_len = len(ori)

    # calibration of pose parameters
    # ---------------------------------------------
    SMPL_MAJOR_JOINTS = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    pose = pose[:, SMPL_MAJOR_JOINTS, :]
    qs = quaternion.from_rotation_vector(pose)
    pose_rot = np.reshape(quaternion.as_rotation_matrix(qs), [seq_len, 15, 9])
    pose = np.reshape(pose_rot, [seq_len, 9 * 15])

    ############### calculation in Rotation Matrix #########################

    # head sensor for the first frame
    head_quat = ori[0, 4, :, :]
    Q = quaternion.as_rotation_matrix(np.quaternion(0.5, 0.5, 0.5, 0.5))
    # calib: R(T_I) which is constant over the frames
    calib = np.linalg.inv(np.dot(head_quat, Q))
    #calib = np.linalg.inv(head_quat)

    # bone2sensor: R(B_S) calculated once for each sensor and remain constant over the frames as used further
    bone2sensor = {}
    for i in range(len(SMPL_SENSOR)):
        sensorid = SMPL_SENSOR[i]
        qbone = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(myUtil.getGlobalBoneOri(sensorid)))
        #qbone = myUtil.getGlobalBoneOriFromPose(pose_rot, sensorid)
        qsensor = np.dot(calib, ori[0, i, :, :])
        boneQuat = np.dot(np.linalg.inv(qsensor), qbone)
        bone2sensor[i] = boneQuat

    # calibrated_ori: R(T_B) calculated as calib * sensor data(changes every frame) * bone2sensor(corresponding sensor)
    calibrated_ori = np.asarray(
        [np.linalg.multi_dot([calib, ori[k, j, :, :], bone2sensor[j]]) for
         k, j in itertools.product(range(seq_len), range(6))]
    )
    calibrated_ori = calibrated_ori.reshape(-1, 6, 3, 3)
    root_inv = np.linalg.inv(calibrated_ori[:, 5, :, :])
    # root_inv = np.transpose(calibrated_ori[:, 5], [0, 2, 1])
    norm_ori = np.asarray(
        [np.matmul(root_inv[k], calibrated_ori[k, j, :, :]) for k, j in itertools.product(range(seq_len), range(6))])
    norm_ori = norm_ori.reshape(-1, 6, 3, 3)

    # calibration of acceleration
    acc_1 = np.asarray(
        [np.dot(ori[k, j, :], acc[k, j, :]) for k, j in
         itertools.product(range(seq_len), range(6))])
    acc_1 = acc_1.reshape(-1, 6, 3)
    calib_acc = np.asarray([np.dot(calib, acc_1[k, j, :]) for k, j in
                            itertools.product(range(seq_len), range(6))])
    calib_acc = calib_acc.reshape(-1, 6, 3)
    norm_acc = np.asarray(
        [np.dot(root_inv[k], (calib_acc[k, j, :] - calib_acc[k, 5, :])) for k, j in
         itertools.product(range(seq_len), range(6))])
    norm_acc = norm_acc.reshape(-1, 6, 3)


    #return norm_acc[:,-1,:],ori_quat[:,-1,:],pose_quat
    return norm_acc[:,0:5,:],norm_ori[:,0:5,:],pose

if __name__ == '__main__':
    ######################## read DataFile - DIP_IMU ###########################
    DIPPath = '/data/Guha/GR/Dataset/DIP_IMU2/{}'
    imu_order = ['head', 'spine2', 'belly', 'lchest', 'rchest', 'lshoulder', 'rshoulder', 'lelbow', 'relbow', 'lhip',
                 'rhip', 'lknee', 'rknee', 'lwrist', 'rwrist', 'lankle', 'rankle']
    SENSORS = ['lelbow', 'relbow', 'lknee', 'rknee', 'head', 'belly']
    #SENSORS = ['lwrist', 'rwrist', 'lknee', 'rknee', 'head', 'belly']
    sensor_idx = [7, 8, 11, 12, 0, 2]
    FolerPath = '/data/Guha/GR/DIPIMUandOthers/DIP_IMU_and_Others/DIP_IMU/s_{}'
    for sub in range(10,11):
        path = FolerPath.format(sub)
        for f in os.listdir(path):
            with open(os.path.join(path,f),'rb') as file:
                data_dict = pkl.load(file,encoding='latin1')
                if (len(data_dict['imu']) == 0):
                    continue
                imu_ori = data_dict['imu'][:, :, 0:9]
                #frames2del = np.unique(np.where(np.isnan(imu_ori) == True)[0])
                #imu_ori = np.delete(imu_ori, frames2del, 0)
                imu_ori = np.asarray([imu_ori[:, k, :] for k in sensor_idx])
                raw_ori_dip = imu_ori.reshape(-1, 6, 3, 3)

                imu_acc = data_dict['imu'][:, :, 9:12]
                #imu_acc = np.delete(imu_acc, frames2del, 0)
                imu_acc = np.asarray([imu_acc[:, k, :] for k in sensor_idx])
                raw_acc_dip = imu_acc.reshape(-1, 6, 3)
                raw_pose_dip = data_dict['gt']

                # calibrate data
                acc_dip, ori_dip, pose_dip = calibrateRawIMU(raw_acc_dip, raw_ori_dip, raw_pose_dip)
                savepath = DIPPath.format(path.split('/')[-1]+'_'+f.split('.pkl')[0])


            #content = {"acc": acc,"ori": ori,"pose":pose}
            np.savez_compressed(savepath,acc = acc_dip,ori =  ori_dip,pose = pose_dip)
            print ('save path-- ', f)
            #break






