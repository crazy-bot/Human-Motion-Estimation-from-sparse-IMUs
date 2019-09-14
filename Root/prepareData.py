import numpy as np
import os
import pickle as pkl
import quaternion
import  myUtil
import  itertools
import  h5py

def getGlobalBoneOriFromPose(pose,boneName):
    pose = pose.reshape(1,15*3*3)
    fullPose = myUtil.smpl_reduced_to_full(pose).reshape(24,3,3)
    return  fullPose[myUtil.boneSMPLDict[boneName]]

def calibrateRawIMU(acc,ori,pose):

    ######### **********the order of IMUs are: [left_lower_wrist, right_lower_wrist, left_lower_leg, right_loewr_leg, head, back]
    SMPL_SENSOR = ['L_Shoulder', 'R_Shoulder', 'L_Knee', 'R_Knee', 'Head', 'Pelvis']

    # safety check if any frame has NAN Values
    frames2del = np.unique(np.where(np.isnan(ori) == True)[0])
    ori = np.delete(ori, frames2del, 0)
    acc = np.delete(acc, frames2del, 0)
    pose = np.delete(pose, frames2del, 0)
    pose = pose.reshape(-1,15,3,3)

    ############### calculation in Rotation Matrix #########################
    seq_len = len(ori)
    # head sensor for the first frame
    head_quat = ori[1, 4, :, :]
    Q = quaternion.as_rotation_matrix(np.quaternion(0.5, 0.5, 0.5, 0.5))
    #Q = np.array([[0,0,1],[0,1,0],[-1,0,0]])
    # calib: R(T_I) which is constant over the frames
    calib = np.linalg.inv(np.dot(head_quat, Q))
    #calib = np.linalg.inv(head_quat)

    # bone2sensor: R(B_S) calculated once for each sensor and remain constant over the frames as used further
    bone2sensor = {}
    for i in range(len(SMPL_SENSOR)):
        sensorid = SMPL_SENSOR[i]
        #qbone = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(myUtil.getGlobalBoneOri(sensorid)))
        qbone = myUtil.getGlobalBoneOriFromPose(pose[0],sensorid)
        qsensor = np.dot(calib, ori[1, i, :, :])
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

    # calibration of pose parameters
    # ---------------------------------------------

    #return norm_acc[:,-1,:],ori_quat[:,-1,:],pose_quat
    return norm_acc[:,0:5,:],norm_ori[:,0:5,:],pose

##################### synthetic data - ###########################

if __name__ == '__main__':
    dataset = 'AMASS_HDM05'
    train_path = '/data/Guha/GR/Dataset/{}'.format(dataset)
    test_path = '/data/Guha/GR/Dataset/Test/{}'
    valid_path = '/data/Guha/GR/Dataset/Validation/{}'

    #os.makedirs(train_path)

    FolerPath = '/data/Guha/GR/synthetic60FPS/{}'.format(dataset)
    fileList = os.listdir(FolerPath)
    train_idx = len(fileList)
    valid_idx = len(fileList) * 0.85

    # train_meta = open(train_path.format('meta.txt'),'wb')
    # test_meta = open(test_path.format('meta.txt'), 'wb')
    # valid_meta = open(valid_path.format('meta.txt'), 'wb')
    # train_size = 0
    # valid_size = 0
    # test_size = 0

    for idx,file_num in enumerate(fileList):
        path = os.path.join(FolerPath,file_num)
        #path = '/data/Guha/GR/synthetic60FPS/H36/S5_Walking.pkl'
        with open(path , 'rb') as f:
            print (path)
            data_dict = pkl.load(f,encoding='latin1')
            raw_acc_syn= np.asarray(data_dict['acc'])
            raw_ori_syn = np.asarray(data_dict['ori'])
            raw_pose_syn = np.asarray(data_dict['poses'])

            if(len(raw_acc_syn) == 0):
                continue

            # calibrate data
            acc, ori, pose = calibrateRawIMU(raw_acc_syn, raw_ori_syn, raw_pose_syn)
            # acc1, ori1, pose1 = calibrateRawIMU1(raw_acc_syn, raw_ori_syn, raw_pose_syn)
            # acc2, ori2, pose2 = calibrateRawIMU2(raw_acc_syn, raw_ori_syn, raw_pose_syn)
            if (idx <= train_idx):
                #path = train_path.format(file_num.split('.pkl')[0])
                savepath = train_path+'/'+file_num.split('.pkl')[0]
                # train_meta.write(str(len(acc))+' '+file_num.split('.pkl')[0]+'\n')
                # train_size += len(acc)
            elif (idx >= train_idx and idx < valid_idx):
                path = valid_path.format(file_num.split('.pkl')[0])
                # valid_meta.write(str(len(acc))+' '+file_num.split('.pkl')[0]+'\n')
                # valid_size += len(acc)
            else:
                path = test_path.format(file_num.split('.pkl')[0])
                # test_meta.write(str(len(acc))+' '+file_num.split('.pkl')[0]+'\n')
                # test_size += len(acc)
            np.savez_compressed(savepath,acc = acc,ori =  ori,pose = pose)
            print ('save path-- ', path)
            # break
    # train_meta.write(str(train_size)+' total_frames'+'\n')
    # valid_meta.write(str(valid_size)+' total_frames'+ '\n')
    # test_meta.write(str(test_size)+ ' total_frames'+'\n')
    # train_meta.close()
    # valid_meta.close()
    # test_meta.close()






