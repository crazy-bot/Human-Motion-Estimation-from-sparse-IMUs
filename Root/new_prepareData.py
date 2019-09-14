import numpy as np
import os
import pickle as pkl
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
    pose = pose.reshape(-1, 15, 3, 3)
    seq_len = len(ori)

    ############### calculation in Rotation Matrix #########################
    # 1. calib: R(T_I) = inv( R(I_S) * R(S_B) * R(B_T) ) [sensor 4: head for 1st frame]. rti is constant for all frames
    ris_head = ori[0, 4, :, :]
    rsb_head = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    rbt_head = np.identity(3)
    rti = np.linalg.inv(np.linalg.multi_dot([ris_head, rsb_head, rbt_head]))
    # 2. R(T_S) = R(T_I) * R(I_S) for all sensors for all frames
    seq_len = len(ori)
    rts = np.asarray(
        [np.dot(rti, ori[k, j, :, :]) for
         k, j in itertools.product(range(seq_len), range(6))]
    ).reshape(-1,6,3,3)
    # 3. calculate R(B_T) for all 6 joints
    rbt = myUtil.getGlobalBoneOriFromPose(pose[0], SMPL_SENSOR)

    # 4. calculate R(B_S) = R(B_T) * R(T_S) for the first frame which will be constant across all frames
    rbs = np.array([np.dot(rbt[k], rts[0, k]) for k in range(6)])

    # 5. calculate bone2smpl (all frames) : R(T_B) = R(T_S) * inv(R(B_S))
    rtb = np.asarray(
        [np.dot(rts[k, j, :, :], rbs[j]) for
         k, j in itertools.product(range(seq_len), range(6))]
    )
    calibrated_ori = rtb.reshape(-1, 6, 3, 3)

    # 6. normalize respect to root
    root_inv = np.linalg.inv(calibrated_ori[:, 5, :, :])
    # root_inv = np.transpose(calibrated_ori[:, 5], [0, 2, 1])
    norm_ori = np.asarray(
        [np.matmul(root_inv[k], calibrated_ori[k, j, :, :]) for k, j in itertools.product(range(seq_len), range(6))])
    norm_ori = norm_ori.reshape(-1, 6, 3, 3)

    #return norm_acc[:,-1,:],ori_quat[:,-1,:],pose_quat
    return norm_ori[:,0:5,:],pose

##################### synthetic data - ###########################

if __name__ == '__main__':
    trainset = ['AMASS_MIXAMO','AMASS_SSM', 'AMASS_Transition', 'CMU','AMASS_HDM05', 'HEva', 'JointLimit']
    for dset in trainset:
        dataset = dset
        train_path = '/data/Guha/GR/Dataset/{}'.format(dataset)
        test_path = '/data/Guha/GR/Dataset/Test/{}'
        valid_path = '/data/Guha/GR/Dataset/Validation/{}'

        if(not os.path.exists(train_path)):
            os.makedirs(train_path)

        FolerPath = '/data/Guha/GR/synthetic60FPS/{}'.format(dataset)
        fileList = os.listdir(FolerPath)
        train_idx = len(fileList)
        valid_idx = len(fileList) * 0.85

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
                ori, pose = calibrateRawIMU(raw_acc_syn, raw_ori_syn, raw_pose_syn)
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
                np.savez_compressed(savepath,ori =  ori,pose = pose)
                print ('save path-- ', path)
                # break
        # train_meta.write(str(train_size)+' total_frames'+'\n')
        # valid_meta.write(str(valid_size)+' total_frames'+ '\n')
        # test_meta.write(str(test_size)+ ' total_frames'+'\n')
        # train_meta.close()
        # valid_meta.close()
        # test_meta.close()






