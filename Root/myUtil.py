import cv2
import numpy as np
from pyquaternion import Quaternion
import transforms3d
import quaternion
import torch
import matplotlib.pyplot as plt

SMPL_MAJOR_JOINTS = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
SMPL_NR_JOINTS = 24

########### these are chain of child to parent till root
parentJointsDict = {
'L_Elbow': [16,13,9,6,3,0],
'R_Elbow': [17,14,9,6,3,0],
'L_Knee': [1,0],
'R_Knee': [2,0],
'Head': [12,9,6,3,0],
'Pelvis' : []
}

############ SMPL bone indexes #############
boneSMPLDict = { 'Pelvis' : 0,
                'L_Hip': 1,
                'R_Hip': 2,
                'Spine1': 3,
                'L_Knee': 4,
                'R_Knee': 5,
                'Spine2': 6,
                'L_Ankle': 7,
                'R_Ankle': 8,
                'Spine3': 9,
                'L_Foot': 10,
                'R_Foot':11,
                'Neck': 12,
                'L_Collar': 13,
                'R_Collar':14,
                'Head': 15,
                'L_Shoulder': 16,
                'R_Shoulder': 17,
                'L_Elbow': 18,
                'R_Elbow': 19,
                'L_Wrist': 20,
                'R_Wrist': 21,
                'L_Hand': 22,
                'R_Hand': 23
                 }
################# IPOSE parameters taken from DIP_IMU file ##################
Ipose = np.asarray([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.40190000e-02,
                    -5.65980000e-02, -6.00000000e-02, 4.01920000e-03, 1.07109000e-01,
                    4.00000000e-02, -6.85148000e-02, 2.97456624e-02, 0.00000000e+00,
                    -1.50640000e-02, 1.22855000e-01, -2.80000000e-03, -2.51200000e-04,
                    -7.49550000e-02, 2.80000000e-03, -1.97083023e-02, -5.90451714e-02,
                    0.00000000e+00, -3.69410000e-02, -1.39870000e-02, 1.09700000e-03,
                    3.08240000e-02, 1.10824000e-01, 5.58300000e-02, 3.68217919e-02,
                    -9.79798425e-03, 0.00000000e+00, 7.38820000e-02, 8.71628260e-02,
                    1.15933226e-01, -1.36454340e-02, 7.27977859e-02, -2.04008074e-01,
                    2.75226449e-02, 3.74526146e-02, -3.26716395e-02, 7.95110800e-02,
                    1.55932400e-02, -3.61916400e-01, 7.95110800e-02, -1.55932400e-02,
                    3.61916400e-01, 4.14048214e-02, -5.75496269e-03, 6.12744933e-02,
                    -1.08706800e-01, -1.39227600e-02, -1.10823788e+00, 7.96932000e-02,
                    2.02324166e-01, 1.06021472e+00, 1.14999360e-01, -1.25600000e-01,
                    -1.25600000e-01, 5.21993600e-02, 1.25600000e-01, 1.25600000e-01,
                    1.34247560e-01, -9.28749200e-02, -8.79514000e-02, 1.31183097e-02,
                    4.85928009e-02, 6.31077200e-02, -2.00966541e-01, -3.42684870e-02,
                    -1.76926440e-01, -1.28807464e-01, 1.02772092e-01, 2.61631080e-01])

def smpl_reduced_to_full(smpl_reduced):
    """
    Converts an np array that uses the reduced smpl representation into the full representation by filling in
    the identity rotation for the missing joints. Can handle either rotation input (dof = 9) or quaternion input
    (dof = 4).
    :param smpl_full: An np array of shape (seq_length, n_joints_reduced*dof)
    :return: An np array of shape (seq_length, 24*dof)
    """
    dof = smpl_reduced.shape[1] // len(SMPL_MAJOR_JOINTS)
    assert dof == 9 or dof == 4
    seq_length = smpl_reduced.shape[0]
    smpl_full = np.zeros([seq_length, SMPL_NR_JOINTS * dof])
    for idx in range(SMPL_NR_JOINTS):
        if idx in SMPL_MAJOR_JOINTS:
            red_idx = SMPL_MAJOR_JOINTS.index(idx)
            smpl_full[:, idx * dof:(idx + 1) * dof] = smpl_reduced[:, red_idx * dof:(red_idx + 1) * dof]
        else:
            if dof == 9:
                identity = np.repeat(np.eye(3, 3)[np.newaxis, ...], seq_length, axis=0)
            else:
                identity = np.concatenate([np.array([[1.0, 0.0, 0.0, 0.0]])] * seq_length, axis=0)
            smpl_full[:, idx * dof:(idx + 1) * dof] = np.reshape(identity, [-1, dof])
    return smpl_full

# Input frames are in shape (no_of_frames, no_of_joints * 4)
def quat_to_aa_representation(frames,no_of_joints):
    aa_angles = []
    out = frames.reshape(-1, no_of_joints, 4)
    for seq in range(out.shape[0]):
        for idx in range(no_of_joints):
            ax, angle = transforms3d.quaternions.quat2axangle(out[seq,idx])
            aa_angles.extend(ax * angle)
            # aa = quaternion.as_rotation_vector(out[seq,idx])
            # aa_angles.extend(aa)00000
    aa_angles = np.asarray(aa_angles).reshape(-1, no_of_joints*3)

    return aa_angles

def rot_matrix_to_aa(data):
    """
    Converts the orientation data given in rotation matrices to angle axis representation. `data` is expected in format
    (seq_length, n*9). Returns an array of shape (seq_length, n*3).
    """
    seq_length, n_joints = data.shape[0], data.shape[1] // 9
    data_r = np.reshape(data, [seq_length, n_joints, 3, 3])
    data_c = np.zeros([seq_length, n_joints, 3])
    for i in range(seq_length):
        for j in range(n_joints):
            data_c[i, j] = np.ravel(cv2.Rodrigues(data_r[i, j])[0])
    return np.reshape(data_c, [seq_length, n_joints * 3])

# this method is called during calibration
def getGlobalBoneOriFromPose(pose,boneNameList):
    pose = pose.reshape(1,15*3*3)
    fullPose = smpl_reduced_to_full(pose).reshape(24,3,3)
    bone2root = []
    for boneName in boneNameList:
        temp = np.identity(3)
        for child in parentJointsDict[boneName]:
            temp = np.dot(temp, fullPose[child,:,:])
        bone2root.append(temp)
    rrt = np.linalg.inv(bone2root[4])
    bone2smpl = np.array([np.dot(bone2root[k],rrt) for k in range(6)])
    return  bone2smpl

# def getGlobalBoneOriFromPose(pose,boneName):
#     pose = pose.reshape(1,15*3*3)
#     fullPose = smpl_reduced_to_full(pose).reshape(24,3,3)
#     return fullPose[boneSMPLDict[boneName]]

def getGlobalBoneOri(boneName):

    # return axis-angle of the bone

    # Ipose from DIP file. joints are in SMPL order
    return Ipose.reshape(24,3)[boneSMPLDict[boneName]]


def qeuler(q, order, epsilon=0):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == 'zxy':
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == 'yxz':
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise
    return torch.stack((x, y, z), dim=1).view(original_shape)

def rot_matrix_to_aa(data):
    """
    Converts the orientation data given in rotation matrices to angle axis representation. `data` is expected in format
    (seq_length, n*9). Returns an array of shape (seq_length, n*3).
    """
    seq_length, n_joints = data.shape[0], data.shape[1]//9
    data_r = np.reshape(data, [seq_length, n_joints, 3, 3])
    data_c = np.zeros([seq_length, n_joints, 3])
    for i in range(seq_length):
        for j in range(n_joints):
            data_c[i, j] = np.ravel(cv2.Rodrigues(data_r[i, j])[0])
    return np.reshape(data_c, [seq_length, n_joints*3])

def split():
    import shutil
    import os
    listofPath = []
    datapath = '/data/Guha/GR/Dataset'
    dataset = ['AMASS_ACCAD','AMASS_BioMotion','AMASS_CMU_Kitchen','AMASS_Eyes','AMASS_HDM05','AMASS_MIXAMO','AMASS_SSM','AMASS_Transition','CMU','H36','HEva','JointLimit']
    for d in dataset:
        folderpath = os.path.join(datapath,d)
        for f in os.listdir(folderpath):
            listofPath.append(os.path.join(folderpath,f))

    np.random.shuffle(np.array(listofPath))
    length = len(listofPath)
    for i in range(length):
        if (i <= length*0.7):
            shutil.copy2(listofPath[i] , '/data/Guha/GR/Dataset/Train/')
        elif (i <= length*0.85):
            shutil.copy2(listofPath[i] , '/data/Guha/GR/Dataset/Validation/')
        else:
            shutil.copy2(listofPath[i] , '/data/Guha/GR/Dataset/Test/')
def plotGraph():
    with open('/data/Guha/GR/Output/loss_model9') as model9, open('/data/Guha/GR/Output/loss_model12') as model12:
        lines_9 = model9.readlines()
        lines_12 = model12.readlines()
        loss_9 = []
        loss_12 = []
        for i in range(len(lines_9)):
            loss_9.append(float(lines_9[i].split(' ')[-1]))
            loss_12.append(float(lines_12[i].split(' ')[-1]))

    loss_9 = np.array(loss_9)
    print(np.mean(loss_9,axis=0))
    loss_12 = np.array(loss_12)
    plt.plot(np.arange(len(loss_9)),loss_9, c='b')
    plt.plot(np.arange(len(lines_9)),loss_12,c='g')
    plt.show()

def parseJson():
    import json
    import numpy as np
    SMPL_SENSOR = ['L_Shoulder', 'R_Shoulder', 'L_Knee', 'R_Knee', 'Head', 'Pelvis']
    sensorDict = {'L_Shoulder': 30, 'R_Shoulder': 25, 'L_knee': 10, 'R_knee': 12, 'Head': 20, 'Pelvis': 15}
    sensorIds = ['30', '25', '10', '12', '20', '15']
    rawOri = []
    with open("/data/Guha/GR/Dataset/DFKI/walking_bending.json", "r") as read_file:
        lines = read_file.readlines()
        for line in lines:
            jsondict = json.loads(line)
            imus = jsondict['data']['imu']

            # loop through 6 sensor ids #
            ori_t = []
            for id in sensorIds:
                if (id in imus):
                    ori_t.append(imus[id]['orientation'])
                else:
                    ori_t.append([0, 0, 0, 0])

            rawOri.append(ori_t)
        rawOri = np.array(rawOri)

        ## impute missing sensor values with previous timestep ##
        while (np.any(~np.any(rawOri, axis=2))):
            m_rows, m_cols = np.where(~np.any(rawOri, axis=2))
            rawOri[m_rows, m_cols] = rawOri[(m_rows - 1), m_cols]

        return rawOri
