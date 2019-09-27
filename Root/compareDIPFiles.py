import sys
sys.path.insert(0, '/data/Guha/GR/code/GR19/smpl/smpl_webuser')
import numpy as np
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from serialization import load_model
import os
import myUtil
import cv2
import matplotlib.pyplot as plt
import itertools

####################################3 Load SMPL model (here we load the female model) ######################################
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
####################################### finish of adapting SMPL python initialization ########################

####################################### read the two files #############################

############### raw file
DIPPath = '/data/Guha/GR/DIPIMUandOthers/DIP_IMU_and_Others/DIP_IMU/s_10'
imu_order = ['head', 'spine2', 'belly', 'lchest', 'rchest', 'lshoulder', 'rshoulder', 'lelbow', 'relbow', 'lhip', 'rhip', 'lknee', 'rknee', 'lwrist', 'lwrist', 'lankle', 'rankle']
SENSORS = [ 'lknee', 'rknee','lelbow', 'relbow', 'head','belly']
SMPL_SENSOR = ['L_Knee','R_Knee','L_Elbow','R_Elbow','Head','Pelvis']
sensor_idx = [11,12,7,8,0,2]


############# calirated file
SMPL_MAJOR_JOINTS = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
file_path1 = '/data/Guha/GR/code/dip18/train_and_eval/data/dipIMU/imu_own_validation.npz'
file_path2 = '/data/Guha/GR/DIPIMUandOthers/DIP_IMU_and_Others/DIP_IMU/s_10'
fileList = os.listdir(file_path2)
dippath = os.path.join(file_path2,fileList[4])
with open(file_path1, 'rb') as file1, open(dippath, 'rb') as file2:
    data_dict_nn = dict(np.load(file1))
    smpl_nn = data_dict_nn['smpl_pose'][1]
    smpl_nn_full = myUtil.smpl_reduced_to_full(smpl_nn)
    aa_nn = myUtil.rot_matrix_to_aa(smpl_nn_full)

    data_dict_imu = dict(np.load(file2))
    smpl_imu = data_dict_imu['gt']

    seq_len = aa_nn.shape[0]
    print (aa_nn - smpl_imu)

    for seq_num in range(seq_len):
        pose1 = aa_nn[seq_num]
        pose2 = smpl_imu[seq_num]

        m1.pose[:] = (pose1).reshape(72)
        m1.pose[0] = np.pi

        m2.pose[:] = (pose2).reshape(72)
        m2.pose[0] = np.pi

        cv2.imshow('GT', rn1.r)
        cv2.imshow('Prediction', rn2.r)
        #Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # nn_img = rn1.r * 255
        # imu_img = rn2.r * 255
        # cv2.imwrite('/data/Guha/GR/Output/folder1/' + str(seq_num) + '.png', nn_img)
        # cv2.imwrite('/data/Guha/GR/Output/folder2/' + str(seq_num) + '.png', imu_img)


    ########################## our calibration code #######################

    imu_ori = data_dict_imu['imu'][:, :, 0:9]
    frames2del = np.unique(np.where(np.isnan(imu_ori) == True)[0])
    imu_ori = np.delete(imu_ori, frames2del, 0)
    imu_ori = np.asarray([imu_ori[:, k, :] for k in sensor_idx])
    imu_ori = imu_ori.reshape(-1, 6, 3, 3)

    imu_acc = data_dict_imu['imu'][:, :, 9:12]
    imu_acc = np.delete(imu_acc, frames2del, 0)
    imu_acc = np.asarray([imu_acc[:, k, :] for k in sensor_idx])
    imu_acc = imu_acc.reshape(-1, 6, 3)

    seq_len = imu_acc.shape[0]
    print('seq len:', seq_len)

    head_0_frame = imu_ori[0, 0, :].reshape(3, 3)
    norm_ori_1 = np.asarray(
        [np.dot(np.linalg.inv(head_0_frame), imu_ori[k, j, :].reshape(3, 3)) for k, j in
         itertools.product(range(seq_len), range(6))])
    norm_ori_1 = norm_ori_1.reshape(-1, 6, 3, 3)

    R_TS_0 = norm_ori_1[0, 0, :, :]
    R_TS_1 = norm_ori_1[0, 1, :, :]
    # inv_R_TB_0 = np.asarray([[0,-1,0],[1,0,0],[0,0,1]])
    inv_R_TB_0 = np.asarray([[-0.0254474, -0.98612685, 0.16403132],
                             [0.99638575, -0.01171777, 0.08413162],
                             [-0.08104237, 0.1655794, 0.98286092]])
    # inv_R_TB_1 = np.asarray([[0,1,0],[-1,0,0],[0,0,1]])
    inv_R_TB_1 = np.asarray([[-0.01307566, 0.97978612, - 0.1996201],
                             [-0.97720228, 0.02978678, 0.21021048],
                             [0.21190735,
                              0.19781786,
                              0.95705975]])

    R_BS_0 = np.dot(inv_R_TB_0, R_TS_0)
    R_BS_1 = np.dot(inv_R_TB_1, R_TS_1)

    bone2S = norm_ori_1[0, :, :, :]

    bone2S[0, :, :] = R_BS_0
    bone2S[1, :, :] = R_BS_1

    # bone_ori = np.asarray([np.dot(norm_ori_1[k, j, :, :], np.linalg.inv(bone2S[j])) for k,j in itertools.product(range(seq_len),range(6))])

    bone_ori = np.asarray([np.dot(bone2S[j], norm_ori_1[k, j, :, :]) for k, j in
                           itertools.product(range(seq_len), range(6))])

    bone_ori = bone_ori.reshape(-1, 6, 3, 3)

    norm_ori_2 = np.asarray(
        [np.dot(np.linalg.inv(bone_ori[i, 5, :, :]), bone_ori[i, j, :, :]) for i, j in
         itertools.product(range(seq_len), range(6))])

    norm_ori_2 = norm_ori_2.reshape(-1, 6, 3, 3)

    # acceleration
    norm_acc_1 = np.asarray(
        [np.dot(norm_ori_1[k, j, :].reshape(3, 3), imu_acc[k, j, :]) for k, j in
         itertools.product(range(seq_len), range(6))])
    norm_acc_1 = norm_acc_1.reshape(-1, 6, 3)

    norm_acc_2 = np.asarray(
        [np.dot(np.linalg.inv(norm_ori_2[k, 5, :, :]), (norm_acc_1[k, j, :] - norm_acc_1[k, 5, :])) for k, j in
         itertools.product(range(seq_len), range(6))])
    norm_acc_2 = norm_acc_2.reshape(-1, 6, 3)

    ori1 = data_dict_nn['orientation'][1]
    ori1 = np.delete(ori1, frames2del, 0)
    ori1 = ori1[:, :].reshape(-1, 5, 9)
    diff_ori = norm_ori_2[:, 0:5, :, :].reshape(-1, 5, 9) - ori1

    acc1 = data_dict_nn['acceleration'][1]
    acc1 = np.delete(acc1, frames2del, 0)
    acc1 = acc1[:, :].reshape(-1, 5, 3)
    diff_acc = norm_acc_2[:, 0:5, :].reshape(-1, 5, 3) - acc1
    for i in range(5):
        plt.figure('Difference of Orientation: {}'.format(SENSORS[i]))
        plt.title('Difference of Orientation: {} between DIP_IMU_nn & our calibration'.format(SENSORS[i]))
        plt.boxplot(diff_ori[:, i, :])

        plt.figure('Difference of Acceleration: {}'.format(SENSORS[i]))
        plt.title('Difference of Acceleration: {} between DIP_IMU_nn & our calibration'.format(SENSORS[i]))
        plt.boxplot(diff_acc[:, i, :])
        plt.show()


