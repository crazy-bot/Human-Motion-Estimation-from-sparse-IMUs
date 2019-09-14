import  os
import numpy as np
import myUtil
import transforms3d
import quaternion


x_labels = ['AMASS_ACCAD', 'AMASS_BioMotion', 'AMASS_CMU_Kitchen', 'AMASS_Eyes', 'AMASS_MIXAMO',
                 'AMASS_SSM', 'AMASS_Transition', 'CMU', 'H36','AMASS_HDM05', 'HEva', 'JointLimit','DIP_IMU']

trainset = ['AMASS_ACCAD', 'AMASS_BioMotion', 'AMASS_CMU_Kitchen', 'AMASS_Eyes', 'AMASS_MIXAMO',
                 'AMASS_SSM', 'AMASS_Transition', 'CMU', 'H36','AMASS_HDM05', 'HEva', 'JointLimit','DIP_IMU']
# trainset = ['AMASS_ACCAD', 'AMASS_BioMotion', 'AMASS_CMU_Kitchen', 'AMASS_Eyes','CMU', 'H36','AMASS_HDM05','HEva']
#
# trainset = ['HEva']

# trainset = ['DIP_IMU']


SMPL_MAJOR_JOINTS = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
SMPL_NR_JOINTS = 24

boneSMPLDict = {'L_Knee': 4,
                'R_Knee': 5,
                'Head': 15,
                'L_Shoulder': 16,
                'R_Shoulder': 17
                 }
jointsToPlot = [3,4,10,11,12]

#datapath = '/data/Guha/GR/synthetic60FPS/'
datapath ='/data/Guha/GR/Dataset/'


poses = []

for t in trainset:
    path = datapath+t
    dset_joints = []
    for f in os.listdir(path):
        filepath = os.path.join(path,f)
        data_dict = np.load(filepath, encoding='latin1')
        #sample_pose = myUtil.smpl_reduced_to_full(np.asarray(data_dict['poses']).reshape(-1,15*3*3)).reshape(-1,24,3,3)
        sample_pose = np.asarray(data_dict['pose']).reshape(-1,15,3,3)[0,jointsToPlot,:,:]
        if (len(data_dict['ori']) == 0):
            continue
        #sample_ori = np.asarray(data_dict['ori']).reshape(-1, 5, 3, 3)[0, :, :, :]

        #pose_x,pose_y,pose_z = transforms3d.euler.mat2euler(sample_pose[0,0,:,:], axes='sxyz')

        #pose_x,pose_y,pose_z = transforms3d.euler.mat2euler(quaternion.as_rotation_matrix(quaternion.from_rotation_vector(sample_pose[0,15,:])))
        #ori_x,ori_y,ori_z = transforms3d.euler.mat2euler(sample_ori[0,2,:,:], axes='sxyz')
        euler_joints = [list(transforms3d.euler.mat2euler(sample_pose[i, :, :], axes='sxyz')) for i in range(5)]
        dset_joints.append(euler_joints)

    poses.append(np.asarray(dset_joints))
############### for raw data of dip imu #####################
# trainset = ['s_01', 's_02','s_03', 's_04', 's_05','s_06','s_07','s_08','s_09','s_10']
# datapath = '/data/Guha/GR/DIPIMUandOthers/DIP_IMU_and_Others/DIP_IMU/'
# jointsToPlot = [4,5,15,16,17]
# dip_sensor_idx = [7, 8, 11, 12, 0, 2]
# dset_joints = []
# for t in trainset:
#     path = datapath+t
#     for f in os.listdir(path):
#         filepath = os.path.join(path,f)
#         data_dict = np.load(filepath, encoding='latin1')
#         if (len(data_dict['imu']) == 0):
#             continue
#         imu_ori = data_dict['imu'][:, :, 0:9]
#         imu_ori = np.asarray([imu_ori[:, k, :] for k in dip_sensor_idx])
#         sample_ori = imu_ori.reshape(-1, 6, 3, 3)[0,:,:,:]
#         print('files--',path,f)
#         # sample_pose = myUtil.smpl_reduced_to_full(np.asarray(data_dict['poses']).reshape(-1,15*3*3)).reshape(-1,24,3,3)
#         #sample_pose = np.asarray(data_dict['gt']).reshape(-1,24,3)[0,jointsToPlot,:]
#
#         if(len(sample_ori)==0):
#             continue
#         #euler_joints = [list(transforms3d.euler.mat2euler(quaternion.as_rotation_matrix(quaternion.from_rotation_vector(sample_ori[i,:,:])))) for i in range(6)]
#         euler_joints = [list(transforms3d.euler.mat2euler(sample_ori[i, :, :], axes='sxyz')) for i in range(6)]
#         dset_joints.append(euler_joints)
#
# poses.append(np.asarray(dset_joints))

euler_dict = {0:'x axis',1:'y axis',2:'z axis'}
oriDict = ['L_Shoulder', 'R_Shoulder', 'L_Knee', 'R_Knee', 'Head']
poseDict = ['L_Knee','R_Knee','Head','L_Shoulder','R_Shoulder']
poses = np.asarray(poses)
# plotpose = [poses[i] for i in range(8)]
# abc = np.asarray(plotpose)
import matplotlib.pyplot as plt

for i,key in enumerate(poseDict):
    for j in range(3):
        title = key + ' ' + euler_dict[j]
        plt.figure(title,figsize = [10, 10])
        temp = []
        for d in range(len(poses)):
            temp.append(poses[d][:,i,j])
        plt.boxplot(temp)
        plt.xticks(np.arange(len(x_labels)),x_labels,rotation=50)
        plt.title(title)
        #plt.show()
        plt.savefig('/data/Guha/GR/Output/graphs/pose/'+title+'.png')








