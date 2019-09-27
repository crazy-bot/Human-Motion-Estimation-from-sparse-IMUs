import  numpy as np
import glob
import os
from pyquaternion import Quaternion
import itertools
import  transforms3d
import  torch
import Config as cfg

class IMUDataset():

    def __init__(self,rootdir):

        self.total_frames = 0
        # self.files = glob.glob(rootdir+'/*.npz')

        for f in self.files:
            data_dict = np.load(f, encoding='latin1')
            sample_acc = data_dict['acc']
            self.total_frames += len(sample_acc)

        # files are sorted in descending order of size
        #self.files = self.get_files_by_file_size(rootdir)
        self.testset = ['AMASS_HDM05', 'HEva', 'JointLimit']

        ################# To normalize acceleration ###################
        imu_dip = dict(np.load('/data/Guha/GR/code/dip18/train_and_eval/data/dipIMU/imu_own_validation.npz', encoding='latin1'))
        data_stats = imu_dip.get('statistics').tolist()
        self.acc_stats = data_stats['acceleration']

    def __init__(self,datapath, trainset):

        self.loadfiles(datapath, trainset)
        self.total_frames = 0
        for f in self.files:
            data_dict = np.load(f, encoding='latin1')
            sample_ori = data_dict['ori']
            self.total_frames += len(sample_ori)

    def get_files_by_file_size(self, dirname, reverse=True):
        """ Return list of file paths in directory sorted by file size """

        # Get list of files
        filepaths = []
        for basename in os.listdir(dirname):
            filename = os.path.join(dirname, basename)
            if os.path.isfile(filename):
                filepaths.append(filename)

        # Re-populate list with filename, size tuples
        for i in range(len(filepaths)):
            filepaths[i] = (filepaths[i], os.path.getsize(filepaths[i]))

        # Sort list by file size
        # If reverse=True sort from largest to smallest
        # If reverse=False sort from smallest to largest
        filepaths.sort(key=lambda filename: filename[1], reverse=reverse)

        # Re-populate list with just filenames
        for i in range(len(filepaths)):
            filepaths[i] = filepaths[i][0]

        return filepaths[1:]


############# prepare batch in quaternion #################
    def prepareBatch_quat2(self):
        self.input = []
        self.target = []
        # act_idx: index of activities of one batch
        act_idx = np.random.choice(len(self.files),cfg.batch_len)
        for idx in act_idx:
            data_dict =  np.load(self.files[idx], encoding='latin1')
            seq_len = data_dict['pose'].shape[0]
            # when length of activity is more than 200 we select 200 frames starting from any position randomly
            if (seq_len > cfg.seq_len):
                # get any random sequence of frames of size cfg.seq_len of the activity
                start_idx = np.random.choice(data_dict['pose'].shape[0] - cfg.seq_len)
                sample_pose = data_dict['pose'][start_idx: start_idx + cfg.seq_len]
                sample_ori = data_dict['ori'][start_idx: start_idx + cfg.seq_len]
                sample_acc = data_dict['acc'][start_idx: start_idx + cfg.seq_len]
            # when length of activity is less than 200 mask it and make size of 200
            else:
                identity_pose = np.repeat(np.eye(3, 3)[np.newaxis, ...], 15, axis=0)
                identity_ori = np.repeat(np.eye(3, 3)[np.newaxis, ...], 5, axis=0)
                sample_pose = np.array([identity_pose]*cfg.seq_len)
                sample_ori = np.array([identity_ori]*cfg.seq_len)
                sample_acc = np.zeros((cfg.seq_len,5,3))
                sample_pose[:seq_len] = data_dict['pose']
                sample_ori[:seq_len] = data_dict['ori']
                sample_acc[:seq_len] = data_dict['acc']

            sample_pose = sample_pose.reshape(-1,15,3,3)

            #################### convert all roation matrices to quaternion ###############
            ori_quat = np.asarray([Quaternion(matrix=sample_ori[k,j,:,:]).elements for k,j in itertools.product(range(sample_ori.shape[0]), range(5))])
            ori_quat = ori_quat.reshape(-1,5*4)
            pose_quat = np.asarray([Quaternion(matrix=sample_pose[k, j, :, :]).elements for k, j in
                                   itertools.product(range(sample_pose.shape[0]), range(15))])
            pose_quat = pose_quat.reshape(-1,15*4)

            ################ normalize acceleration ###################
            # sample_acc[:,:,0] = (sample_acc[:,:,0] - np.min(sample_acc[:,:,0])) / (np.max(sample_acc[:,:,0])-np.min(sample_acc[:,:,0]))
            # sample_acc[:, :, 1] = (sample_acc[:,:,1] - np.min(sample_acc[:,:,1])) / (np.max(sample_acc[:,:,1])-np.min(sample_acc[:,:,1]))
            # sample_acc[:, :, 2] = (sample_acc[:,:,2] - np.min(sample_acc[:,:,2])) / (np.max(sample_acc[:,:,2])-np.min(sample_acc[:,:,2]))

            sample_acc = sample_acc.reshape(-1, 5 * 3)
            sample_acc = (sample_acc - self.acc_stats['mean_channel']) / self.acc_stats['std_channel']


            #concat = np.concatenate((ori_quat,sample_acc),axis=1)

            self.input.append(ori_quat)
            self.target.append(pose_quat)

        self.input = np.asarray(self.input)
        self.target = np.asarray(self.target)

############### prepare batch in euler ################
    def prepareBatch_euler2(self):

        self.input = []
        self.target = []
        # act_idx: index of activities of one batch
        act_idx = np.random.choice(len(self.files),cfg.batch_len)
        for idx in act_idx:
            data_dict =  np.load(self.files[idx], encoding='latin1')
            seq_len = data_dict['pose'].shape[0]

            if(seq_len > cfg.seq_len):
                # get any random sequence of frames of size cfg.seq_len of the activity
                start_idx = np.random.choice(data_dict['pose'].shape[0]-cfg.seq_len)
                sample_pose = data_dict['pose'][start_idx : start_idx+cfg.seq_len]
                sample_ori = data_dict['ori'][start_idx : start_idx+cfg.seq_len]
                sample_acc = data_dict['acc'][start_idx : start_idx+cfg.seq_len]
            else:
                sample_pose = data_dict['pose']
                sample_ori = data_dict['ori']
                sample_acc = data_dict['acc']

            #################### convert all roation matrices to quaternion ###############
            ori_euler = np.asarray([transforms3d.euler.mat2euler(sample_ori[k, j, :, :]) for k, j in
                                    itertools.product(range(cfg.seq_len), range(5))])
            ori_euler = ori_euler.reshape(-1, 5, 3)
            ori_euler = ori_euler[:, :, 0:2].reshape(-1, 5 * 2)

            pose_quat = np.asarray([Quaternion(matrix=sample_pose[k, j, :, :]).elements for k, j in
                                   itertools.product(range(cfg.seq_len), range(15))])
            pose_quat = pose_quat.reshape(-1,15*4)

            ################ normalize acceleration ###################

            sample_acc = sample_acc.reshape(-1, 5 * 3)
            sample_acc = (sample_acc - self.acc_stats['mean_channel']) / self.acc_stats['std_channel']


            concat = np.concatenate((ori_euler,sample_acc),axis=1)

            self.input.append(concat)
            self.target.append(pose_quat)

        self.input = np.asarray(self.input)
        self.target = np.asarray(self.target)

    ############### prepare batch in quaternion variant ################
    def prepareBatch_quat(self,batch_no):

        inputs = []
        targets = []
        self.input = []
        self.target = []
        # from sorted files pick sequentiallly
        for idx in range(batch_no, batch_no + cfg.batch_len ):

            data_dict = np.load(self.files[idx], encoding='latin1')
            sample_pose = data_dict['pose']
            sample_ori = data_dict['ori']
            sample_acc = data_dict['acc']
            seq_len = sample_pose.shape[0]

            #################### convert orientation matrices to Quat ###############
            ori_quat = np.asarray([Quaternion(matrix=sample_ori[k, j, :, :]).elements for k, j in
                                   itertools.product(range(seq_len), range(5))])
            ori_quat = ori_quat.reshape(-1, 5 * 4)

            #################### standardize acceleration #################
            # sample_acc = sample_acc.reshape(-1, 5 * 3)
            # sample_acc = (sample_acc - self.acc_stats['mean_channel']) / self.acc_stats['std_channel']
            #
            # concat = np.concatenate((ori_quat, sample_acc), axis=1)



            #################### convert pose matrices to quaternion #################
            pose = np.asarray([Quaternion(matrix=sample_pose[k, j, :, :]).elements for k, j in
                                    itertools.product(range(seq_len), range(15))])
            pose = pose.reshape(-1, 15 * 4)
            inputs.append(ori_quat)
            targets.append(pose)

        # padding of input and output to make the batch of same sequence length
        max_len = max([pose.shape[0] for pose in targets])
        for input,target in zip(inputs,targets):
            quat_ori = np.repeat(Quaternion(matrix=np.eye(3,3)).elements[np.newaxis, ...], 5, axis=0)
            ori = np.array([quat_ori]*max_len).reshape(max_len,-1)
            # acc = np.zeros((max_len,15))
            # padded_in =  np.concatenate((ori, acc), axis=1)
            seq_len = input.shape[0]
            ori[:seq_len, :] = input
            padded_in = torch.Tensor(ori).cuda()

            quat_pose = np.repeat(Quaternion(matrix=np.eye(3, 3)).elements[np.newaxis, ...], 15, axis=0)
            pose = np.array([quat_pose] * max_len).reshape(max_len,-1)
            pose[:seq_len, :] = target
            padded_pose = torch.Tensor(pose).cuda()
            self.input.append(padded_in)
            self.target.append(padded_pose)

        self.input = torch.stack(self.input)
        self.target = torch.stack(self.target)


    def prepareBatchOfMotion(self, batch_sz):
        inputs = []
        targets = []
        self.input = []
        self.target = []
        # from sorted files pick sequentiallly
        for i in range(batch_sz):
            if (len(self.files) == 0):
                break
            idx = np.random.choice(len(self.files))
            print('reading file--', self.files[idx])

            data_dict = np.load(self.files.pop(idx), encoding='latin1')
            sample_pose = np.array(data_dict['pose']).reshape(-1, 15, 3, 3)
            sample_ori = np.array(data_dict['ori']).reshape(-1, 5, 3, 3)
            seq_len = sample_pose.shape[0]

            #################### convert orientation matrices to Quat ###############
            ori_quat = np.asarray([Quaternion(matrix=sample_ori[k, j, :, :]).elements for k, j in
                                   itertools.product(range(seq_len), range(5))])
            ori_quat = ori_quat.reshape(-1, 5 * 4)

            #################### convert pose matrices to quaternion #################
            pose = np.asarray([Quaternion(matrix=sample_pose[k, j, :, :]).elements for k, j in
                               itertools.product(range(seq_len), range(15))])
            pose = pose.reshape(-1, 15 * 4)
            inputs.append(ori_quat)
            targets.append(pose)

        # padding of input and output to make the batch of same sequence length
        max_len = max([pose.shape[0] for pose in targets])
        for input, target in zip(inputs, targets):
            quat_ori = np.repeat(Quaternion(matrix=np.eye(3, 3)).elements[np.newaxis, ...], 5, axis=0)
            ori = np.array([quat_ori] * max_len).reshape(max_len, -1)
            seq_len = input.shape[0]
            ori[:seq_len, :] = input
            padded_in = torch.Tensor(ori).cuda()

            quat_pose = np.repeat(Quaternion(matrix=np.eye(3, 3)).elements[np.newaxis, ...], 15, axis=0)
            pose = np.array([quat_pose] * max_len).reshape(max_len, -1)
            pose[:seq_len, :] = target
            padded_pose = torch.Tensor(pose).cuda()
            self.input.append(padded_in)
            self.target.append(padded_pose)

        if (len(self.input) > 0):
            self.input = torch.stack(self.input)
            self.target = torch.stack(self.target)

    ############### prepare batch in euler variant ################
    def prepareBatch_euler(self):

        self.input = []
        self.target = []
        # act_idx: index of activities of one batch
        act_idx = np.random.choice(len(self.files),cfg.batch_len)
        for idx in act_idx:
            data_dict =  np.load(self.files[idx], encoding='latin1')
            #start_idx = np.random.choice(data_dict['pose'].shape[0]-seq_len+1)
            sample_pose = data_dict['pose']
            sample_ori = data_dict['ori']
            sample_acc = data_dict['acc']
            seq_len = sample_pose.shape[0]
            #################### convert orientation matrices to euler ###############
            ori_euler = np.asarray([transforms3d.euler.mat2euler(sample_ori[k, j, :, :]) for k, j in
                                   itertools.product(range(seq_len), range(5))])
            ori_euler = ori_euler.reshape(-1, 5, 3)
            ori_euler = ori_euler[:,:,0:2].reshape(-1,5*2)
            #################### convert pose matrices to quaternion #################
            pose_quat = np.asarray([Quaternion(matrix=sample_pose[k, j, :, :]).elements for k, j in
                                    itertools.product(range(seq_len), range(15))])
            pose_quat = pose_quat.reshape(-1, 15 * 4)
            pose = torch.Tensor(pose_quat).cuda()
            #################### standardize acceleration #################
            sample_acc = sample_acc.reshape(-1, 5 * 3)
            sample_acc = (sample_acc - self.acc_stats['mean_channel']) / self.acc_stats['std_channel']

            concat = np.concatenate((ori_euler,sample_acc),axis=1)
            concat = torch.Tensor(concat).cuda()

            self.input.append(concat)
            self.target.append(pose)

        # self.input = np.asarray(self.input)
        # self.target = np.asarray(self.target)

# create batch random without replacement
    def createbatch_no_replacement(self):
        self.input = []
        self.target = []
        # act_idx: index of activities of one batch
        #act_idx = np.random.choice(len(self.files), cfg.batch_len)
        no_ofFiles = cfg.batch_len
        if(len(self.files) <= cfg.batch_len):
            no_ofFiles = len(self.files)
        for i in range(no_ofFiles):
            idx = np.random.choice(len(self.files))
            data_dict = np.load(self.files.pop(idx), encoding='latin1')
            sample_pose = data_dict['pose']
            sample_ori = data_dict['ori']
            sample_pose = sample_pose.reshape(-1, 15, 3, 3)
            sample_ori = sample_ori.reshape(-1, 5, 3, 3)
            # seq_len = data_dict['pose'].shape[0]
            # norm = np.linalg.norm(sample_ori[0, 0, :, :], axis=1)
            #################### convert all roation matrices to quaternion ###############
            ori_quat = np.asarray([Quaternion(matrix=sample_ori[k, j, :, :]).elements for k, j in
                                   itertools.product(range(sample_ori.shape[0]), range(5))])
            ori_quat = ori_quat.reshape(-1, 5 * 4)
            pose_quat = np.asarray([Quaternion(matrix=sample_pose[k, j, :, :]).elements for k, j in
                                    itertools.product(range(sample_pose.shape[0]), range(15))])
            pose_quat = pose_quat.reshape(-1, 15 * 4)

            self.input.extend(ori_quat)
            self.target.extend(pose_quat)

        self.input = np.asarray(self.input)
        self.target = np.asarray(self.target)

##### load all files in the dataset
    def loadfiles(self,datapath,trainset):
        listofPath = []

        for d in trainset:
            folderpath = os.path.join(datapath, d)
            for f in os.listdir(folderpath):
                listofPath.append(os.path.join(folderpath, f))
        self.files = listofPath
        return listofPath


########## read one single file
    def readfile(self, file):
        data_dict = np.load(file, encoding='latin1')
        sample_pose = data_dict['pose'].reshape(-1,15,3,3)
        sample_ori = data_dict['ori']
        #sample_acc = data_dict['acc']
        seq_len = sample_pose.shape[0]

        #################### convert orientation matrices to quaternion ###############
        ori_quat = np.asarray([Quaternion(matrix=sample_ori[k, j, :, :]).elements for k, j in
                               itertools.product(range(seq_len), range(5))])
        ori_quat = ori_quat.reshape(-1, 5 * 4)

        #################### convert orientation matrices to euler ###############
        # ori_euler = np.asarray([transforms3d.euler.mat2euler(sample_ori[k, j, :, :]) for k, j in
        #                         itertools.product(range(seq_len), range(5))])
        # ori_euler = ori_euler.reshape(-1, 5, 3)
        # ori_euler = ori_euler[:, :, 0:2].reshape(-1, 5 * 2)

        #################### convert pose matrices to quaternion ###############
        pose_quat = np.asarray([Quaternion(matrix=sample_pose[k, j, :, :]).elements for k, j in
                                itertools.product(range(seq_len), range(15))])

        pose_quat = pose_quat.reshape(-1, 15*4)
        #################### standardize acceleration #################
        # sample_acc = sample_acc.reshape(-1, 5 * 3)
        # sample_acc = (sample_acc - self.acc_stats['mean_channel']) / self.acc_stats['std_channel']
        #
        # concat = np.concatenate((ori_quat, sample_acc), axis=1)

        self.input = ori_quat
        self.target = pose_quat

