import numpy as np

def parseJson(filepath):
    import json
    SMPL_SENSOR = ['L_Shoulder', 'R_Shoulder', 'L_Knee', 'R_Knee', 'Head', 'Pelvis']
    sensorDict = {'L_Shoulder':30,'R_Shoulder':25,'L_knee':10,'R_knee':12,'Head':20,'Pelvis':15}
    sensorIds = ['30','25','10','12','20','15']
    #sensorIds = ['25', '30', '12', '11', '20', '16']
    rawOri = []
    with open(filepath, "r") as read_file:
        lines = read_file.readlines()
        for i,line in enumerate(lines):
            print(i)
            jsondict = json.loads(line)
            imus = jsondict['data']['imu']

            # loop through 6 sensor ids #
            ori_t=[]
            for id in sensorIds:
                if(id in imus):
                    ori_t.append(imus[id]['orientation'])
                else:
                    ori_t.append([0,0,0,0])

            rawOri.append(ori_t)
        rawOri = np.array(rawOri)

        ## impute missing sensor values with previous timestep ##
        while(np.any(~np.any(rawOri,axis=2))):
            m_rows,m_cols = np.where(~np.any(rawOri,axis=2))
            rawOri[m_rows,m_cols] = rawOri[(m_rows-1),m_cols]
    return rawOri


def calibrateRawIMU(ori):
    import quaternion
    import itertools
    import myUtil
    ######### **********the order of IMUs are: [left_lower_wrist, right_lower_wrist, left_lower_leg, right_loewr_leg, head, back]
    SMPL_SENSOR = ['L_Shoulder', 'R_Shoulder', 'L_Knee', 'R_Knee', 'Head', 'Pelvis']

    ############### calculation in Rotation Matrix #########################
    seq_len = len(ori)
    # head sensor for the first frame
    head_quat = quaternion.from_float_array( ori[100, 4, :])
    Q = np.quaternion(0.5, 0.5, 0.5, 0.5)
    # calib: R(T_I) which is constant over the frames
    calib = np.linalg.inv(quaternion.as_rotation_matrix( np.dot(head_quat, Q)))

    # bone2sensor: R(B_S) calculated once for each sensor and remain constant over the frames as used further
    bone2sensor = {}
    for i in range(len(SMPL_SENSOR)):
        sensorid = SMPL_SENSOR[i]
        qbone = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(myUtil.getGlobalBoneOri(sensorid)))
        qsensor = np.dot(calib, quaternion.as_rotation_matrix(quaternion.from_float_array(ori[100, i, :])))
        boneQuat = np.dot(np.linalg.inv(qsensor), qbone)
        bone2sensor[i] = boneQuat

    # calibrated_ori: R(T_B) calculated as calib * sensor data(changes every frame) * bone2sensor(corresponding sensor)
    calibrated_ori = np.asarray(
        [np.linalg.multi_dot([calib, quaternion.as_rotation_matrix(quaternion.from_float_array(ori[k, j, :] )), bone2sensor[j]]) for
         k, j in itertools.product(range(seq_len), range(6))]
    )
    calibrated_ori = calibrated_ori.reshape(-1, 6, 3,3)
    root_inv = np.linalg.inv(calibrated_ori[:, 5, :,:])
    # root_inv = np.transpose(calibrated_ori[:, 5], [0, 2, 1])
    norm_ori = np.asarray(
        [np.matmul(root_inv[k], calibrated_ori[k, j, :,:]) for k, j in itertools.product(range(seq_len), range(6))])
    norm_ori = norm_ori.reshape(-1, 6, 3,3)

    return norm_ori[:,0:5,:,:]

if __name__ == '__main__':
    filepath = "/data/Guha/GR/Dataset/DFKI/walking_1.json"
    ori = parseJson(filepath)
    #calibrated = calibrateRawIMU(ori)
    np.savez_compressed( "/data/Guha/GR/Dataset/DFKI/walking_1", ori=ori)
