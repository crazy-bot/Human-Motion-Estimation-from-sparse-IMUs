# human-pose
This file is to give a brief overview of each file included in this project.
For the full report kindly see here [Deep Learning Precise 3D Human Pose from Sparse IMUs](Reports/GR_report.pdf)
File description:
1. analyseData:
    • reads raw orientation from syntehtic and DIP_IMU
    • reads ground truth pose from syntehtic and DIP_IMU
    • reads calibrated orientation from prepared syntehtic and DIP_IMU dataset
    • visualize data distribution in boxplot
2. calculateDIpLoss:
    • calculates per joint per frame loss in euler angle degrees of provided DIP_model from generated npz files by run_evaluation file of DIP_IMU
    • saves the losses in text file
3. compareDIPFiles:
    • reads DIP_IMU_nn
    • reads DIP_IMU and calibrate according to our 1st understanding
    • compare values between the above two
4. Config:
    • contains common configuration parameters used by multiple files
5. createData:
    • calibrates DIP_IMU raw data following 1st approach
    • save them in separate files per activity
6. DIP_calib_BiRNN:
    • training of short LSTM on our calibrated DIP_IMU
7. eval_dip_nn:
    • evaluate own trained model on DIP_IMU_nn data
    • use testWindow method to evaluate Short BiLSTM
    • use test method to evaluate MLP
8. eval_model:
    • evaluate all BiLSTM model trained on synthetic data
9. eval_ori_pose:
    • evaluate forward kinematic
    • get predicted orientation
    • feed results of forward kinematic to get pose from trained model
10. IMUDataset:
    • reads data from generated calibrated files
    • multiple methods to create batch as and when required based on different strategies – to be called during training
    • read single file – to be called during test
11. myUtil:
    • contains some generic utility methods
12. Network:
    • contains the architectures of different network as experimented
13. new_createData:
    • calibrates DIP_IMU raw data following last approach
    • save them in separate files per activity
14. new_prepareData:
    • calibrates synthetic raw data following last approach
    • save them in separate files per activity in separate folders for different datasets
15. parseJson -
    • parse json files 
    • calibrate data and store in desired format
    • It was written to process our own recordings
16. plotGraph –
    • generate figures given in the report from loss files
17. prepareData-
    • calibrates synthetic raw data following first approach
    • save them in separate files per activity in separate folders for different datasets
18. saveVideo -
    • creates video from frames stored in a folder
    • used to create video of activities for target and prediction
19. test_MLP-
    • tests trained MLP model
    • datapath and modelpath need to be specified in the file as required
20. train_BiLSTM -
    • trains RNN model with sequential stateful strategy
21. train_BiRNN-
    • trains RNN model with random stateless strategy
22. train_BiRNN_Full-
    • trains RNN model with random stateless strategy on multiple datasets
23.  train_MLP-
    • trains MLP model 
24. visualize _<dip/DFKI/>-
    • visualize result generated from eval_model
    • uses SMPL python package dependencies
    • it also can store the frames for making video by saveVideo file. 

Refer to the readme file for data description in the respective folder.
Data can be downloaded from - http://dip.is.tue.mpg.de/downloads
