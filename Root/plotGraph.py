import  matplotlib.pyplot as plt
import numpy as np

############# file paths of loss files #############
file_path1 = '/data/Guha/GR/Output/loss/loss_dipf9.txt'
file_path2 = '/data/Guha/GR/Output/loss/loss_dipsynthetic.txt'
file_path3 = '/data/Guha/GR/Output/loss/loss_dip_birnn.txt'
file_path4 = '/data/Guha/GR/Output/loss/loss_dip_mlp.txt'

loss1,loss2, loss3, loss4 = [],[],[],[]
with open(file_path1, 'r') as file1, open(file_path2, 'r') as file2, open(file_path3, 'r') as file3, open(file_path4, 'r') as file4:
    for line1,line2,line3,line4 in zip (file1.readlines(),file2.readlines(),file3.readlines(),file4.readlines()):
        loss1.append(float(line1))
        loss2.append(float(line2))
        loss3.append(float(line3))
        loss4.append(float(line4))

x_ind = np.arange(len(loss1))
fig, ax = plt.subplots(figsize=(10,7))

################### to plot bar plots ####################
# width = 0.18
# r1 = np.arange(len(loss1))
# r2 = [x + width for x in r1]
# r3 = [x + width for x in r2]
# r4 = [x + width for x in r3]
# rects1 = ax.bar(r1, loss1, width, edgecolor='white',label='model:BiRNN',align='edge')
# rects2 = ax.bar(r2, loss2,  width, edgecolor='white', label='model: MLP',align='edge')
# rects1 = ax.bar(r3, loss3, width, edgecolor='white',label='DIP:Synthetic',align='edge')
# rects2 = ax.bar(r4, loss4,  width, edgecolor='white',label='DIP:Fine-tuned',align='edge')

############# to plot line graph ###############
ax.plot(np.arange(len(loss1)),loss1,'b8:',label='DIP:synthetic')
ax.plot(np.arange(len(loss2)),loss2,'Pg-',label='DIP:finetuned')
ax.plot(np.arange(len(loss1)),loss3,'rs--',label='Our Short BiLSTM',dashes=[2,2])
ax.plot(np.arange(len(loss2)),loss4,'>c-.',label='Our MLP')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('mean joint angle error (euler in degrees)')
ax.set_xlabel('test activities')
ax.set_title('State-of-the-art models vs our trained models')
ax.set_xticks(x_ind)
#ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))
ax.legend()
plt.show()

