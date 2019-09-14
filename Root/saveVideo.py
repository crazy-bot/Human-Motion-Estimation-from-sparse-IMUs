import cv2
import os

gt_arr = []
pred_arr = []

height, width, layers = (640, 480,3)
size = (640, 480)
files = os.listdir('/data/Guha/GR/Output/Prediction/')
for i in range(len(files)):
    #file = glob.('/data/Guha/Tour20/frames/BF6.mp4/frame{}.jpg'.format(i))
    gt_img = cv2.imread('/data/Guha/GR/Output/GT/{}.png'.format(i))
    pred_img = cv2.imread('/data/Guha/GR/Output/Prediction/{}.png'.format(i))

    gt_arr.append(gt_img)
    pred_arr.append(pred_img)

#out = cv2.VideoWriter('AW7.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, size))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
gt_out = cv2.VideoWriter('/data/Guha/GR/Output/TestSet/13/'+'gt.mp4',fourcc, 30.0, size)
pred_out = cv2.VideoWriter('/data/Guha/GR/Output/TestSet/13/'+'pred.mp4',fourcc, 30.0, size)
#
for i in range(len(gt_arr)):
    gt_out.write(gt_arr[i])
print ('gt finished')
for i in range(len(pred_arr)):
    pred_out.write(pred_arr[i])
print ('pred finished')
gt_out.release()
pred_out.release()
