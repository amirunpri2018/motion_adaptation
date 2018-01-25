import numpy as np
import sys
import os
import cv2
import numpy
import pdb
import matplotlib.pyplot as plt

def compute_measures_for_binary_segmentation(prediction, target):
  T = target.sum()
  P = prediction.sum()
  I = numpy.logical_and(prediction == 1, target == 1).sum()
  U = numpy.logical_or(prediction == 1, target == 1).sum()

  if U == 0:
    recall = 1.0
    precision = 1.0
    iou = 1.0
  else:
    if T == 0:
      recall = 1.0
    else:
      recall = float(I) / T

    if P == 0:
      precision = 1.0
    else:
      precision = float(I) / P
  if U==0:
      iou= 0
  else:
      iou = float(I) / U

  if recall==0 and precision==0:
      fmeasure =0
  else:
      fmeasure= (2*recall*precision)/ (recall+precision)
  measures = {"recall": recall, "precision": precision, "iou": iou, "fmeasure":fmeasure}

  return measures



test_dir1= '/home/eren/Data/DAVIS/ARP/'
test_dir2= '/home/eren/Data/DAVIS/Motion_4/'
gt_dir= '/home/eren/Data/DAVIS/Annotations/480p/'
mean_ious= []
for d in sorted(os.listdir(gt_dir)):
    ious=[]
    for f in os.listdir(gt_dir+d):
        if 'png' not in f:
            continue
        gt = cv2.imread(gt_dir+d+'/'+f)[:,:,0]
        if not os.path.exists(test_dir1+d+'/'+f):
            continue
        if not os.path.exists(test_dir2+d+'/'+f):
            continue

        test1= cv2.imread(test_dir1+d+'/'+f)[:,:,0]
        test2= cv2.imread(test_dir2+d+'/'+f)[:,:,0]
#        test= (test1+test2)/2
        th= 127
        gt[gt<th]=0
#        test1[test1]=0
#        test2[test2<th]=0
        gt[gt>=th]=1
        test1[test1!=0]=1
        test2[test2!=0]=1

        test= numpy.logical_and(test1, test2)
        measures= compute_measures_for_binary_segmentation(test, gt)
        ious.append(measures["iou"])
    print('Mean IoU ', np.mean(ious), ' for ', d)
    if not np.isnan(np.mean(ious)):
        mean_ious.append(np.mean(ious))


print('Total mean: ', np.mean(mean_ious), ' For #sqs: ', len(mean_ious))


