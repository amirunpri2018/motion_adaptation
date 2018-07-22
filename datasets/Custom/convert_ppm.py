import pdb
import os
import sys
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import subprocess

main_dir= sys.argv[1]+'JPEGImages/'
out_dir = sys.argv[1]+'Annotations/'
for d in os.listdir(main_dir):
    current_dir= main_dir+d+'/GroundTruth/'
    if not os.path.exists(out_dir+d):
        os.mkdir(out_dir+d)
    for f in os.listdir(current_dir):
        print(f)
        if 'PROB' in f or 'pgm' in f or 'dat' in f or 'png' in f:
            continue

        img= Image.open(current_dir+f)
        img_arr= np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)[:,:,0]
        mask= np.zeros_like(img_arr)
        mask[img_arr==0]= 255
        misc.imsave(out_dir+d+'/'+f.split('.')[0]+'.png', mask)



