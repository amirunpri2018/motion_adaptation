import numpy as np
import os
import sys
from shutil import copyfile

main_dir = '/home/nray1/ms/ford/segmentation/daylight/scale/'
out_dir = '/home/nray1/ms/FORDS_Scale/Annotations/480p/'
objects = os.listdir(main_dir)
for o in objects:
    if not os.path.exists(out_dir+o):
        os.mkdir(out_dir+o)
    for f in sorted(os.listdir(main_dir+o+'/'+o)):
        if 'mask' in f:
            copyfile(main_dir+o+'/'+o+'/'+f, out_dir+o+'/'+f)

