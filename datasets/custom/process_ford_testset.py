import numpy as np
import os
import sys
from shutil import copyfile

#NFRAMES = 2
#
#train_dir = '/home/nray1/ms/FORDS_Translation/'
#main_dir = '/home/nray1/ms/FORDS_Scale/'
#
##directories = ['JPEGImages', 'Annotations']
#
#
#current_dir = 'Annotations/480p/'
#current_dir2 = 'JPEGImages/480p/'
#objects = os.listdir(train_dir+current_dir)
#for o in objects:
#    nfiles = 0
#    for f in sorted(os.listdir(train_dir+current_dir+o)):
#        if nfiles > NFRAMES-1:
#            break
#        copyfile(train_dir+current_dir+o+'/'+f, main_dir+current_dir+o+'/0'+f)
#        f = f.split('_mask')[0]+'.png'
#        copyfile(train_dir+current_dir2+o+'/'+f, main_dir+current_dir2+o+'/0'+f)
#        print(train_dir+current_dir+o+'/'+f)
#        nfiles += 1

current_dir = '/ms/ford/objects_full/daylight/rotation/'
out_dir = '/ms/FORDS_Rotation/JPEGImages/480p/'

objects = os.listdir(current_dir)
for o in objects:
    files = sorted(os.listdir(current_dir+o))
    for f in files:
        if not os.path.exists(out_dir+o):
            os.mkdir(out_dir+o)

        copyfile(current_dir+o+'/'+f, out_dir+o+'/'+f)

#        if 'mask' in f:
#            copyfile(current_dir+o+'/'+o+'/'+f, out_dir+o+'/'+f)

#current_dir = 'JPEGImages/480p/'
#current_dir2 = 'Annotations/480p/'
#objects = os.listdir(main_dir+current_dir)
#for o in objects:
#    nfiles = 0
#    for f in sorted(os.listdir(main_dir+current_dir+o)):
#        if not os.path.exists(main_dir+current_dir2+o+'/'+f):
#            os.system('rm -f '+main_dir+current_dir+o+'/'+f)
#
