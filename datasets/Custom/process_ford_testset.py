import numpy as np
import os
import sys
from shutil import copyfile

NFRAMES = 1

train_dir = '/home/nray1/ms/FORDS_Translation/'
main_dir = '/home/nray1/ms/FORDS_Scale/'

#directories = ['JPEGImages', 'Annotations']


current_dir = 'Annotations/480p/'
current_dir2 = 'JPEGImages/480p/'
objects = os.listdir(train_dir+current_dir)
for o in objects:
    nfiles = 0
    for f in sorted(os.listdir(train_dir+current_dir+o)):
        if nfiles > 1:
            break
        copyfile(train_dir+current_dir+o+'/'+f, main_dir+current_dir+o+'/0'+f)
        f = f.split('_mask')[0]+'.png'
        copyfile(train_dir+current_dir2+o+'/'+f, main_dir+current_dir2+o+'/0'+f)
        print(train_dir+current_dir+o+'/'+f)
        nfiles += 1

#current_dir = 'JPEGImages/480p/'
#current_dir2 = 'Annotations/480p/'
#objects = os.listdir(main_dir+current_dir)
#for o in objects:
#    nfiles = 0
#    for f in sorted(os.listdir(main_dir+current_dir+o)):
#        if not os.path.exists(main_dir+current_dir2+o+'/'+f):
#            os.system('rm -f '+main_dir+current_dir+o+'/'+f)
#
