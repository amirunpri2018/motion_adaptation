import numpy as np
import os
import sys

img_dir= sys.argv[1]+'JPEGImages/480p/'
mask_dir= sys.argv[1]+'Annotations/480p/'
write_file= open(sys.argv[2], 'w')

for d in os.listdir(mask_dir):
    current_dir= mask_dir+d+'/'
    for f in os.listdir(current_dir):
        if f.split('.')[1]=='png':
            write_file.write('JPEGImages/480p/'+d+'/'+f.split('_gt')[0]+'.jpg '+'Annotations/480p/'+d+'/'+f+'\n')

write_file.close()

