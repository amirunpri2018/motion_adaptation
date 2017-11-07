import numpy as np
import os
import sys
import pdb

split= sys.argv[3]
f= open(sys.argv[2], 'w')
pdb.set_trace()
flist= sorted(os.listdir(sys.argv[1]+split+'/images/'))
for d in flist:
    f.write(sys.argv[1]+split+'/images/'+d+' '+sys.argv[1]+split+'/mask/'+d+'\n')
f.close()
