from datasets.Dataset import ImageDataset
import tensorflow as tf

KITTI_IMAGE_SIZE= (375, 1242)
NUM_CLASSES = 2
VOID_LABEL = 255  # for translation augmentation

def read_image_flow_and_annotation_list(fn, data_dir):
  imgs = []
  flows= []
  ans = []
  with open(fn) as f:
    for l in f:
      sp = l.split()
      an = data_dir + sp[1]
      im = data_dir + sp[0]
      flow = data_dir + sp[0].replace('images','optflow')
      imgs.append(im)
      ans.append(an)
      flows.append(flow)
  return imgs, flows, ans


class CustomDataset(ImageDataset):
  def __init__(self, config, subset, coord, fraction=1.0):
    super(CustomDataset, self).__init__("kitti", "", NUM_CLASSES, config, subset, coord,
                                            KITTI_IMAGE_SIZE, VOID_LABEL, fraction, lambda x: x / 255, ignore_classes=[VOID_LABEL])

  def read_inputfile_lists(self):
     list_file= '/home/eren/Work/motion_adaptation/datasets/Custom/training.txt'
     imgs, flows, ans= read_image_flow_and_annotation_list(self.data_dir+list_file, self.data_dir)
     return imgs, flows, ans

