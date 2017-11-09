from datasets.Dataset import ImageDataset
import tensorflow as tf


def zero_label(img_path, label_path):
  #TODO: we load the image again just to get it's size which is kind of a waste (but should not matter for most cases)
  img_contents = tf.read_file(img_path)
  img = tf.image.decode_image(img_contents, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  img.set_shape((None, None, 3))
  label = tf.zeros_like(img, dtype=tf.uint8)[..., 0:1]
  res = {"label": label}
  return res

def read_image_and_annotation_list(fn, data_dir):
  imgs = []
  ans = []
  with open(fn) as f:
    for l in f:
      sp = l.split()
      an = data_dir + sp[1]
      im = data_dir + sp[0]
      imgs.append(im)
      ans.append(an)
  return imgs, ans

KITTI_IMAGE_SIZE= (375, 1242)
NUM_CLASSES = 2
VOID_LABEL = 255  # for translation augmentation

class CustomDataset(ImageDataset):
  def __init__(self, config, subset, coord, fraction=1.0):
#    super(CustomDataset, self).__init__("custom", "", 2, config, subset, coord, None, 255, fraction,
#                                        label_load_fn=zero_label)
    super(CustomDataset, self).__init__("kitti", "", NUM_CLASSES, config, subset, coord,
                                            KITTI_IMAGE_SIZE, VOID_LABEL, fraction, lambda x: x / 255, ignore_classes=[VOID_LABEL])

#    self.file_list = config.unicode("file_list")

  def read_inputfile_lists(self):
     list_file= '/home/eren/Work/motion_adaptation/datasets/Custom/training.txt'
     imgs, ans= read_image_and_annotation_list(self.data_dir+list_file, self.data_dir)
     return imgs, ans
#    imgs = [x.strip() for x in open(self.file_list).readlines()]
#    labels = ["" for _ in imgs]
#    return [imgs, labels]
