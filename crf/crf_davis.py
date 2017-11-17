#!/usr/bin/env python

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from scipy.ndimage import imread
from scipy.misc import imsave
import cPickle
import numpy
import glob
import os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import sys
import pdb
#imgs_path = "/home/eren/Data/DAVIS/JPEGImages/480p/"
imgs_path = "/home/eren/Data/SegTrackv2/JPEGImages/"
#annots_path = "/home/eren/Data/DAVIS/Annotations/480p/"
annots_path = "/home/eren/Data/SegTrackv2/Annotations/"
preds_path_prefix = "/home/eren/Work/motion_adaptation/forwarded/"
import scipy.misc

def convert_path(inp):
  sp = inp.split("/")
  fwd_idx = sp.index("forwarded")

  seq = sp[fwd_idx + 3]
  fn = sp[-1]
  im_path = imgs_path + seq + "/" + fn.replace(".pickle", ".jpg")
  gt_path = annots_path + seq + "/" + fn.replace(".pickle", ".png")

  sp[fwd_idx + 1] += "_crf"
  sp[-1] = sp[-1].replace(".pickle", ".png")
  out_path = "/".join(sp)
  return im_path, gt_path, out_path

def mkdir_p(d):
  try:
    os.makedirs(d)
  except OSError as err:
    if err.errno != 17:
      raise


def apply_crf(im, pred):
  im = numpy.ascontiguousarray(im)
  pred = numpy.ascontiguousarray(pred.swapaxes(0, 2).swapaxes(1, 2))
  d = dcrf.DenseCRF2D(im.shape[1], im.shape[0], 2)  # width, height, nlabels
  unaries = unary_from_softmax(pred, scale=1.0)
  d.setUnaryEnergy(unaries)

  #print im.shape
  # print annot.shape
  #print pred.shape

  d.addPairwiseGaussian(sxy=0.220880737269, compat=1.24845093352)
  d.addPairwiseBilateral(sxy=22.3761305044, srgb=7.70254062277, rgbim=im, compat=1.40326787165)
  processed = d.inference(12)
  res = numpy.argmax(processed, axis=0).reshape(im.shape[0], im.shape[1])

  return res


def do_seq(seq, model, counter, save=True):
  preds_path = preds_path_prefix + model + "/valid/"
  files = sorted(glob.glob(preds_path + seq + "/*.pickle"))
  ious = []
  fw= open('/home/eren/Data/SegTrackv2/ImageSets/480p/val.txt', 'r')
  imgs= []
  annots= []
  for line in fw:
      sp= line.strip().split(' ')
      imgs.append('/home/eren/Data/SegTrackv2/'+sp[0])
      annots.append('/home/eren/Data/SegTrackv2/'+sp[1])
  for f in files:
    pred_path = f
    im_path, gt_path, out_path = convert_path(f)
    im_path = imgs[counter]
    gt_path = annots[counter]
    counter += 1
    pred = cPickle.load(open(pred_path))
    shape = pred.shape[:-1]
    #pred= scipy.misc.imresize(pred, (480,854), 'nearest')
    im = imread(im_path)
    #im= scipy.misc.imresize(im, (480,854))
    res = apply_crf(im, pred).astype("uint8") * 255
    # before = numpy.argmax(pred, axis=2)
    if save:
      dir_ = "/".join(out_path.split("/")[:-1])
      mkdir_p(dir_)
      imsave(out_path, res)

    #compute iou as well
    groundtruth = imread(gt_path)
    if len(groundtruth.shape)==3:
        groundtruth= groundtruth[:,:,0]
    #groundtruth= scipy.misc.imresize(groundtruth, (480,854), 'nearest')
    I = numpy.logical_and(res == 255, groundtruth == 255).sum()
    U = numpy.logical_or(res == 255, groundtruth == 255).sum()
    IOU = float(I) / U
    ious.append(IOU)

    print out_path, "IOU", IOU

    # plt.imshow(before)
    # plt.figure()
    # plt.imshow(res)
    # plt.show()
  return numpy.mean(ious[1:-1]), counter


def main():

  save = True
  assert len(sys.argv) == 2
  model = sys.argv[1]
#  seqs = ["blackswan", "bmx-trees", "breakdance", "camel", "car-roundabout", "car-shadow", "cows", "dance-twirl",
#          "dog", "drift-chicane",  "drift-straight",  "goat", "horsejump-high", "kite-surf",
#          "libby", "motocross-jump", "paragliding-launch", "parkour", "scooter-black", "soapbox"]
  #seqs= ["bird_of_paradise", "birdfall", "bmx", "cheetah", "drift", "frog", "girl", "humming_bird", "monkey", "monkeydog", "parachute", "penguin", "soldier", "worm"]
  seqs= ["hummingbird", "monkey", "monkeydog", "parachute", "penguin", "soldier", "worm"]
  ious = []
  counter = 567
  for seq in seqs:
    iou, counter = do_seq(seq, model, counter, save=save)
    print iou
    ious.append(iou)

  #ious = Parallel(n_jobs=20)(delayed(do_seq)(seq, model, save=save) for seq in seqs)
  #do_seq(seqs[0], model, save=save)
  print ious
  print numpy.mean(ious)


if __name__ == "__main__":
  main()
