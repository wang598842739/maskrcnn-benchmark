#!/usr/bin/env python

import subprocess
import os
import glob


## symlink the coco dataset
# cd ~/github/maskrcnn-benchmark
# mkdir -p datasets/coco
# ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
# ln -s /path_to_coco_dataset/train2014 datasets/coco/train2014
# ln -s /path_to_coco_dataset/test2014 datasets/coco/test2014
# ln -s /path_to_coco_dataset/val2014 datasets/coco/val2014
# for pascal voc dataset:
# ln -s /path_to_VOCdevkit_dir datasets/voc

## symlink the pre-trained model
# cd ~/github/maskrcnn-benchmark
# mkdir -p models/pre_trained
# ln -s /path_to_pre_trained_model/pre_trained models/pre_trained


path_to_coco_dataset = '/home/jario/dataset/coco'
# path_to_coco_dataset = None
path_to_coco_spire_dataset = '/media/jario/949AF0D79AF0B738/Dataset/spire_dataset'
# path_to_coco_spire_dataset = None
path_to_pre_trained_model = '/home/jario/spire-net-1810/maskrcnn-benchmark-models'
# path_to_pre_trained_model = None


assert path_to_coco_dataset is not None and path_to_pre_trained_model is not None, \
    "Please set Dataset and Models Dir first..."

path_to_maskrcnn = os.path.abspath(os.path.dirname(os.path.realpath(__file__))+os.path.sep+"..")
cmd = "cd {}; ".format(path_to_maskrcnn)
cmd += "mkdir -p datasets/coco; "
cmd += "ln -s {}/annotations datasets/coco/; ".format(path_to_coco_dataset)
cmd += "ln -s {}/train2014 datasets/coco/; ".format(path_to_coco_dataset)
cmd += "ln -s {}/val2014 datasets/coco/; ".format(path_to_coco_dataset)
# path_to_coco_spire_dataset is an optional choise
if path_to_coco_spire_dataset is not None:
    cmd += "ln -s {} datasets/; ".format(path_to_coco_spire_dataset)
# cmd += "ln -s {}/test2014 datasets/coco/test2014; ".format(path_to_coco_dataset)
cmd += "mkdir -p models; "
cmd += "ln -s {}/pre_trained models/; ".format(path_to_pre_trained_model)
print(cmd)
subprocess.call(cmd, shell=True)

print("all done...")
