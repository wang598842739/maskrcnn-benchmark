#!/usr/bin/env python

import subprocess
import os
import glob


exps = ["/home/jario/spire-net-1812/exps/light_head_64_64", 
        "/home/jario/spire-net-1812/exps/baseline", 
        "/home/jario/spire-net-1812/exps/light_head_64_128",
        "/home/jario/spire-net-1812/exps/light_head_64_256",
        "/home/jario/spire-net-1812/exps/light_head_128_128",
        "/home/jario/spire-net-1812/exps/light_head_128_256"]

num_gpus = 1
path_to_maskrcnn = os.path.abspath(os.path.dirname(os.path.realpath(__file__))+os.path.sep+"..")

for exp in exps:
    yaml_fn = glob.glob(os.path.join(exp, '*.yaml'))
    yaml_fn.sort()

    cmd = "cd {}; ".format(path_to_maskrcnn)    

    if len(yaml_fn) > 0:
        if num_gpus == 1:
            cmd += "python tools/train_net.py --config-file {} OUTPUT_DIR {}".format(yaml_fn[0], exp)
        else:
            cmd += "export NGPUS={}; ".format(num_gpus)
            cmd += "python -m torch.distributed.launch --nproc_per_node=$NGPUS " \
                   "tools/train_net.py --config-file {} OUTPUT_DIR {}".format(yaml_fn[0], exp)
        print(cmd)
        subprocess.call(cmd, shell=True)
    else:
        print("In Dir:{}, yaml not found.".format(exp))

print("all done.")
