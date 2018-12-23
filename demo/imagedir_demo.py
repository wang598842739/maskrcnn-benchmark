# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import os
import cv2

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

import time


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Image Dir Demo")
    parser.add_argument(
        "--image-dir",
        required=True,
        help="path to image dir",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="path to output dir",
    )
    parser.add_argument(
        "--config-file",
        default="configs/e2e_faster_rcnn_R_50_FPN_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--templete",
        default='coco',
        help="Categories templete for pretty print",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=800,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
        categories_templete=args.templete,
    )

    annotation_dir = os.path.join(args.output_dir, 'annotations')
    vis_dir = os.path.join(args.output_dir, 'vis')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.makedirs(annotation_dir)
        os.makedirs(vis_dir)

    for f in os.listdir(args.image_dir):
        if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg'):
            image_fn = os.path.join(args.image_dir, f)
            img = cv2.imread(image_fn)
            start_time = time.time()
            prediction = coco_demo.compute_prediction(img)
            json_str = coco_demo.prediction_to_json_str(f, prediction)
            ofs = open(os.path.join(annotation_dir, f+'.json'), 'w')
            ofs.write(json_str)
            ofs.close()
            composite = coco_demo.run_on_opencv_image(img)
            print("Time: {:.2f} s / img".format(time.time() - start_time))
            cv2.imshow("COCO detections", composite)
            # cv2.imwrite(os.path.join(vis_dir, f+'.png'), composite)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
