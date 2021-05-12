# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------

import os
import numpy as np
import argparse
import pprint
import time
import cv2
import torch

from skimage import io
from model.utils.config import cfg
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--model_dir', dest='model_dir',
                        help='directory to load models',
                        default="./data/pretrained_model", type=str)
    parser.add_argument('--model_weights', dest='model_weights',
                        help='the filename of the pre-trained model weights',
                        default=None, type=str)
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="./images")
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)

    args = parser.parse_args()
    return args


def _get_image_blob(img):
    """
    Given an image, normalise and reshape it to size (600, x) where x<=800
    @param img: BGR images (nd array)
    @return: blob, 4D array, (num_images, h_max, w_max, 3)
             im_scale_factors, 1D array of image scale_factor
    """
    im_orig = img.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape    # (h, w, 3)
    im_size_min = np.min(im_shape[0:2])    # w or h
    im_size_max = np.max(im_shape[0:2])    # w or h

    processed_ims = []
    im_scale_factors = []

    # reshape img size to (600, x) where x<=800
    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)    # scale = 600 / shorter_side(w/h)
        if np.round(im_scale*im_size_max) > cfg.TEST.MAX_SIZE:    # make sure the longer_size <= 1000
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)

        img = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(img)

    # Create a blob (curtain) to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cfg.USE_GPU_NMS = True if torch.cuda.is_available() else False
    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # Initialise the network
    pascal_classes = np.asarray(['__background__',
                                 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                                 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                                 'sheep', 'sofa', 'train', 'tvmonitor'])
    if args.net == 'vgg16':
        fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        raise Exception("network is not defined")

    fasterRCNN.create_architecture()

    # Load weights
    if not os.path.exists(args.model_dir):
        raise Exception('There is no input directory for loading network from ' + args.model_dir)
    load_name = os.path.join(args.model_dir, args.model_weights)
    print("load checkpoint %s" % (load_name))
    if torch.cuda.is_available():
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))

    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    print('load model successfully!')

    # Initialize the tensor holder
    im_data = torch.FloatTensor(1).to(device)
    im_info = torch.FloatTensor(1).to(device)
    num_boxes = torch.LongTensor(1).to(device)
    gt_boxes = torch.FloatTensor(1).to(device)

    """start predictions"""
    fasterRCNN.to(device)
    fasterRCNN.eval()
    start = time.time()
    max_per_image = 100
    thresh = 0.05

    # Load images from local folder
    imglist = os.listdir(args.image_dir)
    num_images = len(imglist)
    print('Loaded: {} images.'.format(num_images))

    while num_images > 0:
        total_tic = time.time()
        num_images -= 1

        # Load images
        im_file = os.path.join(args.image_dir, imglist[num_images])
        im_in = np.array(io.imread(im_file))    # shape (h, w, c)
        im = im_in[:, :, ::-1]    # RGB -> BGR by reversing the channel

        # get image blob (curtain) and scale_factor
        im_blob, im_scales = _get_image_blob(im)
        # print(im_blob.shape)
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)    # 2D array [[h, w, 1.3]]
        im_data_pt = torch.from_numpy(im_blob)    # 4D array, image (1, h, w, 3)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)    # 4D array, image (1, 3, h, w), channel goes first in pytorch
        im_info_pt = torch.from_numpy(im_info_np)

        with torch.no_grad():
            im_data.resize_(im_data_pt.size()).copy_(im_data_pt)    # reshape to (1, 3, h, w) and fill in the img
            im_info.resize_(im_info_pt.size()).copy_(im_info_pt)    # 2D tensor [[height, width, 1.3]]
            gt_boxes.resize_(1, 1, 5).zero_()    # 3D tensor [[[conf, x, y, w, h]]]
            num_boxes.resize_(1).zero_()

        # get the predictions
        det_tic = time.time()
        rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        # Apply bounding-box regression deltas
        if cfg.TEST.BBOX_REG:
            box_deltas = bbox_pred.data
            # Optionally normalize targets by a precomputed mean and stdev
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                if args.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).to(device) \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).to(device)
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).to(device) \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).to(device)
                    box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

        # Simply repeat the boxes, once for each class
        else:
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]
        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        print('remain images:{:d}/{:d}, current image use:{:.3f}s'.format(num_images+1, len(imglist), detect_time))
        im2show = np.copy(im)

        # remove low score bbox, do NMS, match the cls name
        for j in range(1, len(pascal_classes)):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j*4 : (j+1)*4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                im2show = vis_detections(im2show, pascal_classes[j], cls_dets.cpu().numpy(), 0.5)

        # save the detected images
        result_path = os.path.join(args.image_dir, imglist[num_images][:-4] + "_det.jpg")
        cv2.imwrite(result_path, im2show)
