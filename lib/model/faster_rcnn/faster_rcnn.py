
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_layers import ROIAlign, ROIPool
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.utils.net_utils import _smooth_l1_loss


class _fasterRCNN(nn.Module):
    """
    Define the RPN & ROI_Pooling in this father class (_fasterRCNN)
    Define the backbone and head in the child class (resnet)
    combine the above two by call _fasterRCNN.create_architecture()
    """

    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic

        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)

        # Produces labels for proposal classification and bbox regression.
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # ROIPooling or ROIAlign
        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)


    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        """
        get call when fasterRCNN(im_data, im_info, gt_boxes, num_boxes), after fasterRCNN.create_architecture
        @param im_data: 4D tensor, (batch, 3, h, w)
        @param im_info: 2D tensor, [[height, width, scale_factor (1.3)]]
        @param gt_boxes: 3D tensor [[[conf, x, y, w, h]]]
        @param num_boxes: 1D tensor [num_boxes]
        @return:
        """
        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        self.data = num_boxes.data
        num_boxes = self.data

        # 1. img --> backbone (resnet) --> feature maps
        base_feat = self.RCNN_base(im_data)    # RCNN_base() is defined in the child class (resnet)

        # 2. feature map --> RPN --> roi bboxes
        # rois: 3D tensor (batch, 300, 5), each column is a bbox [batch_ind, x1, y1, x2, y2]
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground truth bboxes for refining
        # self.training is a class attribute in nn.module
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        # 3. rois --> roi pooling --> pooled features 4D tensor (num_proposals, 1024, 7, 7)
        rois = Variable(rois)
        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))

        # 4. pooled features --> get bbox predictions
        pooled_feat = self._head_to_tail(pooled_feat)    # _head_to_tail() is defined in the child class (resnet)
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)    # RCNN_bbox_pred() is defined in the child class (resnet)

        # select the corresponding columns according to roi labels
        if self.training and not self.class_agnostic:
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1)/4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1,
                                            rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # 4. pooled features --> get class predictions
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        # loss function
        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        if self.training:
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label


    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initializer: truncated normal and random normal, m is a parameter
            """
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)


    def create_architecture(self):
        """
        Merge resnet backbone to fasterRCNN, initialise the entire fasterRCNN
        """
        self._init_modules()    # _init_modules() is defined in the child class (resnet)
        self._init_weights()
