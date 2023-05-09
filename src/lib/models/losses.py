# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _transpose_and_gather_feat
import torch.nn.functional as F
from PIL import Image, ImageDraw
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import time
from utils.image import draw_ellipse_gaussian
import warnings

DRAW = False

def _slow_neg_loss(pred, gt):
  '''focal loss from CornerNet'''
  pos_inds = gt.eq(1)
  neg_inds = gt.lt(1)

  neg_weights = torch.pow(1 - gt[neg_inds], 4)

  loss = 0
  pos_pred = pred[pos_inds]
  neg_pred = pred[neg_inds]

  pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
  neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if pos_pred.nelement() == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss


def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

def _not_faster_neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()    
    num_pos  = pos_inds.float().sum()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = torch.log(1 - trans_pred) * torch.pow(trans_pred, 2) * weight
    all_loss = all_loss.sum()

    if num_pos > 0:
        all_loss /= num_pos
    loss -=  all_loss
    return loss

def _slow_reg_loss(regr, gt_regr, mask):
    num  = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr    = regr[mask]
    gt_regr = gt_regr[mask]
    
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr,  reduction='sum')
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

def _reg_loss(regr, gt_regr, mask):
  ''' L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
  '''
  num = mask.float().sum()
  mask = mask.unsqueeze(2).expand_as(gt_regr).float()

  regr = regr * mask
  gt_regr = gt_regr * mask
    
  regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr,  reduction='sum')
  regr_loss = regr_loss / (num + 1e-4)
  return regr_loss

def dice_score(inputs, targets):
    inputs = inputs.sigmoid()
    numerator = 2*torch.matmul(inputs, targets.t())
    denominator = (inputs * inputs).sum(-1)[:, None] + (targets * targets).sum(-1)
    score = numerator / (denominator + 1e-4)
    return score

def differentiable_gaussian(H, W, centers, radius, ceiling = 'sigmoid', r_variation = 'one'):

  device = centers.device

  N = centers.shape[-1]//2

  centers = centers.view(centers.shape[0], centers.shape[1], N, 2)
  # batch_size, nb_max_obj, nb_points, 2

  centers = centers.unsqueeze(2).unsqueeze(2)
  # batch_size, nb_max_obj, 1, 1, nb_points, 2

  centers = centers.repeat(1, 1, H, W, 1, 1)
  # batch_size, nb_max_obj, H, W, nb_points, 2

  indexes = torch.FloatTensor([[[i,j] for i in range(W)] for j in range(H)]).to(device)
  indexes = indexes.unsqueeze(-2)
  # batch_size, nb_max_obj, H, W, nb_points, 2
  indexes = indexes.expand(centers.shape[0], centers.shape[1], H, W, N, 2)

  centers = centers - indexes

  centers = - torch.pow(centers,2)

  centers = centers.sum(-1)
  # batch_size, nb_max_obj, H, W, nb_points

  # batch_size, nb_max_obj, nb_points
  radius = radius.unsqueeze(-2).unsqueeze(-2)
  if r_variation == 'two':
    radius = radius.repeat(1, 1, 1, 1, N//2)
  elif r_variation == 'four':
    radius = radius.repeat(1, 1, 1, 1, N//4)
  radius = radius.expand(radius.shape[0], radius.shape[1], H, W,N)
  # batch_size, nb_max_obj, H, W, nb_points

  centers = torch.exp(centers/(2*torch.pow(radius,2)))

  centers = centers.sum(-1)
  # batch_size, nb_max_obj, H, W

  if ceiling == 'sigmoid':
    centers = torch.sigmoid(centers)
  elif ceiling == 'clamp':
    centers = torch.clamp(centers, 0, 1)
  elif ceiling == 'tanh':
    centers = torch.tanh(centers)

  return centers


class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)

class RegLoss(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(RegLoss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    loss = _reg_loss(pred, target, mask)
    return loss

class RegL1Loss(nn.Module):
  def __init__(self):
    super(RegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()

    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask,  reduction='sum')

    loss = loss / (mask.sum() + 1e-4)

    return loss


class PolyLoss(nn.Module):
    def __init__(self, opt):
        super(PolyLoss, self).__init__()
        self.opt = opt

    def forward(self, output, mask, ind, target, freq_mask = None, peak = None, hm = None):
        """
        Parameters:
            output: output of polygon head
              [batch_size, 2*nb_vertices, height, width]
            mask: selected objects
              [batch_size, nb_max_objects]
            ind:
              [batch_size, nb_max_objects]
            target: ground-truth for the polygons
              [batch_size, nb_max_objects, 2*nb_vertices]
            hm: output of heatmap head
              [batch_size, nb_categories, height, width]
        Returns:
            loss: scalar
        """

        pred = _transpose_and_gather_feat(output, ind)

        if self.opt.poly_loss == 'l1': 
            mask = mask.unsqueeze(2).expand_as(pred).float()
            loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
            loss /= (mask.sum() + 1e-6) #/ (freq_mask.mean() + 1e-4)
        else: 
            raise NotImplementedError

        return loss

class GaussianLoss(nn.Module):
    def __init__(self, opt):
        super(GaussianLoss, self).__init__()
        self.opt = opt

        self.bce = torch.nn.BCELoss(reduction='mean')

    def forward(self, centers, radius, mask, ind, target, peak):
        """
        Parameters:
            centers: output of gaussian centers head
              [batch_size, 2*nb_vertices, height, width]
            radius: output of gaussian std head
              [batch_size, nb_radius, height, width]
            mask: selected objects
              [batch_size, nb_max_objects]
            ind: peak of heatmap (encoded)
              [batch_size, nb_max_objects]
            target: ground-truth for the segmentation mask
              [batch_size, nb_max_objects, height, width]
            peak: ground-truth for the peaks of heatmap
              [batch_size, nb_max_objects, 2]
        Returns:
            loss: scalar
        """        

        #print('peak', peak.shape)
        #print('centers', centers.shape)
        #print('radius', radius.shape)
        #print('target', target.shape)


        pred = _transpose_and_gather_feat(centers, ind)
        pred_radius = _transpose_and_gather_feat(radius, ind)#+10 #.detach().cpu().numpy() +10
        #pred_radius = np.squeeze(pred_radius)


        # Recenter
        pred[:,:,0::2]+=torch.unsqueeze(peak[:,:,0],2)
        pred[:,:,1::2]+=torch.unsqueeze(peak[:,:,1],2)

        #pred_gaussian_tensor = torch.zeros_like(target)
        #pred_gaussian_tensor = display_gaussian_image(pred_gaussian_tensor, pred, pred_radius)#, peak)

        #print(pred_gaussian_tensor.shape)
        #print(pred_radius)

        H, W = target[0][0].shape

        pred_oneradius = differentiable_gaussian(H,W, pred, pred_radius[:,:,0].unsqueeze(-1), self.opt.gaussian_ceiling, 'one')

        pred = differentiable_gaussian(H,W, pred, pred_radius, self.opt.gaussian_ceiling, self.opt.r_variation)

        mask = mask.unsqueeze(-1).unsqueeze(-1)
        # batch_size, nb_max_obj, H, W, nb_points, 2
        mask = mask.expand(mask.shape[0], mask.shape[1], H, W)

        #bce_loss += F.binary_cross_entropy_with_logits(pred*mask, target*mask, reduction='mean')
        #bce_loss = bce_loss/(mask.sum() + 1e-6)

        if self.opt.gaussian_loss == 'bce':
          loss = self.bce(pred*mask, target*mask)
        elif self.opt.gaussian_loss == 'dice':

          inputs = (pred*mask).view(-1)
          targets = (target*mask).view(-1)

          intersection = (inputs * targets).sum()
          dice = (2.*intersection)/(inputs.sum() + targets.sum() + 1e-9)
          loss = 1 - dice
        else :
          raise(NotImplementedError)

        loss_oneradius = 0.0

        if self.opt.r_variation == 'composed':
          if self.opt.gaussian_loss == 'bce':
            loss_oneradius = self.bce(pred_oneradius*mask, target*mask)
          elif self.opt.gaussian_loss == 'dice':

            inputs = (pred_oneradius*mask).view(-1)
            targets = (target*mask).view(-1)

            intersection = (inputs * targets).sum()
            dice = (2.*intersection)/(inputs.sum() + targets.sum() + 1e-9)
            loss_oneradius = 1 - dice
          else :
            raise(NotImplementedError)

        return loss + loss_oneradius, loss + loss_oneradius


class AreaPolyLoss(nn.Module):
    def __init__(self):
        super(AreaPolyLoss, self).__init__()

    def forward(self, output, mask, ind, target, centers):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = 0

        for batch in range(output.shape[0]):
            polygon_mask = Image.new('L', (output.shape[-1], output.shape[-2]), 0)
            poly_points = []
            for i in range(0, pred[batch].shape[0]):  # nbr objects
                for j in range(0, pred[batch].shape[1] - 1, 2):  # points
                    poly_points.append((int(pred[batch][i][j] + centers[batch][i][0]),
                                        int(pred[batch][i][j+1] + centers[batch][i][1])))

            ImageDraw.Draw(polygon_mask).polygon(poly_points, outline=0, fill=255)
            polygon_mask = torch.Tensor(np.array(polygon_mask)).cuda()
            loss += nn.MSELoss()(polygon_mask, target[batch])
        # loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss


class NormRegL1Loss(nn.Module):
  def __init__(self):
    super(NormRegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    pred = pred / (target + 1e-4)
    target = target * 0 + 1
    loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss

class RegWeightedL1Loss(nn.Module):
  def __init__(self):
    super(RegWeightedL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask,  reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss

class L1Loss(nn.Module):
  def __init__(self):
    super(L1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    return loss

class BinRotLoss(nn.Module):
  def __init__(self):
    super(BinRotLoss, self).__init__()
  
  def forward(self, output, mask, ind, rotbin, rotres):
    pred = _transpose_and_gather_feat(output, ind)
    loss = compute_rot_loss(pred, rotbin, rotres, mask)
    return loss

def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')

# TODO: weight
def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')

def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
          valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
          valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
          valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
          valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res
