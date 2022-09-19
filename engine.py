import math
import os
import pdb
import sys
from typing import Iterable
import numpy as np
import copy
import itertools
import json
import torch

import util.misc as utils
from datasets.hico_eval import HICOEvaluator
from datasets.vcoco_eval import VCOCOEvaluator
from torch.utils.tensorboard import SummaryWriter

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    remove: bool = False, batch_weight_mode: int = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="\t")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if hasattr(criterion, 'loss_labels'):
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    else:
        metric_logger.add_meter('obj_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    if isinstance(data_loader, list):
        obj_vb_matrix_hico = np.load('data/hico_20160224_det/hico_obj_vb_matrix.npy')
        obj_vb_matrix_vcoco = np.load('data/v-coco/vcoco_obj_vb_matrix.npy')
        for data_sub1, data_sub2 in metric_logger.log_every(data_loader, print_freq, header):
            samples = []
            sample = utils.nested_tensor_from_tensor_list(
                [data_sub1[0].tensors.squeeze(0).to(device), data_sub2[0].tensors.squeeze(0).to(device)])
            samples.append(sample)
            targets = []
            target1 = [{k: v.to(device) for k, v in t.items() if k != 'filename'} for t in data_sub1[1]]
            target2 = [{k: v.to(device) for k, v in t.items() if k != 'filename'} for t in data_sub2[1]]
            targets.append((target1[0], target2[0]))

            if targets[0][0]['verb_labels'].shape[1] == 117:
                obj_vb_matrix = obj_vb_matrix_hico
            elif targets[0][0]['verb_labels'].shape[1] == 29:
                obj_vb_matrix = obj_vb_matrix_vcoco
            targets_compo = []
            for i in range(2):
                target_compo = copy.deepcopy(targets[0][i])
                num_HO_1 = target_compo['verb_labels'].shape[0]
                num_HO_2 = targets[0][1 - i]['verb_labels'].shape[0]
                verb_list = []
                for j in range(num_HO_2):
                    verb_indices = np.argwhere(targets[0][1 - i]['verb_labels'][j, :].cpu().numpy() == 1).reshape(-1)
                    for verb_index in verb_indices:
                        verb_list.append(verb_index)
                for k in range(num_HO_1):
                    obj_index = target_compo['obj_labels'][k]
                    target_compo['verb_labels'][k, :] = 0
                    for verb in verb_list:
                        if remove:
                            target_compo['verb_labels'][k, verb] = obj_vb_matrix[obj_index, verb]
                        else:
                            target_compo['verb_labels'][k, verb] = 1
                    if not target_compo['verb_labels'][k, :].type(torch.uint8).any():
                        target_compo['matching_labels'][k] = 0
                matching_indices = target_compo['matching_labels'] == 1
                if not target_compo['obj_labels'][matching_indices].shape[0] == 0:
                    target_compo['obj_labels'] = target_compo['obj_labels'][matching_indices]
                    target_compo['verb_labels'] = target_compo['verb_labels'][matching_indices]
                    target_compo['sub_boxes'] = target_compo['sub_boxes'][matching_indices]
                    target_compo['obj_boxes'] = target_compo['obj_boxes'][matching_indices]
                    target_compo['matching_labels'] = target_compo['matching_labels'][matching_indices]
                targets_compo.append(target_compo)
            targets.append((targets_compo[0], targets_compo[1]))
            '''
            for i in range(2):
                target_compo = copy.deepcopy(targets[i])
                num_HO_1 = target_compo[0]['verb_labels'].shape[0]
                num_HO_2 = targets[1 - i][0]['verb_labels'].shape[0]
                if num_HO_1 > num_HO_2:
                    padding = torch.zeros([num_HO_1 - num_HO_2, target_compo[0]['verb_labels'].shape[1]]).to(device)
                    target_compo[0]['verb_labels'] = torch.cat((targets[1 - i][0]['verb_labels'], padding), 0)
                elif num_HO_1 < num_HO_2:
                    target_compo[0]['verb_labels'] = copy.deepcopy(targets[1 - i][0]['verb_labels'][0:num_HO_1, :])
                else:
                    target_compo[0]['verb_labels'] = copy.deepcopy(targets[1 - i][0]['verb_labels'])
                
                for j in range(num_HO_1):
                    obj_index = target_compo[0]['obj_labels'][j]
                    verb_indices = np.argwhere(target_compo[0]['verb_labels'][j, :].cpu().numpy() == 1).reshape(-1)
                    
                    if target_compo[0]['verb_labels'].shape[1] == 117:
                        for k in range(verb_indices.shape[0]):
                            target_compo[0]['verb_labels'][j, verb_indices[k]] = obj_vb_matrix_hico[obj_index, verb_indices[k]]
                        if not target_compo[0]['verb_labels'][j, :].type(torch.uint8).any():
                            target_compo[0]['verb_labels'][j, 57] = 1
                    elif target_compo[0]['verb_labels'].shape[1] == 29:
                        for k in range(verb_indices.shape[0]):
                            target_compo[0]['verb_labels'][j, verb_indices[k]] = obj_vb_matrix_vcoco[obj_index, verb_indices[k]]
                        if not target_compo[0]['verb_labels'][j, :].type(torch.uint8).any():
                            target_compo[0]['matching_labels'][j] = 0
'''
            outputs = model(samples)
            batch_weight_list = [[1, 1],
                                 [1.5, 0.5],
                                 [1.8, 0.2],
                                 [1.95, 0.05],
                                 [2, 0],
                                 [0, 2]]
            batch_weight = batch_weight_list[batch_weight_mode]
            losses_avg = 0
            for i in range(len(outputs)):
                loss_dict = criterion(outputs[i], targets[i])
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                losses = losses * batch_weight[i]
                losses_avg += losses

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                              for k, v in loss_dict_reduced.items()}
                loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                            for k, v in loss_dict_reduced.items() if k in weight_dict}
                losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

                loss_value = losses_reduced_scaled.item()

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    print(loss_dict_reduced)
                    sys.exit(1)

            losses_avg = losses_avg / len(outputs)
            optimizer.zero_grad()
            losses_avg.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
            if hasattr(criterion, 'loss_labels'):
                metric_logger.update(class_error=loss_dict_reduced['class_error'])
            else:
                metric_logger.update(obj_class_error=loss_dict_reduced['obj_class_error'])
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    else:
        for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items() if k != 'filename'} for t in targets]
            # samples: NestedTensor, samples.mask: tensor[batch_size, H, W], samples.tensors: tensor[batch_size, 3, H, W]
            # targets: tuple
            # targets[0:batch_size]: dict{'orig_size': tensor([480, 640]), 'size': tensor([640, 944]), 'boxes': tensor([num_boxes, 4]),
            # 'labels': tensor([num_boxes]), 'iscrowd': tensor([num_boxes]), 'area': tensor([num_boxes]),
            # 'filename': 'COCO_train2014_000000323728.jpg', 'obj_labels': tensor([num_objects]), 'verb_labels': tensor([num_objects, 29]),
            # 'sub_boxes': tensor([num_objects, 4]), 'obj_boxes': tensor([num_objects, 4]), 'matching_labels': tensor([num_objects])}
            outputs = model(samples)

            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                          for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
            if hasattr(criterion, 'loss_labels'):
                metric_logger.update(class_error=loss_dict_reduced['class_error'])
            else:
                metric_logger.update(obj_class_error=loss_dict_reduced['obj_class_error'])
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_hoi(dataset_file, model, postprocessors, data_loader, subject_category_id, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = []
    gts = []
    indices = []
    for samples, targets in metric_logger.log_every(data_loader, 100, header):
        samples = samples.to(device)

        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['hoi'](outputs, orig_target_sizes)

        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))


    metric_logger.synchronize_between_processes()

    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]

    if dataset_file == 'hico':
        evaluator = HICOEvaluator(preds, gts, data_loader.dataset.rare_triplets,
                                  data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat, args=args)
    elif dataset_file == 'vcoco':
        evaluator = VCOCOEvaluator(preds, gts, data_loader.dataset.correct_mat, use_nms_filter=args.use_nms_filter)

    stats = evaluator.evaluate()

    return stats
