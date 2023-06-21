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

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    remove: bool = False, batch_weight_mode: int = 0,
                    label_smoothing: bool = False, length: int = 0):
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
    optimizer.zero_grad()
    if isinstance(data_loader, list):
        obj_vb_matrix_hico = np.load('data/hico_20160224_det/hico_obj_vb_matrix.npy')
        obj_vb_matrix_vcoco = np.load('data/v-coco/vcoco_obj_vb_matrix.npy')
        batch_idx = 0
        for data_sub1, data_sub2 in metric_logger.log_every(data_loader, print_freq, header):
            half_bs = data_sub1[0].tensors.shape[0]
            samples = []
            sample_list = [data_sub1[0].tensors[i, :, :, :].to(device) for i in range(half_bs)] + \
                           [data_sub2[0].tensors[i, :, :, :].to(device) for i in range(half_bs)]
            sample = utils.nested_tensor_from_tensor_list(sample_list)
            samples.append(sample)
            targets = []
            target1 = [{k: v.to(device) for k, v in t.items() if k != 'filename'} for t in data_sub1[1]]
            target2 = [{k: v.to(device) for k, v in t.items() if k != 'filename'} for t in data_sub2[1]]
            targets.append(tuple(target1) + tuple(target2))

            if targets[0][0]['verb_labels'].shape[1] == 117:
                obj_vb_matrix = obj_vb_matrix_hico
            elif targets[0][0]['verb_labels'].shape[1] == 29:
                obj_vb_matrix = obj_vb_matrix_vcoco
            targets_compo = []
            for i in range(half_bs * 2):
                index = half_bs + i if i < half_bs else i - half_bs
                target_compo = copy.deepcopy(targets[0][i])
                num_HO_1 = target_compo['verb_labels'].shape[0]
                num_HO_2 = targets[0][index]['verb_labels'].shape[0]
                num_HO_max = 100
                obj_labels_compo = []
                verb_labels_compo = []
                sub_boxes_compo = []
                obj_boxes_compo = []
                #cross compo
                if num_HO_1 and num_HO_2:
                    for j in range(num_HO_1):
                        is_repeat = True
                        obj_index = target_compo['obj_labels'][j]
                        verb_labels = copy.deepcopy(targets[0][index]['verb_labels'])
                        verb_indices = np.argwhere(verb_labels.cpu().numpy() == 1)
                        for p in range(verb_indices.shape[0]):
                            verb_labels[verb_indices[p, 0], verb_indices[p, 1]] = obj_vb_matrix[
                                int(obj_index), verb_indices[p, 1]]
                        if not verb_labels.type(torch.uint8).any():
                            if obj_index != 80:
                                verb_labels = verb_labels[0, :].unsqueeze(0)
                                if verb_labels.shape[1] == 117:
                                    verb_labels[0, 57] = 1
                                verb_labels_compo.append(verb_labels)
                            else:
                                is_repeat = False
                        else:
                            not_zero = torch.zeros(verb_labels.shape[0], dtype=bool)
                            for k in range(verb_labels.shape[0]):
                                if verb_labels[k, :].type(torch.uint8).any():
                                    not_zero[k] = True
                            verb_labels = verb_labels[not_zero, :]
                            verb_labels_compo.append(verb_labels)
                        if is_repeat:
                            obj_labels_compo.append(obj_index.repeat(verb_labels.shape[0]))
                            sub_boxes_compo.append(target_compo['sub_boxes'][j,:].repeat(verb_labels.shape[0], 1))
                            obj_boxes_compo.append(target_compo['obj_boxes'][j,:].repeat(verb_labels.shape[0], 1))

                if len(obj_labels_compo):
                    cross_obj_labels = torch.cat(obj_labels_compo)[0:num_HO_max]
                    cross_verb_labels = torch.cat(verb_labels_compo, 0)[0:num_HO_max]
                    cross_sub_boxes = torch.cat(sub_boxes_compo, 0)[0:num_HO_max]
                    cross_obj_boxes = torch.cat(obj_boxes_compo, 0)[0:num_HO_max]
                    cross_matching_labels = torch.ones(target_compo['verb_labels'].shape[0], device=cross_obj_labels.device)
                else:
                    device = target_compo['obj_labels'].device
                    cross_obj_labels = torch.Tensor(0).long().to(device)
                    cross_verb_labels = torch.Tensor(0, target_compo['verb_labels'].shape[1]).to(device)
                    cross_sub_boxes = torch.Tensor(0, 4).to(device)
                    cross_obj_boxes = torch.Tensor(0, 4).to(device)
                    cross_matching_labels = torch.Tensor(0).long().to(device)

                verb_labels_compo = []
                obj_labels_compo = []
                sub_boxes_compo = []
                obj_boxes_compo = []
                # self compo
                if num_HO_1 > 1:
                    for j in range(num_HO_1):
                        is_repeat = True
                        obj_index = target_compo['obj_labels'][j]
                        verb_labels = copy.deepcopy(target_compo['verb_labels'])
                        mask = torch.ones(verb_labels.shape[0], dtype=bool)
                        mask[j] = False
                        verb_labels = verb_labels[mask, :]
                        verb_indices = np.argwhere(verb_labels.cpu().numpy() == 1)
                        for p in range(verb_indices.shape[0]):
                            verb_labels[verb_indices[p, 0], verb_indices[p, 1]] = obj_vb_matrix[
                                int(obj_index), verb_indices[p, 1]]
                        if not verb_labels.type(torch.uint8).any():
                            if obj_index != 80:
                                verb_labels = verb_labels[0, :].unsqueeze(0)
                                if verb_labels.shape[1] == 117:
                                    verb_labels[0, 57] = 1
                                verb_labels_compo.append(verb_labels)
                            else:
                                is_repeat = False
                        else:
                            not_zero = torch.zeros(verb_labels.shape[0], dtype=bool)
                            for k in range(verb_labels.shape[0]):
                                if verb_labels[k, :].type(torch.uint8).any():
                                    not_zero[k] = True
                            verb_labels = verb_labels[not_zero, :]
                            verb_labels_compo.append(verb_labels)
                        if is_repeat:
                            obj_labels_compo.append(obj_index.repeat(verb_labels.shape[0]))
                            sub_boxes_compo.append(target_compo['sub_boxes'][j, :].repeat(verb_labels.shape[0], 1))
                            obj_boxes_compo.append(target_compo['obj_boxes'][j, :].repeat(verb_labels.shape[0], 1))
                if len(obj_labels_compo):
                    self_obj_labels = torch.cat(obj_labels_compo)[0:num_HO_max]
                    self_verb_labels = torch.cat(verb_labels_compo, 0)[0:num_HO_max]
                    self_sub_boxes = torch.cat(sub_boxes_compo, 0)[0:num_HO_max]
                    self_obj_boxes = torch.cat(obj_boxes_compo, 0)[0:num_HO_max]
                    self_matching_labels = torch.ones(target_compo['verb_labels'].shape[0], device=cross_matching_labels.device)
                    target_compo['obj_labels'] = torch.cat((cross_obj_labels, self_obj_labels))
                    target_compo['verb_labels'] = torch.cat((cross_verb_labels, self_verb_labels))
                    target_compo['sub_boxes'] = torch.cat((cross_sub_boxes, self_sub_boxes))
                    target_compo['obj_boxes'] = torch.cat((cross_obj_boxes, self_obj_boxes))
                    target_compo['matching_labels'] = torch.cat((cross_matching_labels, self_matching_labels))
                else:
                    target_compo['obj_labels'] = cross_obj_labels
                    target_compo['verb_labels'] = cross_verb_labels
                    target_compo['sub_boxes'] = cross_sub_boxes
                    target_compo['obj_boxes'] = cross_obj_boxes
                    target_compo['matching_labels'] = cross_matching_labels
                if label_smoothing:
                    target_compo['verb_labels'] = target_compo['verb_labels'].clamp(0.1, 0.9)
                targets_compo.append(target_compo)
            targets.append(tuple(targets_compo))

            outputs = []
            human_out = None
            obj_out = None
            hopd_out = None
            if model.module.separate:
                output, human_out, obj_out, interaction_decoder_out = model(samples)
            else:
                output, hopd_out, interaction_decoder_out = model(samples)
            outputs.append(output)
            output_without_aux = {k: v for k, v in output.items() if k != 'aux_outputs'}
            indices = criterion.matcher(output_without_aux, targets[0])
            output_compo = model.module.forward_compo(human_out, obj_out, hopd_out, interaction_decoder_out, indices)
            outputs.append(output_compo)
            batch_weight_list = torch.Tensor([[0.5, 0.5],
                                 [0.75, 0.25],
                                 [0.9, 0.1],
                                 [1, 0],
                                 [0, 1]])
            batch_weight = batch_weight_list[batch_weight_mode]
            losses_list = []
            losses_avg = 0
            is_uctt = False
            for i in range(len(outputs)):
                loss_dict = criterion(outputs[i], targets[i])
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                losses_list.append(losses)
                if 'loss_uctt' in loss_dict.keys():
                    is_uctt = True
                    batch_weight[i] = -loss_dict['loss_uctt']

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                              for k, v in loss_dict_reduced.items()}
                loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                            for k, v in loss_dict_reduced.items() if k in weight_dict}
                losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

                loss_value = losses_reduced_scaled.item()
                # print(loss_dict['loss_verb_ce'],loss_dict_reduced['loss_verb_ce'],loss_dict_reduced_scaled['loss_verb_ce'])

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    print(loss_dict_reduced)
                    sys.exit(1)

            #print(batch_weight)
            #batch_weight = batch_weight / batch_weight.sum()
            if is_uctt:
                batch_weight = torch.softmax(batch_weight,dim=0)
            #print(batch_weight)
            #batch_weight_sum = sum(batch_weight)
            for i in range(len(outputs)):
                #batch_weight[i] = batch_weight[i] / batch_weight_sum
                losses_avg += batch_weight[i] * losses_list[i]
            accum_iter = int(8 / half_bs)
            losses_avg = losses_avg / accum_iter
            losses_avg.backward()
            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == length):
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                optimizer.zero_grad()
            batch_idx += 1

            metric_logger.update(batch_weight_orig=batch_weight[0].item(),batch_weight_compo=batch_weight[1].item())
            metric_logger.update(loss_orig=losses_list[0].item(), loss_compo=losses_list[1].item())
            metric_logger.update(loss=losses_avg, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
            if hasattr(criterion, 'loss_labels'):
                metric_logger.update(class_error=loss_dict_reduced['class_error'])
            else:
                metric_logger.update(obj_class_error=loss_dict_reduced['obj_class_error'])
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    else:
        batch_idx = 0
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

            bs = samples.tensors.shape[0]
            accum_iter = int(16 / bs)
            losses = losses / accum_iter
            losses.backward()
            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == length):
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                optimizer.zero_grad()
            batch_idx += 1

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
