import torch
import os
import numpy as np
import json
import copy
from torch.utils.data import DataLoader
import argparse
from datasets.vcoco import build as build_vcoco
from datasets.hico import build as build_hico
import util.misc as utils
from typing import Iterable
from pycocotools.coco import COCO

class CompoDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, sampler, collate_fn, num_workers):
        super().__init__(dataset, batch_size, sampler, collate_fn, num_workers)

    #def

def compo():
    # parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--dataset_file', default='vcoco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--hoi_path', type=str, default='data/v-coco')

    parser.add_argument('--output_dir', default='logs/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")

    args = parser.parse_args()
    dataset_train = build_vcoco('train', args)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    # data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
    #                               collate_fn=utils.collate_fn, num_workers=args.num_workers)
    dataset_train_sub1, dataset_train_sub2 = torch.utils.data.random_split(dataset_train, [int(len(dataset_train) / 2),
                                                                                           len(dataset_train)-int(len(dataset_train) / 2)])
    sampler_train_sub1 = torch.utils.data.RandomSampler(dataset_train_sub1)
    sampler_train_sub2 = torch.utils.data.RandomSampler(dataset_train_sub2)
    batch_sampler_train_sub1 = torch.utils.data.BatchSampler(sampler_train_sub1, args.batch_size, drop_last=True)
    batch_sampler_train_sub2 = torch.utils.data.BatchSampler(sampler_train_sub2, args.batch_size, drop_last=True)
    data_loader_train_sub1 = DataLoader(dataset_train_sub1, batch_sampler=batch_sampler_train_sub1,
                                        collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_train_sub2 = DataLoader(dataset_train_sub2, batch_sampler=batch_sampler_train_sub2,
                                        collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_train = zip(data_loader_train_sub1, data_loader_train_sub2)

    for data_sub1, data_sub2 in data_loader_train:
        samples = []
        targets = []
        samples.append(data_sub1[0])
        samples.append(data_sub2[0])
        targets.append(data_sub1[1])
        targets.append(data_sub2[1])
        for i in range(2):
            target_compo = copy.deepcopy(targets[i])
            num_HO_1 = target_compo[0]['verb_labels'].shape[0]
            num_HO_2 = targets[1 - i][0]['verb_labels'].shape[0]
            if num_HO_1 > num_HO_2:
                padding = torch.zeros([num_HO_1 - num_HO_2, target_compo[0]['verb_labels'].shape[1]])
                target_compo[0]['verb_labels'] = torch.cat((targets[1 - i][0]['verb_labels'], padding), 0)
            elif num_HO_1 < num_HO_2:
                target_compo[0]['verb_labels'] = targets[1 - i][0]['verb_labels'][0:num_HO_1, :]
            else:
                target_compo[0]['verb_labels'] = targets[1 - i][0]['verb_labels']
            for j in range(num_HO_1):
                obj_index = target_compo[0]['obj_labels'][j]
                verb_indices = np.argwhere(target_compo[0]['verb_labels'][j, :].numpy() == 1).reshape(-1)
                if target_compo[0]['verb_labels'].shape[1] == 117:
                    obj_vb_matrix = np.load('data/hico_20160224_det/hico_obj_vb_matrix.npy')
                    for k in range(verb_indices.shape[0]):
                        target_compo[0]['verb_labels'][j, verb_indices[k]] = obj_vb_matrix[
                            obj_index, verb_indices[k]]
                    if not target_compo[0]['verb_labels'].type(torch.uint8).any():
                        target_compo[0]['verb_labels'][j, 57] = 1
                elif target_compo[0]['verb_labels'].shape[1] == 29:
                    obj_vb_matrix = np.load('data/v-coco/vcoco_obj_vb_matrix.npy')
                    for k in range(verb_indices.shape[0]):
                        target_compo[0]['verb_labels'][j, verb_indices[k]] = obj_vb_matrix[
                            obj_index, verb_indices[k]]
            targets.append(target_compo)
        print('a')


if __name__ == '__main__':
    compo()
    '''
    hico_obj_vb_matrix = np.load('data/hico_20160224_det/hico_obj_vb_matrix.npy')
    vcoco_obj_vb_matrix = np.load('data/v-coco/vcoco_obj_vb_matrix.npy')
    vb_idx_map = [36, -1, 87, 76, -1, 41, -1, 35, 23, -1,
                  43, 49, 101, 8, 104, 9, 16, 15, -1, 109,
                  -1, -1, -1, -1, 20, 44, -1, 73, -1]
    for i in range(29):
        vb_hico = vb_idx_map[i]
        if vb_hico != -1:
            obj_vb_hico = np.array(hico_obj_vb_matrix[:, vb_hico], dtype=bool)
            obj_vb_vcoco = np.array(vcoco_obj_vb_matrix[:80, i], dtype=bool)
            obj_vb_hico_vcoco = obj_vb_hico | obj_vb_vcoco
            vcoco_obj_vb_matrix[:80, i] = np.array(obj_vb_hico_vcoco, dtype=np.float64)
    np.save('data/v-coco/vcoco_obj_vb_matrix2.npy', vcoco_obj_vb_matrix)
    obj_vb_matrix = np.load('data/v-coco/vcoco_obj_vb_matrix2.npy')
    coco = COCO('data/v-coco/data/instances_vcoco_all_2014.json')
    category_ids = coco.getCatIds()
    categories = coco.loadCats(category_ids)
    with open('data/v-coco/vcoco_vb_obj2.json') as f:
        vb_list = json.load(f)
    for i in range(29):
        obj_indices = np.argwhere(obj_vb_matrix[:, i] == 1)
        obj_indices = np.array(obj_indices, dtype=int)
        vb_list[i]['objects'] = []
        vb_list[i]['obj_indices'] = []
        for obj in obj_indices:
            obj = int(obj[0])
            if obj != 80:
                vb_list[i]['objects'].append(categories[obj]['name'])
                vb_list[i]['obj_indices'].append(obj)
    with open('data/v-coco/vcoco_vb_obj3.json', 'w') as f:
        json.dump(vb_list, f)
        '''
    '''
    with open('data/v-coco/vcoco_vb_obj2.json') as f:
        vb_list = json.load(f)
    obj_vb_matrix = np.zeros((81, 29))
    for i, vb in enumerate(vb_list):
        obj_indices = vb['obj_indices']
        verb_idx = i
        obj_vb_matrix[obj_indices, verb_idx] = 1
        if obj_indices == []:
            obj_vb_matrix[-1, verb_idx] = 1
    np.save('data/v-coco/vcoco_obj_vb_matrix.npy', obj_vb_matrix)
    
    set_list = [(0, 38), (1, 31), (1, 32), (2, 43), (2, 44), (2, 77), (4, 1), (4, 19), (4, 28), (4, 46), (4, 47),
                (4, 48),
                (4, 49), (4, 51), (4, 52), (4, 54), (4, 55), (4, 56), (5, 2), (5, 3), (5, 4), (5, 6), (5, 7), (5, 8),
                (5, 9), (5, 18), (5, 21), (6, 68), (7, 33), (8, 64), (9, 47), (9, 48), (9, 49), (9, 50), (9, 51),
                (9, 52),
                (9, 53), (9, 54), (9, 55), (9, 56), (10, 2), (10, 4), (10, 14), (10, 18), (10, 21), (10, 25), (10, 27),
                (10, 29), (10, 57), (10, 58), (10, 60), (10, 61), (10, 62), (10, 64), (11, 31), (11, 32), (11, 37),
                (11, 38), (12, 14), (12, 57), (12, 58), (12, 60), (12, 61), (13, 40), (13, 41), (13, 42), (13, 46),
                (14, 1),
                (14, 25), (14, 26), (14, 27), (14, 29), (14, 30), (14, 31), (14, 32), (14, 33), (14, 34), (14, 35),
                (14, 37), (14, 38), (14, 39), (14, 40), (14, 41), (14, 42), (14, 47), (14, 50), (14, 68), (14, 74),
                (14, 75), (14, 78), (15, 30), (15, 33), (16, 43), (16, 44), (16, 45), (18, 1), (18, 2), (18, 3),
                (18, 4),
                (18, 5), (18, 6), (18, 7), (18, 8), (18, 11), (18, 14), (18, 15), (18, 16), (18, 17), (18, 18),
                (18, 19),
                (18, 20), (18, 21), (18, 24), (18, 25), (18, 26), (18, 27), (18, 28), (18, 29), (18, 30), (18, 31),
                (18, 32), (18, 33), (18, 34), (18, 35), (18, 36), (18, 37), (18, 38), (18, 39), (18, 40), (18, 41),
                (18, 42), (18, 43), (18, 44), (18, 45), (18, 46), (18, 47), (18, 48), (18, 49), (18, 51), (18, 53),
                (18, 54), (18, 55), (18, 56), (18, 57), (18, 61), (18, 62), (18, 63), (18, 64), (18, 65), (18, 66),
                (18, 67), (18, 68), (18, 73), (18, 74), (18, 75), (18, 77), (19, 35), (19, 39), (20, 33), (21, 31),
                (21, 32), (23, 1), (23, 11), (23, 19), (23, 20), (23, 24), (23, 28), (23, 34), (23, 49), (23, 53),
                (23, 56),
                (23, 61), (23, 63), (23, 64), (23, 67), (23, 68), (23, 73), (24, 74), (25, 1), (25, 2), (25, 4),
                (25, 8),
                (25, 9), (25, 14), (25, 15), (25, 16), (25, 17), (25, 18), (25, 19), (25, 21), (25, 25), (25, 26),
                (25, 27),
                (25, 28), (25, 29), (25, 30), (25, 31), (25, 32), (25, 33), (25, 34), (25, 35), (25, 36), (25, 37),
                (25, 38), (25, 39), (25, 40), (25, 41), (25, 42), (25, 43), (25, 44), (25, 45), (25, 46), (25, 47),
                (25, 48), (25, 49), (25, 50), (25, 51), (25, 52), (25, 53), (25, 54), (25, 55), (25, 56), (25, 57),
                (25, 64), (25, 65), (25, 66), (25, 67), (25, 68), (25, 73), (25, 74), (25, 77), (25, 78), (25, 79),
                (25, 80), (26, 32), (26, 37), (28, 30), (28, 33)]
    vb_idx_map = [21, 20, 16, 4, 17, 3, 12, 25, 19, 8, 2, 10, 11, 24, 13, 14, 9, 23, 5, 6, 7, 28, 18, 26, 27, 0, 22, 1, 15]
    coco = COCO('data/v-coco/data/instances_vcoco_all_2014.json')
    category_ids = coco.getCatIds()
    categories = coco.loadCats(category_ids)
    obj_idx_dict = {}
    for i, cat in enumerate(categories):
        obj_idx_dict[cat['name']] = i
    with open('data/v-coco/vcoco_vb_obj.json') as f:
        vb_list = json.load(f)
    vb_obj_list = []
    for pair in set_list:
        vb_idx = vb_idx_map[pair[0]]
        obj_idx = pair[1]-1
        if obj_idx not in vb_list[vb_idx]['obj_indices']:
            vb_list[vb_idx]['obj_indices'].append(obj_idx)
            vb_list[vb_idx]['objects'].append(categories[obj_idx]['name'])
    with open('data/v-coco/vcoco_vb_obj2.json', 'w') as f:
        json.dump(vb_list, f)
'''
'''
    for vb in vb_list:
        vb_obj = {'verb': vb['verb'], 'objects': [], 'obj_indices': []}
        if len(vb['role_name']) == 2:
            vb_obj['verb'] = vb['action_name'] + '_' + vb['role_name'][1]
            vb_obj['objects'] = vb['include'][1]
            for obj in vb['include'][1]:
                vb_obj['obj_indices'].append(obj_idx_dict[obj])
            vb_obj_list.append(vb_obj)
        elif len(vb['role_name']) == 3:
            for j in range(2):
                vb_obj = {'verb': vb['action_name'], 'objects': [], 'obj_indices': []}
                vb_obj['verb'] = vb['action_name'] + '_' + vb['role_name'][j+1]
                vb_obj['objects'] = vb['include'][j+1]
                for obj in vb['include'][j+1]:
                    vb_obj['obj_indices'].append(obj_idx_dict[obj])
                vb_obj_list.append(vb_obj)
        else:
            vb_obj_list.append(vb_obj)
'''
'''
    with open('data/hico_20160224_det/hico_list_hoi.json') as f:
        hoi_list = json.load(f)
    obj_vb_matrix = np.zeros((80,117))
    for hoi in hoi_list:
        obj_idx = hoi['obj_index']
        verb_idx = hoi['verb_index']
        obj_vb_matrix[obj_idx,verb_idx] = 1
    np.save('data/hico_20160224_det/hico_obj_vb_matrix.npy',obj_vb_matrix)
'''
'''
    with open('data/hico_20160224_det/hico_list_hoi.json') as f:
        hoi_list = json.load(f)
    coco = COCO('data/v-coco/data/instances_vcoco_all_2014.json')
    category_ids = coco.getCatIds()
    categories = coco.loadCats(category_ids)
    obj_id_dict = {}
    obj_index_dict = {}
    for i, category in enumerate(categories):
        obj_id_dict[category['name'].replace(' ','_')]=category['id']
        obj_index_dict[category['name'].replace(' ', '_')] = i
    with open('data/hico_20160224_det/hico_list_vb.txt') as f:
        vb_list = f.readlines()
        vb_index_dict = {}
        del vb_list[0]
        del vb_list[0]
    for i,vb in enumerate(vb_list):
        vb = vb.partition('  ')[2].replace(' ','').replace('\n','')
        vb_index_dict[vb]=i
    for hoi in hoi_list:
        obj = hoi['obj']
        verb = hoi['verb']
        hoi['obj_id'] = obj_id_dict[obj]
        hoi['obj_index'] = obj_index_dict[obj]
        hoi['verb_index'] = vb_index_dict[verb]
    with open('data/hico_20160224_det/hico_list_hoi.json', 'w') as f:
        json.dump(hoi_list, f)
'''

'''
coco = COCO('data/v-coco/data/instances_vcoco_all_2014.json')
category_ids = coco.getCatIds()
categories = coco.loadCats(category_ids)
for i,cat in enumerate(categories):
    print(i,cat)
   
verb_classes = [
'hold_obj', 'stand', 'sit_instr', 'ride_instr', 'walk',
 'look_obj', 'hit_instr', 'hit_obj', 'eat_obj', 'eat_instr',
 'jump_instr', 'lay_instr', 'talk_on_phone_instr', 'carry_obj', 'throw_obj',
 'catch_obj', 'cut_instr', 'cut_obj', 'run', 'work_on_computer_instr',
 'ski_instr', 'surf_instr', 'skateboard_instr', 'smile', 'drink_instr',
 'kick_obj', 'point_instr', 'read_obj', 'snowboard_instr'
 ]
for i,cat in enumerate(verb_classes):
    print(i,cat)
'''

'''
 with open('data/hico_20160224_det/hico_list_hoi.txt') as f:
     hoi_list = f.readlines()
     hoi_list_ = []
     del hoi_list[0]
     del hoi_list[0]
 for hoi in hoi_list:
     hoi = hoi.split(' ')
     hoi_ = []
     for e in hoi:
         if e != '' and e != '\n':
             hoi_.append(e)
     hoi_dict = {'hoi_id': int(hoi_[0]), 'obj': hoi_[1], 'verb': hoi_[2]}
     hoi_list_.append(hoi_dict)
 with open('data/hico_20160224_det/hico_list_hoi.json', 'w') as f:
     json.dump(hoi_list_,f)
 '''

