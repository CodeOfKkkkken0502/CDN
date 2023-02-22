from pathlib import Path
from PIL import Image
import json
from collections import defaultdict
import numpy as np

import torch
import torch.utils.data
import torchvision

import datasets.transforms as T

hico_unseen_index = {
    "default": [],
    # start from 0
    "rare_first": [509, 279, 280, 402, 504, 286, 499, 498, 289, 485, 303, 311, 325, 439, 351, 358, 66, 427, 379, 418,
                   70, 416,
                   389, 90, 395, 76, 397, 84, 135, 262, 401, 592, 560, 586, 548, 593, 526, 181, 257, 539, 535, 260, 596,
                   345, 189,
                   205, 206, 429, 179, 350, 405, 522, 449, 261, 255, 546, 547, 44, 22, 334, 599, 239, 315, 317, 229,
                   158, 195,
                   238, 364, 222, 281, 149, 399, 83, 127, 254, 398, 403, 555, 552, 520, 531, 440, 436, 482, 274, 8, 188,
                   216, 597,
                   77, 407, 556, 469, 474, 107, 390, 410, 27, 381, 463, 99, 184, 100, 292, 517, 80, 333, 62, 354, 104,
                   55, 50,
                   198, 168, 391, 192, 595, 136, 581],  # 120
    "non_rare_first": [38, 41, 20, 18, 245, 11, 19, 154, 459, 42, 155, 139, 60, 461, 577, 153, 582, 89, 141, 576, 75,
                       212, 472, 61,
                       457, 146, 208, 94, 471, 131, 248, 544, 515, 566, 370, 481, 226, 250, 470, 323, 169, 480, 479,
                       230, 385, 73,
                       159, 190, 377, 176, 249, 371, 284, 48, 583, 53, 162, 140, 185, 106, 294, 56, 320, 152, 374, 338,
                       29, 594, 346,
                       456, 589, 45, 23, 67, 478, 223, 493, 228, 240, 215, 91, 115, 337, 559, 7, 218, 518, 297, 191,
                       266, 304, 6, 572,
                       529, 312, 9, 308, 417, 197, 193, 163, 455, 25, 54, 575, 446, 387, 483, 534, 340, 508, 110, 329,
                       246, 173, 506,
                       383, 93, 516, 64],  # 120
    "unseen_object": [111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
                      126, 127, 128, 224, 225, 226, 227, 228, 229, 230, 231, 290, 291, 292, 293,
                      294, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 336, 337,
                      338, 339, 340, 341, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428,
                      429, 430, 431, 432, 433, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462,
                      463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 533, 534, 535, 536,
                      537, 558, 559, 560, 561, 595, 596, 597, 598, 599],  # 100
    "unseen_verb": [4, 6, 12, 15, 18, 25, 34, 38, 40, 49, 58, 60, 68, 69, 72, 73, 77, 82, 96, 97, 104, 113, 116, 118,
                    122, 129, 139, 147,
                    150, 153, 165, 166, 172, 175, 176, 181, 190, 202, 210, 212, 219, 227, 228, 233, 235, 243, 298, 313,
                    315, 320, 326, 336,
                    342, 345, 354, 372, 401, 404, 409, 431, 436, 459, 466, 470, 472, 479, 481, 488, 491, 494, 498, 504,
                    519, 523, 535, 536,
                    541, 544, 562, 565, 569, 572, 591, 595]
    # 84, 20 unseen verbs: [41, 100, 99, 91, 34, 42, 97, 84, 26, 106, 38, 56, 92, 79, 19, 76, 80, 2, 114, 62]
}

hico_triplet_labels = [(4, 4), (17, 4), (25, 4), (30, 4), (41, 4), (52, 4), (76, 4), (87, 4), (111, 4), (57, 4),
                            (8, 1), (36, 1), (41, 1), (43, 1), (37, 1), (62, 1), (71, 1), (75, 1), (76, 1), (87, 1),
                            (98, 1), (110, 1), (111, 1), (57, 1), (10, 14), (26, 14), (36, 14), (65, 14), (74, 14),
                            (112, 14),
                            (57, 14), (4, 8), (21, 8), (25, 8), (41, 8), (43, 8), (47, 8), (75, 8), (76, 8), (77, 8),
                            (79, 8), (87, 8), (93, 8), (105, 8), (111, 8), (57, 8), (8, 39), (20, 39), (36, 39),
                            (41, 39),
                            (48, 39), (58, 39), (69, 39), (57, 39), (4, 5), (17, 5), (21, 5), (25, 5), (41, 5), (52, 5),
                            (76, 5), (87, 5), (111, 5), (113, 5), (57, 5), (4, 2), (17, 2), (21, 2), (38, 2), (41, 2),
                            (43, 2), (52, 2), (62, 2), (76, 2), (111, 2), (57, 2), (22, 15), (26, 15), (36, 15),
                            (39, 15),
                            (45, 15), (65, 15), (80, 15), (111, 15), (10, 15), (57, 15), (8, 56), (36, 56), (49, 56),
                            (87, 56),
                            (93, 56), (57, 56), (8, 57), (49, 57), (87, 57), (57, 57), (26, 19), (34, 19), (36, 19),
                            (39, 19),
                            (45, 19), (46, 19), (55, 19), (65, 19), (76, 19), (110, 19), (57, 19), (12, 60), (24, 60),
                            (86, 60),
                            (57, 60), (8, 16), (22, 16), (26, 16), (33, 16), (36, 16), (38, 16), (39, 16), (41, 16),
                            (45, 16),
                            (65, 16), (78, 16), (80, 16), (98, 16), (107, 16), (110, 16), (111, 16), (10, 16), (57, 16),
                            (26, 17),
                            (33, 17), (36, 17), (39, 17), (43, 17), (45, 17), (52, 17), (37, 17), (65, 17), (72, 17),
                            (76, 17),
                            (78, 17), (98, 17), (107, 17), (110, 17), (111, 17), (57, 17), (36, 3), (41, 3), (43, 3),
                            (37, 3),
                            (62, 3), (71, 3), (72, 3), (76, 3), (87, 3), (98, 3), (108, 3), (110, 3), (111, 3), (57, 3),
                            (8, 0), (31, 0), (36, 0), (39, 0), (45, 0), (92, 0), (100, 0), (102, 0), (48, 0), (57, 0),
                            (8, 58), (36, 58), (38, 58), (57, 58), (8, 18), (26, 18), (34, 18), (36, 18), (39, 18),
                            (45, 18),
                            (65, 18), (76, 18), (83, 18), (110, 18), (111, 18), (57, 18), (4, 6), (21, 6), (25, 6),
                            (52, 6),
                            (76, 6), (87, 6), (111, 6), (57, 6), (13, 62), (75, 62), (112, 62), (57, 62), (7, 47),
                            (15, 47),
                            (23, 47), (36, 47), (41, 47), (64, 47), (66, 47), (89, 47), (111, 47), (57, 47), (8, 24),
                            (36, 24),
                            (41, 24), (58, 24), (114, 24), (57, 24), (7, 46), (8, 46), (15, 46), (23, 46), (36, 46),
                            (41, 46),
                            (64, 46), (66, 46), (89, 46), (57, 46), (5, 34), (8, 34), (36, 34), (84, 34), (99, 34),
                            (104, 34),
                            (115, 34), (57, 34), (36, 35), (114, 35), (57, 35), (26, 21), (40, 21), (112, 21), (57, 21),
                            (12, 59),
                            (49, 59), (87, 59), (57, 59), (41, 13), (49, 13), (87, 13), (57, 13), (8, 73), (36, 73),
                            (58, 73),
                            (73, 73), (57, 73), (36, 45), (96, 45), (111, 45), (48, 45), (57, 45), (15, 50), (23, 50),
                            (36, 50),
                            (89, 50), (96, 50), (111, 50), (57, 50), (3, 55), (8, 55), (15, 55), (23, 55), (36, 55),
                            (51, 55),
                            (54, 55), (67, 55), (57, 55), (8, 51), (14, 51), (15, 51), (23, 51), (36, 51), (64, 51),
                            (89, 51),
                            (96, 51), (111, 51), (57, 51), (8, 67), (36, 67), (73, 67), (75, 67), (101, 67), (103, 67),
                            (57, 67),
                            (11, 74), (36, 74), (75, 74), (82, 74), (57, 74), (8, 41), (20, 41), (36, 41), (41, 41),
                            (69, 41),
                            (85, 41), (89, 41), (27, 41), (111, 41), (57, 41), (7, 54), (8, 54), (23, 54), (36, 54),
                            (54, 54),
                            (67, 54), (89, 54), (57, 54), (26, 20), (36, 20), (38, 20), (39, 20), (45, 20), (37, 20),
                            (65, 20),
                            (76, 20), (110, 20), (111, 20), (112, 20), (57, 20), (39, 10), (41, 10), (58, 10), (61, 10),
                            (57, 10),
                            (36, 42), (50, 42), (95, 42), (48, 42), (111, 42), (57, 42), (2, 29), (9, 29), (36, 29),
                            (90, 29),
                            (104, 29), (57, 29), (26, 23), (45, 23), (65, 23), (76, 23), (112, 23), (57, 23), (36, 78),
                            (59, 78),
                            (75, 78), (57, 78), (8, 26), (36, 26), (41, 26), (57, 26), (8, 52), (14, 52), (15, 52),
                            (23, 52),
                            (36, 52), (54, 52), (57, 52), (8, 66), (12, 66), (36, 66), (109, 66), (57, 66), (1, 33),
                            (8, 33),
                            (30, 33), (36, 33), (41, 33), (47, 33), (70, 33), (57, 33), (16, 43), (36, 43), (95, 43),
                            (111, 43),
                            (115, 43), (48, 43), (57, 43), (36, 63), (58, 63), (73, 63), (75, 63), (109, 63), (57, 63),
                            (12, 68),
                            (58, 68), (59, 68), (57, 68), (13, 64), (36, 64), (75, 64), (57, 64), (7, 49), (15, 49),
                            (23, 49),
                            (36, 49), (41, 49), (64, 49), (66, 49), (91, 49), (111, 49), (57, 49), (12, 69), (36, 69),
                            (41, 69),
                            (58, 69), (75, 69), (59, 69), (57, 69), (11, 12), (63, 12), (75, 12), (57, 12), (7, 53),
                            (8, 53),
                            (14, 53), (15, 53), (23, 53), (36, 53), (54, 53), (67, 53), (88, 53), (89, 53), (57, 53),
                            (12, 72),
                            (36, 72), (56, 72), (58, 72), (57, 72), (36, 65), (68, 65), (99, 65), (57, 65), (8, 48),
                            (14, 48),
                            (15, 48), (23, 48), (36, 48), (54, 48), (57, 48), (16, 76), (36, 76), (58, 76), (57, 76),
                            (12, 71),
                            (75, 71), (111, 71), (57, 71), (8, 36), (28, 36), (32, 36), (36, 36), (43, 36), (67, 36),
                            (76, 36),
                            (87, 36), (93, 36), (57, 36), (0, 30), (8, 30), (36, 30), (41, 30), (43, 30), (67, 30),
                            (75, 30),
                            (76, 30), (93, 30), (114, 30), (57, 30), (0, 31), (8, 31), (32, 31), (36, 31), (43, 31),
                            (76, 31),
                            (93, 31), (114, 31), (57, 31), (36, 44), (48, 44), (111, 44), (85, 44), (57, 44), (2, 32),
                            (8, 32),
                            (9, 32), (19, 32), (35, 32), (36, 32), (41, 32), (44, 32), (67, 32), (81, 32), (84, 32),
                            (90, 32),
                            (104, 32), (57, 32), (36, 11), (94, 11), (97, 11), (57, 11), (8, 28), (18, 28), (36, 28),
                            (39, 28),
                            (52, 28), (58, 28), (60, 28), (67, 28), (116, 28), (57, 28), (8, 37), (18, 37), (36, 37),
                            (41, 37),
                            (43, 37), (49, 37), (52, 37), (76, 37), (93, 37), (87, 37), (111, 37), (57, 37), (8, 77),
                            (36, 77),
                            (39, 77), (45, 77), (57, 77), (8, 38), (36, 38), (41, 38), (99, 38), (57, 38), (0, 27),
                            (15, 27),
                            (36, 27), (41, 27), (70, 27), (105, 27), (114, 27), (57, 27), (36, 70), (59, 70), (75, 70),
                            (57, 70),
                            (12, 61), (29, 61), (58, 61), (75, 61), (87, 61), (93, 61), (111, 61), (57, 61), (6, 79),
                            (36, 79),
                            (111, 79), (57, 79), (42, 9), (75, 9), (94, 9), (97, 9), (57, 9), (17, 7), (21, 7), (41, 7),
                            (52, 7), (75, 7), (76, 7), (87, 7), (111, 7), (57, 7), (8, 25), (36, 25), (53, 25),
                            (58, 25),
                            (75, 25), (82, 25), (94, 25), (57, 25), (36, 75), (54, 75), (61, 75), (57, 75), (27, 40),
                            (36, 40),
                            (85, 40), (106, 40), (48, 40), (111, 40), (57, 40), (26, 22), (36, 22), (65, 22), (112, 22),
                            (57, 22)]

class HICODetection(torch.utils.data.Dataset):

    def __init__(self, img_set, img_folder, anno_file, transforms, num_queries, zero_shot_type):
        self.img_set = img_set
        self.img_folder = img_folder
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        self._transforms = transforms

        self.num_queries = num_queries

        self._valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                               14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                               24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                               37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                               58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                               72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                               82, 84, 85, 86, 87, 88, 89, 90)
        self._valid_verb_ids = list(range(1, 118))
        self.unseen_index = hico_unseen_index.get(zero_shot_type, [])
        self.hico_triplet_labels = hico_triplet_labels

        if img_set == 'train' and len(self.unseen_index) != 0:
            self.ids = []
            for idx, img_anno in enumerate(self.annotations):
                new_img_anno = []
                skip_pair = []
                for hoi in img_anno['hoi_annotation']:
                    if hoi['hoi_category_id'] - 1 in self.unseen_index:
                        skip_pair.append((hoi['subject_id'], hoi['object_id']))
                for hoi in img_anno['hoi_annotation']:
                    if hoi['subject_id'] >= len(img_anno['annotations']) or hoi['object_id'] >= len(
                            img_anno['annotations']):
                        new_img_anno = []
                        break
                    if (hoi['subject_id'], hoi['object_id']) not in skip_pair:
                        new_img_anno.append(hoi)
                if len(new_img_anno) > 0:
                    self.ids.append(idx)
                    img_anno['hoi_annotation'] = new_img_anno
        else:
            self.ids = list(range(len(self.annotations)))
        print("{} contains {} images".format(img_set, len(self.ids)))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_anno = self.annotations[self.ids[idx]]

        img = Image.open(self.img_folder / img_anno['file_name']).convert('RGB')
        w, h = img.size

        if self.img_set == 'train' and len(img_anno['annotations']) > self.num_queries:
            img_anno['annotations'] = img_anno['annotations'][:self.num_queries]

        boxes = [obj['bbox'] for obj in img_anno['annotations']]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        if self.img_set == 'train':
            classes = [(i, self._valid_obj_ids.index(obj['category_id'])) for i, obj in enumerate(img_anno['annotations'])]
        else:
            classes = [self._valid_obj_ids.index(obj['category_id']) for obj in img_anno['annotations']]
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size'] = torch.as_tensor([int(h), int(w)])
        if self.img_set == 'train':
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            classes = classes[keep]

            target['boxes'] = boxes
            target['labels'] = classes
            target['iscrowd'] = torch.tensor([0 for _ in range(boxes.shape[0])])
            target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            if self._transforms is not None:
                img, target = self._transforms(img, target)

            kept_box_indices = [label[0] for label in target['labels']]

            target['labels'] = target['labels'][:, 1]

            obj_labels, verb_labels, sub_boxes, obj_boxes = [], [], [], []
            sub_obj_pairs = []
            for hoi in img_anno['hoi_annotation']:
                if hoi['subject_id'] not in kept_box_indices or hoi['object_id'] not in kept_box_indices:
                    continue
                sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
                if sub_obj_pair in sub_obj_pairs:
                    verb_labels[sub_obj_pairs.index(sub_obj_pair)][self._valid_verb_ids.index(hoi['category_id'])] = 1
                else:
                    sub_obj_pairs.append(sub_obj_pair)
                    obj_labels.append(target['labels'][kept_box_indices.index(hoi['object_id'])])
                    verb_label = [0 for _ in range(len(self._valid_verb_ids))]
                    verb_label[self._valid_verb_ids.index(hoi['category_id'])] = 1
                    sub_box = target['boxes'][kept_box_indices.index(hoi['subject_id'])]
                    obj_box = target['boxes'][kept_box_indices.index(hoi['object_id'])]
                    verb_labels.append(verb_label)
                    sub_boxes.append(sub_box)
                    obj_boxes.append(obj_box)

            target['filename'] = img_anno['file_name']
            if len(sub_obj_pairs) == 0:
                target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['verb_labels'] = torch.zeros((0, len(self._valid_verb_ids)), dtype=torch.float32)
                target['sub_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['obj_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['matching_labels'] = torch.zeros((0,), dtype=torch.int64)
            else:
                target['obj_labels'] = torch.stack(obj_labels)
                target['verb_labels'] = torch.as_tensor(verb_labels, dtype=torch.float32)
                target['sub_boxes'] = torch.stack(sub_boxes)
                target['obj_boxes'] = torch.stack(obj_boxes)
                target['matching_labels'] = torch.ones_like(target['obj_labels'])
        else:
            target['filename'] = img_anno['file_name']
            target['boxes'] = boxes
            target['labels'] = classes
            target['id'] = idx

            if self._transforms is not None:
                img, _ = self._transforms(img, None)

            hois = []
            for hoi in img_anno['hoi_annotation']:
                hois.append((hoi['subject_id'], hoi['object_id'], self._valid_verb_ids.index(hoi['category_id'])))
            target['hois'] = torch.as_tensor(hois, dtype=torch.int64)

        return img, target

    def set_rare_hois(self, anno_file):
        with open(anno_file, 'r') as f:
            annotations = json.load(f)

        if len(self.unseen_index) == 0:
            # no unseen category, use rare to evaluate
            counts = defaultdict(lambda: 0)
            for img_anno in annotations:
                hois = img_anno['hoi_annotation']
                bboxes = img_anno['annotations']
                for hoi in hois:
                    triplet = (self._valid_verb_ids.index(hoi['category_id']),
                               self._valid_obj_ids.index(bboxes[hoi['object_id']]['category_id'])
                               )
                    counts[triplet] += 1
            self.rare_triplets = []
            self.non_rare_triplets = []
            for triplet, count in counts.items():
                if count < 10:
                    self.rare_triplets.append(triplet)
                else:
                    self.non_rare_triplets.append(triplet)
            print("rare:{}, non-rare:{}".format(len(self.rare_triplets), len(self.non_rare_triplets)))
        else:
            self.rare_triplets = []
            self.non_rare_triplets = []
            for img_anno in annotations:
                hois = img_anno['hoi_annotation']
                bboxes = img_anno['annotations']
                for hoi in hois:
                    triplet = (self._valid_verb_ids.index(hoi['category_id']),
                               self._valid_obj_ids.index(bboxes[hoi['object_id']]['category_id']))
                    if self.hico_triplet_labels.index(triplet) in self.unseen_index:
                        self.rare_triplets.append(triplet)
                    else:
                        self.non_rare_triplets.append(triplet)
            print("unseen:{}, seen:{}".format(len(self.rare_triplets), len(self.non_rare_triplets)))

    def load_correct_mat(self, path):
        self.correct_mat = np.load(path)


def make_hico_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(.4, .4, .4),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.hoi_path)
    assert root.exists(), f'provided HOI path {root} does not exist'
    PATHS = {
        'train': (root / 'images' / 'train2015', root / 'annotations' / 'trainval_hico.json'),
        'val': (root / 'images' / 'test2015', root / 'annotations' / 'test_hico.json')
    }
    CORRECT_MAT_PATH = root / 'annotations' / 'corre_hico.npy'

    img_folder, anno_file = PATHS[image_set]
    dataset = HICODetection(image_set, img_folder, anno_file, transforms=make_hico_transforms(image_set),
                            num_queries=args.num_queries, zero_shot_type=args.zero_shot_type)
    if image_set == 'val':
        dataset.set_rare_hois(PATHS['train'][1])
        dataset.load_correct_mat(CORRECT_MAT_PATH)
    return dataset
