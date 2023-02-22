import numpy as np
from collections import defaultdict
import os, cv2, json


class HICOEvaluator():
    def __init__(self, preds, gts, rare_triplets, non_rare_triplets, correct_mat, args):
        self.overlap_iou = 0.5
        self.max_hois = 100

        self.zero_shot_type = args.zero_shot_type

        self.use_nms_filter = args.use_nms_filter
        self.thres_nms = args.thres_nms
        self.nms_alpha = args.nms_alpha
        self.nms_beta = args.nms_beta

        self.rare_triplets = rare_triplets
        self.non_rare_triplets = non_rare_triplets

        self.fp = defaultdict(list)
        self.tp = defaultdict(list)
        self.score = defaultdict(list)
        self.sum_gts = defaultdict(lambda: 0)
        self.gt_triplets = []
        self.hico_triplet_labels = [(4, 4), (17, 4), (25, 4), (30, 4), (41, 4), (52, 4), (76, 4), (87, 4), (111, 4), (57, 4),
                                    (8, 1), (36, 1), (41, 1), (43, 1), (37, 1), (62, 1), (71, 1), (75, 1), (76, 1), (87, 1),
                                    (98, 1), (110, 1), (111, 1), (57, 1), (10, 14), (26, 14), (36, 14), (65, 14), (74, 14), (112, 14),
                                    (57, 14), (4, 8), (21, 8), (25, 8), (41, 8), (43, 8), (47, 8), (75, 8), (76, 8), (77, 8),
                                    (79, 8), (87, 8), (93, 8), (105, 8), (111, 8), (57, 8), (8, 39), (20, 39), (36, 39), (41, 39),
                                    (48, 39), (58, 39), (69, 39), (57, 39), (4, 5), (17, 5), (21, 5), (25, 5), (41, 5), (52, 5),
                                    (76, 5), (87, 5), (111, 5), (113, 5), (57, 5), (4, 2), (17, 2), (21, 2), (38, 2), (41, 2),
                                    (43, 2), (52, 2), (62, 2), (76, 2), (111, 2), (57, 2), (22, 15), (26, 15), (36, 15), (39, 15),
                                    (45, 15), (65, 15), (80, 15), (111, 15), (10, 15), (57, 15), (8, 56), (36, 56), (49, 56), (87, 56),
                                    (93, 56), (57, 56), (8, 57), (49, 57), (87, 57), (57, 57), (26, 19), (34, 19), (36, 19), (39, 19),
                                    (45, 19), (46, 19), (55, 19), (65, 19), (76, 19), (110, 19), (57, 19), (12, 60), (24, 60), (86, 60),
                                    (57, 60), (8, 16), (22, 16), (26, 16), (33, 16), (36, 16), (38, 16), (39, 16), (41, 16), (45, 16),
                                    (65, 16), (78, 16), (80, 16), (98, 16), (107, 16), (110, 16), (111, 16), (10, 16), (57, 16), (26, 17),
                                    (33, 17), (36, 17), (39, 17), (43, 17), (45, 17), (52, 17), (37, 17), (65, 17), (72, 17), (76, 17),
                                    (78, 17), (98, 17), (107, 17), (110, 17), (111, 17), (57, 17), (36, 3), (41, 3), (43, 3), (37, 3),
                                    (62, 3), (71, 3), (72, 3), (76, 3), (87, 3), (98, 3), (108, 3), (110, 3), (111, 3), (57, 3),
                                    (8, 0), (31, 0), (36, 0), (39, 0), (45, 0), (92, 0), (100, 0), (102, 0), (48, 0), (57, 0),
                                    (8, 58), (36, 58), (38, 58), (57, 58), (8, 18), (26, 18), (34, 18), (36, 18), (39, 18), (45, 18),
                                    (65, 18), (76, 18), (83, 18), (110, 18), (111, 18), (57, 18), (4, 6), (21, 6), (25, 6), (52, 6),
                                    (76, 6), (87, 6), (111, 6), (57, 6), (13, 62), (75, 62), (112, 62), (57, 62), (7, 47), (15, 47),
                                    (23, 47), (36, 47), (41, 47), (64, 47), (66, 47), (89, 47), (111, 47), (57, 47), (8, 24), (36, 24),
                                    (41, 24), (58, 24), (114, 24), (57, 24), (7, 46), (8, 46), (15, 46), (23, 46), (36, 46), (41, 46),
                                    (64, 46), (66, 46), (89, 46), (57, 46), (5, 34), (8, 34), (36, 34), (84, 34), (99, 34), (104, 34),
                                    (115, 34), (57, 34), (36, 35), (114, 35), (57, 35), (26, 21), (40, 21), (112, 21), (57, 21), (12, 59),
                                    (49, 59), (87, 59), (57, 59), (41, 13), (49, 13), (87, 13), (57, 13), (8, 73), (36, 73), (58, 73),
                                    (73, 73), (57, 73), (36, 45), (96, 45), (111, 45), (48, 45), (57, 45), (15, 50), (23, 50), (36, 50),
                                    (89, 50), (96, 50), (111, 50), (57, 50), (3, 55), (8, 55), (15, 55), (23, 55), (36, 55), (51, 55),
                                    (54, 55), (67, 55), (57, 55), (8, 51), (14, 51), (15, 51), (23, 51), (36, 51), (64, 51), (89, 51),
                                    (96, 51), (111, 51), (57, 51), (8, 67), (36, 67), (73, 67), (75, 67), (101, 67), (103, 67), (57, 67),
                                    (11, 74), (36, 74), (75, 74), (82, 74), (57, 74), (8, 41), (20, 41), (36, 41), (41, 41), (69, 41),
                                    (85, 41), (89, 41), (27, 41), (111, 41), (57, 41), (7, 54), (8, 54), (23, 54), (36, 54), (54, 54),
                                    (67, 54), (89, 54), (57, 54), (26, 20), (36, 20), (38, 20), (39, 20), (45, 20), (37, 20), (65, 20),
                                    (76, 20), (110, 20), (111, 20), (112, 20), (57, 20), (39, 10), (41, 10), (58, 10), (61, 10), (57, 10),
                                    (36, 42), (50, 42), (95, 42), (48, 42), (111, 42), (57, 42), (2, 29), (9, 29), (36, 29), (90, 29),
                                    (104, 29), (57, 29), (26, 23), (45, 23), (65, 23), (76, 23), (112, 23), (57, 23), (36, 78), (59, 78),
                                    (75, 78), (57, 78), (8, 26), (36, 26), (41, 26), (57, 26), (8, 52), (14, 52), (15, 52), (23, 52),
                                    (36, 52), (54, 52), (57, 52), (8, 66), (12, 66), (36, 66), (109, 66), (57, 66), (1, 33), (8, 33),
                                    (30, 33), (36, 33), (41, 33), (47, 33), (70, 33), (57, 33), (16, 43), (36, 43), (95, 43), (111, 43),
                                    (115, 43), (48, 43), (57, 43), (36, 63), (58, 63), (73, 63), (75, 63), (109, 63), (57, 63), (12, 68),
                                    (58, 68), (59, 68), (57, 68), (13, 64), (36, 64), (75, 64), (57, 64), (7, 49), (15, 49), (23, 49),
                                    (36, 49), (41, 49), (64, 49), (66, 49), (91, 49), (111, 49), (57, 49), (12, 69), (36, 69), (41, 69),
                                    (58, 69), (75, 69), (59, 69), (57, 69), (11, 12), (63, 12), (75, 12), (57, 12), (7, 53), (8, 53),
                                    (14, 53), (15, 53), (23, 53), (36, 53), (54, 53), (67, 53), (88, 53), (89, 53), (57, 53), (12, 72),
                                    (36, 72), (56, 72), (58, 72), (57, 72), (36, 65), (68, 65), (99, 65), (57, 65), (8, 48), (14, 48),
                                    (15, 48), (23, 48), (36, 48), (54, 48), (57, 48), (16, 76), (36, 76), (58, 76), (57, 76), (12, 71),
                                    (75, 71), (111, 71), (57, 71), (8, 36), (28, 36), (32, 36), (36, 36), (43, 36), (67, 36), (76, 36),
                                    (87, 36), (93, 36), (57, 36), (0, 30), (8, 30), (36, 30), (41, 30), (43, 30), (67, 30), (75, 30),
                                    (76, 30), (93, 30), (114, 30), (57, 30), (0, 31), (8, 31), (32, 31), (36, 31), (43, 31), (76, 31),
                                    (93, 31), (114, 31), (57, 31), (36, 44), (48, 44), (111, 44), (85, 44), (57, 44), (2, 32), (8, 32),
                                    (9, 32), (19, 32), (35, 32), (36, 32), (41, 32), (44, 32), (67, 32), (81, 32), (84, 32), (90, 32),
                                    (104, 32), (57, 32), (36, 11), (94, 11), (97, 11), (57, 11), (8, 28), (18, 28), (36, 28), (39, 28),
                                    (52, 28), (58, 28), (60, 28), (67, 28), (116, 28), (57, 28), (8, 37), (18, 37), (36, 37), (41, 37),
                                    (43, 37), (49, 37), (52, 37), (76, 37), (93, 37), (87, 37), (111, 37), (57, 37), (8, 77), (36, 77),
                                    (39, 77), (45, 77), (57, 77), (8, 38), (36, 38), (41, 38), (99, 38), (57, 38), (0, 27), (15, 27),
                                    (36, 27), (41, 27), (70, 27), (105, 27), (114, 27), (57, 27), (36, 70), (59, 70), (75, 70), (57, 70),
                                    (12, 61), (29, 61), (58, 61), (75, 61), (87, 61), (93, 61), (111, 61), (57, 61), (6, 79), (36, 79),
                                    (111, 79), (57, 79), (42, 9), (75, 9), (94, 9), (97, 9), (57, 9), (17, 7), (21, 7), (41, 7),
                                    (52, 7), (75, 7), (76, 7), (87, 7), (111, 7), (57, 7), (8, 25), (36, 25), (53, 25), (58, 25),
                                    (75, 25), (82, 25), (94, 25), (57, 25), (36, 75), (54, 75), (61, 75), (57, 75), (27, 40), (36, 40),
                                    (85, 40), (106, 40), (48, 40), (111, 40), (57, 40), (26, 22), (36, 22), (65, 22), (112, 22), (57, 22)]

        self.preds = []
        for index, img_preds in enumerate(preds):
            img_preds = {k: v.to('cpu').numpy() for k, v in img_preds.items()}
            bboxes = [{'bbox': list(bbox), 'category_id': label} for bbox, label in zip(img_preds['boxes'], img_preds['labels'])]
            hoi_scores = img_preds['verb_scores']
            verb_labels = np.tile(np.arange(hoi_scores.shape[1]), (hoi_scores.shape[0], 1))
            subject_ids = np.tile(img_preds['sub_ids'], (hoi_scores.shape[1], 1)).T
            object_ids = np.tile(img_preds['obj_ids'], (hoi_scores.shape[1], 1)).T

            hoi_scores = hoi_scores.ravel()
            verb_labels = verb_labels.ravel()
            subject_ids = subject_ids.ravel()
            object_ids = object_ids.ravel()

            if len(subject_ids) > 0:
                object_labels = np.array([bboxes[object_id]['category_id'] for object_id in object_ids])
                masks = correct_mat[verb_labels, object_labels]
                hoi_scores *= masks

                hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score} for
                        subject_id, object_id, category_id, score in zip(subject_ids, object_ids, verb_labels, hoi_scores)]
                hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
                hois = hois[:self.max_hois]
            else:
                hois = []

            filename = gts[index]['filename']
            self.preds.append({
                'filename':filename,
                'predictions': bboxes,
                'hoi_prediction': hois
            })


        if self.use_nms_filter:
            self.preds = self.triplet_nms_filter(self.preds)


        self.gts = []
        for i, img_gts in enumerate(gts):
            filename = img_gts['filename']
            img_gts = {k: v.to('cpu').numpy() for k, v in img_gts.items() if k != 'id' and k != 'filename'}
            self.gts.append({
                'filename':filename,
                'annotations': [{'bbox': list(bbox), 'category_id': label} for bbox, label in zip(img_gts['boxes'], img_gts['labels'])],
                'hoi_annotation': [{'subject_id': hoi[0], 'object_id': hoi[1], 'category_id': hoi[2]} for hoi in img_gts['hois']]
            })
            for hoi in self.gts[-1]['hoi_annotation']:
                triplet = (hoi['category_id'],self.gts[-1]['annotations'][hoi['object_id']]['category_id'])

                if triplet not in self.gt_triplets:
                    self.gt_triplets.append(triplet)

                self.sum_gts[triplet] += 1


        with open(args.json_file, 'w') as f:
            f.write(json.dumps(str({'preds':self.preds, 'gts':self.gts})))



    def evaluate(self):
        for img_preds, img_gts in zip(self.preds, self.gts):
            pred_bboxes = img_preds['predictions']
            gt_bboxes = img_gts['annotations']
            pred_hois = img_preds['hoi_prediction']
            gt_hois = img_gts['hoi_annotation']
            if len(gt_bboxes) != 0:
                bbox_pairs, bbox_overlaps = self.compute_iou_mat(gt_bboxes, pred_bboxes)
                self.compute_fptp(pred_hois, gt_hois, bbox_pairs, pred_bboxes, bbox_overlaps)
            else:
                for pred_hoi in pred_hois:
                    triplet = (pred_hoi['category_id'], pred_bboxes[pred_hoi['object_id']]['category_id'])
                    if triplet not in self.gt_triplets:
                        continue
                    self.tp[triplet].append(0)
                    self.fp[triplet].append(1)
                    self.score[triplet].append(pred_hoi['score'])
        map = self.compute_map()
        return map

    def compute_map(self):
        ap = defaultdict(lambda: 0)
        rare_ap = defaultdict(lambda: 0)
        non_rare_ap = defaultdict(lambda: 0)
        max_recall = defaultdict(lambda: 0)
        for triplet in self.gt_triplets:
            sum_gts = self.sum_gts[triplet]
            if sum_gts == 0:
                continue

            tp = np.array((self.tp[triplet]))
            fp = np.array((self.fp[triplet]))
            if len(tp) == 0:
                ap[triplet] = 0
                max_recall[triplet] = 0
                if triplet in self.rare_triplets:
                    rare_ap[triplet] = 0
                elif triplet in self.non_rare_triplets:
                    non_rare_ap[triplet] = 0
                else:
                    print('Warning: triplet {} is neither in rare triplets nor in non-rare triplets'.format(triplet))
                continue

            score = np.array(self.score[triplet])
            sort_inds = np.argsort(-score)
            fp = fp[sort_inds]
            tp = tp[sort_inds]
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / sum_gts
            prec = tp / (fp + tp)
            ap[triplet] = self.voc_ap(rec, prec)
            max_recall[triplet] = np.amax(rec)
            if triplet in self.rare_triplets:
                rare_ap[triplet] = ap[triplet]
            elif triplet in self.non_rare_triplets:
                non_rare_ap[triplet] = ap[triplet]
            else:
                print('Warning: triplet {} is neither in rare triplets nor in non-rare triplets'.format(triplet))
        m_ap = np.mean(list(ap.values()))
        m_ap_rare = np.mean(list(rare_ap.values()))
        m_ap_non_rare = np.mean(list(non_rare_ap.values()))
        m_max_recall = np.mean(list(max_recall.values()))

        print('--------------------')
        if self.zero_shot_type == "default":
            print('mAP full: {} mAP rare: {}  mAP non-rare: {}  mean max recall: {}'.format(m_ap, m_ap_rare,
                                                                                            m_ap_non_rare,
                                                                                            m_max_recall))
            return_dict = {'mAP': m_ap, 'mAP rare': m_ap_rare, 'mAP non-rare': m_ap_non_rare,
                           'mean max recall': m_max_recall}

        elif self.zero_shot_type == "unseen_object":
            print('mAP full: {} mAP unseen-obj: {}  mAP seen-obj: {}  mean max recall: {}'.format(m_ap, m_ap_rare,
                                                                                                  m_ap_non_rare,
                                                                                                  m_max_recall))
            return_dict = {'mAP': m_ap, 'mAP unseen-obj': m_ap_rare, 'mAP seen-obj': m_ap_non_rare,
                           'mean max recall': m_max_recall}

        else:
            print(
                'mAP full: {} mAP unseen: {}  mAP seen: {}  mean max recall: {}'.format(m_ap, m_ap_rare, m_ap_non_rare,
                                                                                        m_max_recall))
            return_dict = {'mAP': m_ap, 'mAP unseen': m_ap_rare, 'mAP seen': m_ap_non_rare,
                           'mean max recall': m_max_recall}

        print('--------------------')

        return return_dict
    def voc_ap(self, rec, prec):
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
        return ap

    def compute_fptp(self, pred_hois, gt_hois, match_pairs, pred_bboxes, bbox_overlaps):
        pos_pred_ids = match_pairs.keys()
        vis_tag = np.zeros(len(gt_hois))
        pred_hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
        if len(pred_hois) != 0:
            for pred_hoi in pred_hois:
                is_match = 0
                if len(match_pairs) != 0 and pred_hoi['subject_id'] in pos_pred_ids and pred_hoi['object_id'] in pos_pred_ids:
                    pred_sub_ids = match_pairs[pred_hoi['subject_id']]
                    pred_obj_ids = match_pairs[pred_hoi['object_id']]
                    pred_sub_overlaps = bbox_overlaps[pred_hoi['subject_id']]
                    pred_obj_overlaps = bbox_overlaps[pred_hoi['object_id']]
                    pred_category_id = pred_hoi['category_id']
                    max_overlap = 0
                    max_gt_hoi = 0
                    for gt_hoi in gt_hois:
                        if gt_hoi['subject_id'] in pred_sub_ids and gt_hoi['object_id'] in pred_obj_ids \
                           and pred_category_id == gt_hoi['category_id']:
                            is_match = 1
                            min_overlap_gt = min(pred_sub_overlaps[pred_sub_ids.index(gt_hoi['subject_id'])],
                                                 pred_obj_overlaps[pred_obj_ids.index(gt_hoi['object_id'])])
                            if min_overlap_gt > max_overlap:
                                max_overlap = min_overlap_gt
                                max_gt_hoi = gt_hoi
                triplet = (pred_hoi['category_id'], pred_bboxes[pred_hoi['object_id']]['category_id'])
                if triplet not in self.gt_triplets:
                    continue
                if is_match == 1 and vis_tag[gt_hois.index(max_gt_hoi)] == 0:
                    self.fp[triplet].append(0)
                    self.tp[triplet].append(1)
                    vis_tag[gt_hois.index(max_gt_hoi)] =1
                else:
                    self.fp[triplet].append(1)
                    self.tp[triplet].append(0)
                self.score[triplet].append(pred_hoi['score'])

    def compute_iou_mat(self, bbox_list1, bbox_list2):
        iou_mat = np.zeros((len(bbox_list1), len(bbox_list2)))
        if len(bbox_list1) == 0 or len(bbox_list2) == 0:
            return {}
        for i, bbox1 in enumerate(bbox_list1):
            for j, bbox2 in enumerate(bbox_list2):
                iou_i = self.compute_IOU(bbox1, bbox2)
                iou_mat[i, j] = iou_i

        iou_mat_ov=iou_mat.copy()
        iou_mat[iou_mat>=self.overlap_iou] = 1
        iou_mat[iou_mat<self.overlap_iou] = 0

        match_pairs = np.nonzero(iou_mat)
        match_pairs_dict = {}
        match_pair_overlaps = {}
        if iou_mat.max() > 0:
            for i, pred_id in enumerate(match_pairs[1]):
                if pred_id not in match_pairs_dict.keys():
                    match_pairs_dict[pred_id] = []
                    match_pair_overlaps[pred_id]=[]
                match_pairs_dict[pred_id].append(match_pairs[0][i])
                match_pair_overlaps[pred_id].append(iou_mat_ov[match_pairs[0][i],pred_id])
        return match_pairs_dict, match_pair_overlaps

    def compute_IOU(self, bbox1, bbox2):
        if isinstance(bbox1['category_id'], str):
            bbox1['category_id'] = int(bbox1['category_id'].replace('\n', ''))
        if isinstance(bbox2['category_id'], str):
            bbox2['category_id'] = int(bbox2['category_id'].replace('\n', ''))
        if bbox1['category_id'] == bbox2['category_id']:
            rec1 = bbox1['bbox']
            rec2 = bbox2['bbox']
            S_rec1 = (rec1[2] - rec1[0]+1) * (rec1[3] - rec1[1]+1)
            S_rec2 = (rec2[2] - rec2[0]+1) * (rec2[3] - rec2[1]+1)

            sum_area = S_rec1 + S_rec2

            left_line = max(rec1[1], rec2[1])
            right_line = min(rec1[3], rec2[3])
            top_line = max(rec1[0], rec2[0])
            bottom_line = min(rec1[2], rec2[2])
            if left_line >= right_line or top_line >= bottom_line:
                return 0
            else:
                intersect = (right_line - left_line+1) * (bottom_line - top_line+1)
                return intersect / (sum_area - intersect)
        else:
            return 0

    def triplet_nms_filter(self, preds):
        preds_filtered = []
        for img_preds in preds:
            pred_bboxes = img_preds['predictions']
            pred_hois = img_preds['hoi_prediction']
            all_triplets = {}
            for index, pred_hoi in enumerate(pred_hois):
                triplet = str(pred_bboxes[pred_hoi['subject_id']]['category_id']) + '_' + \
                          str(pred_bboxes[pred_hoi['object_id']]['category_id']) + '_' + str(pred_hoi['category_id'])

                if triplet not in all_triplets:
                    all_triplets[triplet] = {'subs':[], 'objs':[], 'scores':[], 'indexes':[]}
                all_triplets[triplet]['subs'].append(pred_bboxes[pred_hoi['subject_id']]['bbox'])
                all_triplets[triplet]['objs'].append(pred_bboxes[pred_hoi['object_id']]['bbox'])
                all_triplets[triplet]['scores'].append(pred_hoi['score'])
                all_triplets[triplet]['indexes'].append(index)

            all_keep_inds = []
            for triplet, values in all_triplets.items():
                subs, objs, scores = values['subs'], values['objs'], values['scores']
                keep_inds = self.pairwise_nms(np.array(subs), np.array(objs), np.array(scores))

                keep_inds = list(np.array(values['indexes'])[keep_inds])
                all_keep_inds.extend(keep_inds)

            preds_filtered.append({
                'filename': img_preds['filename'],
                'predictions': pred_bboxes,
                'hoi_prediction': list(np.array(img_preds['hoi_prediction'])[all_keep_inds])
                })

        return preds_filtered

    def pairwise_nms(self, subs, objs, scores):
        sx1, sy1, sx2, sy2 = subs[:, 0], subs[:, 1], subs[:, 2], subs[:, 3]
        ox1, oy1, ox2, oy2 = objs[:, 0], objs[:, 1], objs[:, 2], objs[:, 3]

        sub_areas = (sx2 - sx1 + 1) * (sy2 - sy1 + 1)
        obj_areas = (ox2 - ox1 + 1) * (oy2 - oy1 + 1)

        order = scores.argsort()[::-1]

        keep_inds = []
        while order.size > 0:
            i = order[0]
            keep_inds.append(i)

            sxx1 = np.maximum(sx1[i], sx1[order[1:]])
            syy1 = np.maximum(sy1[i], sy1[order[1:]])
            sxx2 = np.minimum(sx2[i], sx2[order[1:]])
            syy2 = np.minimum(sy2[i], sy2[order[1:]])

            sw = np.maximum(0.0, sxx2 - sxx1 + 1)
            sh = np.maximum(0.0, syy2 - syy1 + 1)
            sub_inter = sw * sh
            sub_union = sub_areas[i] + sub_areas[order[1:]] - sub_inter

            oxx1 = np.maximum(ox1[i], ox1[order[1:]])
            oyy1 = np.maximum(oy1[i], oy1[order[1:]])
            oxx2 = np.minimum(ox2[i], ox2[order[1:]])
            oyy2 = np.minimum(oy2[i], oy2[order[1:]])

            ow = np.maximum(0.0, oxx2 - oxx1 + 1)
            oh = np.maximum(0.0, oyy2 - oyy1 + 1)
            obj_inter = ow * oh
            obj_union = obj_areas[i] + obj_areas[order[1:]] - obj_inter

            ovr = np.power(sub_inter/sub_union, self.nms_alpha) * np.power(obj_inter / obj_union, self.nms_beta)
            inds = np.where(ovr <= self.thres_nms)[0]

            order = order[inds + 1]
        return keep_inds

