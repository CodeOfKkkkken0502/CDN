import copy
import pdb
from PIL import Image

import numpy as np
from collections import defaultdict
import os, cv2, json

class VCOCOEvaluator():

    def __init__(self, preds, gts, correct_mat, use_nms_filter=False):
        self.overlap_iou = 0.5
        self.max_hois = 100

        self.fp = defaultdict(list)
        self.tp = defaultdict(list)
        self.score = defaultdict(list)
        self.sum_gts = defaultdict(lambda: 0)

        self.verb_classes = ['hold_obj', 'stand', 'sit_instr', 'ride_instr', 'walk', 'look_obj', 'hit_instr', 'hit_obj',
                             'eat_obj', 'eat_instr', 'jump_instr', 'lay_instr', 'talk_on_phone_instr', 'carry_obj',
                             'throw_obj', 'catch_obj', 'cut_instr', 'cut_obj', 'run', 'work_on_computer_instr',
                             'ski_instr', 'surf_instr', 'skateboard_instr', 'smile', 'drink_instr', 'kick_obj',
                             'point_instr', 'read_obj', 'snowboard_instr']
        self.obj_classes = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
                            'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
                            'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
                            'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
                            'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',
                            'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed',
                            'dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
                            'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']
        self.thesis_map_indices = [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 24, 25, 27, 28]

        self.preds = []
        for img_preds in preds:
            img_preds = {k: v.to('cpu').numpy() for k, v in img_preds.items()}
            bboxes = [{'bbox': bbox, 'category_id': label} for bbox, label in zip(img_preds['boxes'], img_preds['labels'])]
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
                correct_mat = np.concatenate((correct_mat, np.ones((correct_mat.shape[0], 1))), axis=1)
                #print(object_ids.shape, object_labels.shape,correct_mat.shape)
                masks = correct_mat[verb_labels, object_labels]
                hoi_scores *= masks

                hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score} for
                        subject_id, object_id, category_id, score in zip(subject_ids, object_ids, verb_labels, hoi_scores)]
                hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
                hois = hois[:self.max_hois]
            else:
                hois = []

            self.preds.append({
                'predictions': bboxes,
                'hoi_prediction': hois
            })

        self.gts = []
        self.filenames = []
        self.img_dir = './data/v-coco/images/val2014/'
        self.output_dir = './vis/v-coco/'
        for img_gts in gts:
            self.filenames.append(img_gts['filename'])
            img_gts = {k: v.to('cpu').numpy() for k, v in img_gts.items() if k != 'id' and k != 'img_id' and k != 'filename'}
            self.gts.append({
                'annotations': [{'bbox': bbox, 'category_id': label} for bbox, label in zip(img_gts['boxes'], img_gts['labels'])],
                'hoi_annotation': [{'subject_id': hoi[0], 'object_id': hoi[1], 'category_id': hoi[2]} for hoi in img_gts['hois']]
            })
            for hoi in self.gts[-1]['hoi_annotation']:
                self.sum_gts[hoi['category_id']] += 1
        # self.preds[i].keys() = dict_keys(['predictions', 'hoi_prediction'])
        # len(self.preds[i]['predictions']) = 200
        # self.preds[i]['predictions'][j] = {'bbox': array(4,), 'category_id': int])
        # len(self.preds[i]['hoi_prediction']) = 100
        # self.preds[i]['hoi_prediction'][j] = {'subject_id': int, 'object_id': int, 'category_id': int, 'score': int}

        # self.gts[i].keys() = dict_keys(['annotations', 'hoi_annotation'])
        # len(self.gts[i]['annotations']) = num_bboxes
        # self.gts[i]['annotations'][j] = {'bbox': array(4,), 'category_id': int])
        # len(self.gts[i]['hoi_annotation']) = num_hois
        # self.gts[i]['hoi_annotation'][j] = {'subject_id': int, 'object_id': int, 'category_id': int}

    def visualize_bbox(self, img, pred, color, thickness=1):
        h, w, _ = img.shape
        x0 = int(max(pred['bbox'][0], 0))
        x0 = min(x0, w)
        x1 = int(max(pred['bbox'][2], 0))
        x1 = min(x1, w)
        y0 = int(max(pred['bbox'][1], 0))
        y0 = min(y0, h)
        y1 = int(max(pred['bbox'][3], 0))
        y1 = min(y1, h)
        x_center = int((x0 + x1)/2)
        y_center = int((y0 + y1)/2)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
        category = self.obj_classes[pred['category_id']]
        img = self.visualize_class(img, (x0, y0), category, color)
        return x_center, y_center, img

    def visualize_class(self, img, pos, class_str, color, font_scale=0.5):
        x0, y0 = int(pos[0]), int(pos[1])
        # Compute text size.
        txt = class_str
        font = cv2.FONT_HERSHEY_SIMPLEX
        ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
        # Place text background.
        back_tl = x0, y0
        back_br = x0 + txt_w, y0 + int(1.3 * txt_h)
        cv2.rectangle(img, back_tl, back_br, color, -1)
        # Show text.
        txt_tl = x0, y0 + txt_h
        cv2.putText(img, txt, txt_tl, font, font_scale, (255, 255, 255), lineType=cv2.LINE_AA)
        return img

    def visualize(self):
        _RED = (255, 0, 0)
        _BLUE = (0, 0, 255)
        _GREEN = (0, 255, 0)
        for img_preds, img_gts, img_filename in zip(self.preds, self.gts, self.filenames):
            img_file = self.img_dir + img_filename
            #img = Image.open(img_file).convert('RGB')
            img = cv2.imread(img_file)
            pred_bboxes = img_preds['predictions']
            gt_bboxes = img_gts['annotations']
            pred_hois = img_preds['hoi_prediction']
            gt_hois = img_gts['hoi_annotation']
            img_pred = copy.deepcopy(img)
            img_gt = copy.deepcopy(img)
            for i in range(len(gt_hois)):
                pred_hoi = pred_hois[i]
                pred_sub = pred_bboxes[pred_hoi['subject_id']]
                x_sub, y_sub, img_pred = self.visualize_bbox(img_pred, pred_sub, _RED, thickness=2)
                pred_category = self.verb_classes[pred_hoi['category_id']]
                pred_score = pred_hoi['score']
                pred_verb = pred_category + ' ' + str(pred_score)
                pred_obj = pred_bboxes[pred_hoi['object_id']]
                if not pred_obj['category_id'] == 80:
                    x_obj, y_obj, img_pred = self.visualize_bbox(img_pred, pred_obj, _BLUE, thickness=2)
                    cv2.line(img_pred, (x_sub, y_sub), (x_obj, y_obj), _GREEN, 2)
                    x_verb = int((x_sub+x_obj)/2)
                    y_verb = int((y_sub+y_obj)/2)
                else:
                    x_verb = x_sub
                    y_verb = y_sub
                img_pred = self.visualize_class(img_pred, (x_verb, y_verb), pred_verb, _GREEN)

                gt_hoi = gt_hois[i]
                gt_sub = gt_bboxes[gt_hoi['subject_id']]
                x_sub, y_sub, img_gt = self.visualize_bbox(img_gt, gt_sub, _RED, thickness=2)
                gt_category = self.verb_classes[gt_hoi['category_id']]
                gt_obj = gt_bboxes[gt_hoi['object_id']]
                if not gt_hoi['object_id'] == -1:
                    x_obj, y_obj, img_gt = self.visualize_bbox(img_gt, gt_obj, _BLUE, thickness=2)
                    cv2.line(img_gt, (x_sub, y_sub), (x_obj, y_obj), _GREEN, 2)
                    x_verb = int((x_sub + x_obj) / 2)
                    y_verb = int((y_sub + y_obj) / 2)
                else:
                    x_verb = x_sub
                    y_verb = y_sub
                img_gt = self.visualize_class(img_gt, (x_verb, y_verb), gt_category, _GREEN)

            output_pred = self.output_dir + 'pred/pred_' + img_filename
            output_gt = self.output_dir + 'gt/gt_' + img_filename
            cv2.imwrite(output_pred, img_pred)
            cv2.imwrite(output_gt, img_gt)


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
                    self.tp[pred_hoi['category_id']].append(0)
                    self.fp[pred_hoi['category_id']].append(1)
                    self.score[pred_hoi['category_id']].append(pred_hoi['score'])
        map = self.compute_map()
        return map

    def compute_map(self):
        print('------------------------------------------------------------')
        ap = defaultdict(lambda: 0)
        aps = {}
        for category_id in sorted(list(self.sum_gts.keys())):
            sum_gts = self.sum_gts[category_id]
            if sum_gts == 0:
                continue

            tp = np.array((self.tp[category_id]))
            fp = np.array((self.fp[category_id]))
            if len(tp) == 0:
                ap[category_id] = 0
            else:
                score = np.array(self.score[category_id])
                sort_inds = np.argsort(-score)
                fp = fp[sort_inds]
                tp = tp[sort_inds]
                fp = np.cumsum(fp)
                tp = np.cumsum(tp)
                rec = tp / sum_gts
                prec = tp / (fp + tp)
                ap[category_id] = self.voc_ap(rec, prec)
            print('{:>23s}: #GTs = {:>04d}, AP = {:>.4f}'.format(self.verb_classes[category_id], sum_gts, ap[category_id]))
            aps['AP_{}'.format(self.verb_classes[category_id])] = ap[category_id]

        m_ap_all = np.mean(list(ap.values()))
        m_ap_thesis = np.mean([ap[category_id] for category_id in self.thesis_map_indices])

        print('------------------------------------------------------------')
        print('mAP all: {:.4f} mAP thesis: {:.4f}'.format(m_ap_all, m_ap_thesis))
        print('------------------------------------------------------------')

        aps.update({'mAP_all': m_ap_all, 'mAP_thesis': m_ap_thesis})

        return aps

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
                max_overlap = 0
                max_gt_hoi = 0
                for gt_hoi in gt_hois:
                    if len(match_pairs) != 0 and pred_hoi['subject_id'] in pos_pred_ids and \
                       gt_hoi['object_id'] == -1:
                        pred_sub_ids = match_pairs[pred_hoi['subject_id']]
                        pred_sub_overlaps = bbox_overlaps[pred_hoi['subject_id']]
                        pred_category_id = pred_hoi['category_id']
                        if gt_hoi['subject_id'] in pred_sub_ids and pred_category_id == gt_hoi['category_id']:
                            is_match = 1
                            min_overlap_gt = pred_sub_overlaps[pred_sub_ids.index(gt_hoi['subject_id'])]
                            if min_overlap_gt > max_overlap:
                                max_overlap = min_overlap_gt
                                max_gt_hoi = gt_hoi
                    elif len(match_pairs) != 0 and pred_hoi['subject_id'] in pos_pred_ids and \
                         pred_hoi['object_id'] in pos_pred_ids:
                        pred_sub_ids = match_pairs[pred_hoi['subject_id']]
                        pred_obj_ids = match_pairs[pred_hoi['object_id']]
                        pred_sub_overlaps = bbox_overlaps[pred_hoi['subject_id']]
                        pred_obj_overlaps = bbox_overlaps[pred_hoi['object_id']]
                        pred_category_id = pred_hoi['category_id']
                        if gt_hoi['subject_id'] in pred_sub_ids and gt_hoi['object_id'] in pred_obj_ids and \
                           pred_category_id == gt_hoi['category_id']:
                            is_match = 1
                            min_overlap_gt = min(pred_sub_overlaps[pred_sub_ids.index(gt_hoi['subject_id'])],
                                                 pred_obj_overlaps[pred_obj_ids.index(gt_hoi['object_id'])])
                            if min_overlap_gt > max_overlap:
                                max_overlap = min_overlap_gt
                                max_gt_hoi = gt_hoi
                if is_match == 1 and vis_tag[gt_hois.index(max_gt_hoi)] == 0:
                    self.fp[pred_hoi['category_id']].append(0)
                    self.tp[pred_hoi['category_id']].append(1)
                    vis_tag[gt_hois.index(max_gt_hoi)] = 1
                else:
                    self.fp[pred_hoi['category_id']].append(1)
                    self.tp[pred_hoi['category_id']].append(0)
                self.score[pred_hoi['category_id']].append(pred_hoi['score'])

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
            # computing area of each rectangles
            S_rec1 = (rec1[2] - rec1[0]+1) * (rec1[3] - rec1[1]+1)
            S_rec2 = (rec2[2] - rec2[0]+1) * (rec2[3] - rec2[1]+1)

            # computing the sum_area
            sum_area = S_rec1 + S_rec2

            # find the each edge of intersect rectangle
            left_line = max(rec1[1], rec2[1])
            right_line = min(rec1[3], rec2[3])
            top_line = max(rec1[0], rec2[0])
            bottom_line = min(rec1[2], rec2[2])
            # judge if there is an intersect
            if left_line >= right_line or top_line >= bottom_line:
                return 0
            else:
                intersect = (right_line - left_line+1) * (bottom_line - top_line+1)
                return intersect / (sum_area - intersect)
        else:
            return 0
