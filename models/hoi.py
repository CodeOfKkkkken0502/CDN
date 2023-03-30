import copy
import sys

import torch
from torch import nn
import torch.nn.functional as F

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
import numpy as np
from queue import Queue
import math
import pdb
from .backbone import build_backbone
from .matcher import build_matcher
from .cdn import build_cdn
from .cdn import TransformerDecoderLayer, TransformerDecoder


class CDNHOI(nn.Module):

    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries, aux_loss=False, args=None):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.use_matching = args.use_matching
        self.dec_layers_hopd = args.dec_layers_hopd
        self.dec_layers_interaction = args.dec_layers_interaction
        if self.use_matching:
            self.matching_embed = nn.Linear(hidden_dim, 2)

    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        # pos:[B,hidden_dim(256),15,20]
        src, mask = features[-1].decompose()  # src:[B,2048,15,20],mask:[B,15,20]
        assert mask is not None
        hopd_out, interaction_decoder_out = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[:2]
        # self.input_proj(src):[B,256,15,20]
        # hopd_out, interaction_decoder_out:[C(3),B,num_queries(100),hidden_dim(256)]
        outputs_sub_coord = self.sub_bbox_embed(hopd_out).sigmoid()  # [C(3),B,num_queries(100),4]
        outputs_obj_coord = self.obj_bbox_embed(hopd_out).sigmoid()  # [C(3),B,num_queries(100),4]
        outputs_obj_class = self.obj_class_embed(hopd_out)  # [C(3),B,num_queries(100),num_obj_classes+1(82)]
        if self.use_matching:
            outputs_matching = self.matching_embed(hopd_out)

        outputs_verb_class = self.verb_class_embed(interaction_decoder_out)
        # [C(3),B,num_queries(100),num_verb_classes(29)]
        out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}
        if self.use_matching:
            out['pred_matching_logits'] = outputs_matching[-1]

        if self.aux_loss:
            if self.use_matching:
                out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                        outputs_sub_coord, outputs_obj_coord,
                                                        outputs_matching)
            else:
                out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                        outputs_sub_coord, outputs_obj_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord, outputs_matching=None):
        min_dec_layers_num = min(self.dec_layers_hopd, self.dec_layers_interaction)
        if self.use_matching:
            return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c,
                     'pred_obj_boxes': d, 'pred_matching_logits': e}
                    for a, b, c, d, e in zip(outputs_obj_class[-min_dec_layers_num : -1], outputs_verb_class[-min_dec_layers_num : -1],
                                             outputs_sub_coord[-min_dec_layers_num : -1], outputs_obj_coord[-min_dec_layers_num : -1],
                                             outputs_matching[-min_dec_layers_num : -1])]
        else:
            return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                    for a, b, c, d in zip(outputs_obj_class[-min_dec_layers_num : -1], outputs_verb_class[-min_dec_layers_num : -1],
                                          outputs_sub_coord[-min_dec_layers_num : -1], outputs_obj_coord[-min_dec_layers_num : -1])]


class CDNHOI2(nn.Module):

    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries, aux_loss=False, args=None):
        super().__init__()
        self.num_obj_classes = num_obj_classes
        self.num_verb_classes = num_verb_classes
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.verb_class_embed = nn.Linear(hidden_dim * 2, num_verb_classes)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.use_matching = args.use_matching
        self.dec_layers_hopd = args.dec_layers_hopd
        self.dec_layers_interaction = args.dec_layers_interaction
        if self.use_matching:
            self.matching_embed = nn.Linear(hidden_dim, 2)
        self.fusion_mode = args.fusion_mode
        if self.fusion_mode:
            self.fusion_module = MLP(hidden_dim*2, hidden_dim*2, hidden_dim*2, 3)

    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        # pos:[B,hidden_dim(256),15,20]
        src, mask = features[-1].decompose()  # src:[B,2048,15,20],mask:[B,15,20]
        assert mask is not None
        hopd_out, interaction_decoder_out = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[:2]
        # self.input_proj(src):[B,256,15,20]
        # hopd_out, interaction_decoder_out:[C(3),B,num_queries(100),hidden_dim(256)]
        outputs_sub_coord = self.sub_bbox_embed(hopd_out).sigmoid()  # [C(3),B,num_queries(100),4]
        outputs_obj_coord = self.obj_bbox_embed(hopd_out).sigmoid()  # [C(3),B,num_queries(100),4]
        outputs_obj_class = self.obj_class_embed(hopd_out)  # [C(3),B,num_queries(100),num_obj_classes+1(82)]
        obj_verb_rep = torch.cat((hopd_out, interaction_decoder_out), 3)
        if self.use_matching:
            outputs_matching = self.matching_embed(hopd_out)
        if self.fusion_mode:
            obj_verb_rep = self.fusion_module(obj_verb_rep)
        outputs_verb_class = self.verb_class_embed(obj_verb_rep)
        # [C(3),B,num_queries(100),num_verb_classes(29)]
        out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}
        if self.use_matching:
            out['pred_matching_logits'] = outputs_matching[-1]

        if self.aux_loss:
            if self.use_matching:
                out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                        outputs_sub_coord, outputs_obj_coord,
                                                        outputs_matching)
            else:
                out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                        outputs_sub_coord, outputs_obj_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord, outputs_matching=None):
        min_dec_layers_num = min(self.dec_layers_hopd, self.dec_layers_interaction)
        if self.use_matching:
            return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c,
                     'pred_obj_boxes': d, 'pred_matching_logits': e}
                    for a, b, c, d, e in zip(outputs_obj_class[-min_dec_layers_num : -1], outputs_verb_class[-min_dec_layers_num : -1],
                                             outputs_sub_coord[-min_dec_layers_num : -1], outputs_obj_coord[-min_dec_layers_num : -1],
                                             outputs_matching[-min_dec_layers_num : -1])]
        else:
            return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                    for a, b, c, d in zip(outputs_obj_class[-min_dec_layers_num : -1], outputs_verb_class[-min_dec_layers_num : -1],
                                          outputs_sub_coord[-min_dec_layers_num : -1], outputs_obj_coord[-min_dec_layers_num : -1])]

class CDNHOICompo(nn.Module):

    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries, aux_loss=False, fusion_mode=0, args=None):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.fusion_mode = fusion_mode
        if self.fusion_mode:
            self.fusion_module = MLP(hidden_dim*2, hidden_dim*2, hidden_dim*2, 1)
            #decoder_layer = TransformerDecoderLayer(d_model=args.hidden_dim, nhead=args.nheads, dim_feedforward=args.dim_feedforward,
            #                                        dropout=args.dropout, normalize_before=args.pre_norm)
            #decoder_norm = nn.LayerNorm(args.hidden_dim)
            #self.fusion_module = TransformerDecoder(decoder_layer, 1, decoder_norm, return_intermediate=True)
        self.verb_class_embed = nn.Linear(hidden_dim * 2, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.use_matching = args.use_matching
        self.dec_layers_hopd = args.dec_layers_hopd
        self.dec_layers_interaction = args.dec_layers_interaction
        if self.use_matching:
            self.matching_embed = nn.Linear(hidden_dim, 2)
        self.uncertainty = args.uncertainty
        self.recouple = args.recouple
        self.separate = args.separate
        if self.uncertainty:
            self.uctt = MLP_UCTT(hidden_dim * 2, hidden_dim * 2, num_verb_classes, 1, num_queries)

    def forward(self, samples):
        # hopd_outs = []
        # interaction_decoder_outs = []
        #outs = []
        if isinstance(samples, list):
            sample = samples[0]
            features, pos = self.backbone(sample)
            # pos:[B,hidden_dim(256),15,20]
            src, mask = features[-1].decompose()
            # src:[B,2048,15,20],mask:[B,15,20]
            assert mask is not None
            if self.separate:
                if self.recouple:
                    human_out, obj_out, interaction_decoder_out, interaction_decoder_out_compo= self.transformer(
                        self.input_proj(src), mask, self.query_embed.weight, pos[-1])
                else:
                    human_out, obj_out, interaction_decoder_out = self.transformer(self.input_proj(src), mask,
                                                                                   self.query_embed.weight, pos[-1])
            else:
                if self.recouple:
                    hopd_out, interaction_decoder_out, interaction_decoder_out_compo = self.transformer(self.input_proj(src), mask,
                                                                         self.query_embed.weight, pos[-1])
                else:
                    hopd_out, interaction_decoder_out = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[:2]
            # hopd_outs.append(hopd_out)
            # interaction_decoder_outs.append(interaction_decoder_out)

            if self.separate:
                outputs_sub_coord = self.sub_bbox_embed(human_out).sigmoid()
                outputs_obj_coord = self.obj_bbox_embed(obj_out).sigmoid()
                outputs_obj_class = self.obj_class_embed(obj_out)
                if self.use_matching:
                    outputs_matching = self.matching_embed(obj_out)
                obj_verb_rep = torch.cat((obj_out, interaction_decoder_out), 3)
            else:
                # self.input_proj(src):[B,256,15,20]
                # hopd_out, interaction_decoder_out:[num_decoder_layers(3),B,num_queries(100),hidden_dim(256)]
                outputs_sub_coord = self.sub_bbox_embed(hopd_out).sigmoid()  # [num_decoder_layers(3),B,num_queries(100),4]
                outputs_obj_coord = self.obj_bbox_embed(hopd_out).sigmoid()  # [num_decoder_layers(3),B,num_queries(100),4]
                outputs_obj_class = self.obj_class_embed(hopd_out)  # [num_decoder_layers(3),B,num_queries(100),num_obj_classes+1(82)]
                if self.use_matching:
                    outputs_matching = self.matching_embed(hopd_out)
                obj_verb_rep = torch.cat((hopd_out, interaction_decoder_out), 3)
            if self.fusion_mode:
                obj_verb_rep = self.fusion_module(obj_verb_rep)
            outputs_verb_class = self.verb_class_embed(obj_verb_rep)
            # [num_decoder_layers(3),B,num_queries(100),num_verb_classes(29)]
            out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
                   'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}
            uctt_verb = None
            if self.uncertainty:
                uctt_verb = []
                for i in range(obj_verb_rep.shape[0]):
                    uctt_i = self.uctt(obj_verb_rep[i])
                    uctt_verb.append(uctt_i)
                uctt_verb = torch.stack(uctt_verb)
                out['uctt_verb'] = uctt_verb[-1]
            if self.use_matching:
                out['pred_matching_logits'] = outputs_matching[-1]
            if self.aux_loss:
                if self.use_matching:
                    out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                            outputs_sub_coord, outputs_obj_coord,
                                                            outputs_matching, outputs_verb_uctt=uctt_verb)
                else:
                    out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                            outputs_sub_coord, outputs_obj_coord,
                                                            outputs_verb_uctt=uctt_verb)
            if self.separate:
                return out, human_out, obj_out, interaction_decoder_out
            else:
                return out, hopd_out, interaction_decoder_out
            #outs.append(out)
        else:
            return self.forward_eval(samples)

    def forward_compo(self, human_out, obj_out, hopd_out, interaction_decoder_out, indices):
        # indices: [(tensor([27, 37]), tensor([1, 0])), (tensor([95]), tensor([0]))] (num_hois * index_of_matched_query),(num_hois * index_of_gt)
        instance_out_compo = []
        human_out_compo = []
        interaction_decoder_out_compo = []
        num_queries = []
        half_bs = int(interaction_decoder_out.shape[1]/2)
        if obj_out is None:
            instance_out = hopd_out
        else:
            instance_out = obj_out
        for i in range(half_bs * 2):
            index = half_bs + i if i < half_bs else i - half_bs
            num_HO = (len(indices[i][0]), len(indices[index][0]))
            if human_out is not None:
                human_out_compo_list = []
            instance_out_compo_list = []
            interaction_decoder_out_compo_list = []
            #cross compo
            for j in range(num_HO[0]):
                instance_out_1 = instance_out[:, i, indices[i][0][j], :]
                for k in range(num_HO[1]):
                    interaction_decoder_out_2 = interaction_decoder_out[:, index, indices[index][0][k], :]
                    instance_out_compo_list.append(instance_out_1)
                    if human_out is not None:
                        human_out_1 = human_out[:, i, indices[i][0][j], :]
                        human_out_compo_list.append(human_out_1)
                    interaction_decoder_out_compo_list.append(interaction_decoder_out_2)
            instance_out_compo_list = instance_out_compo_list[0:100]
            if human_out is not None:
                human_out_compo_list = human_out_compo_list[0:100]
            interaction_decoder_out_compo_list = interaction_decoder_out_compo_list[0:100]
            #self compo
            for j in range(num_HO[0]):
                instance_out_1 = instance_out[:, i, indices[i][0][j], :]
                for l in range(num_HO[0]):
                    if l != j:
                        interaction_decoder_out_1 = interaction_decoder_out[:, i, indices[i][0][l], :]
                        instance_out_compo_list.append(instance_out_1)
                        if human_out is not None:
                            human_out_1 = human_out[:, i, indices[i][0][j], :]
                            human_out_compo_list.append(human_out_1)
                        interaction_decoder_out_compo_list.append(interaction_decoder_out_1)
            instance_out_compo_list = instance_out_compo_list[0:200]
            if human_out is not None:
                human_out_compo_list = human_out_compo_list[0:200]
            interaction_decoder_out_compo_list = interaction_decoder_out_compo_list[0:200]

            if len(instance_out_compo_list):
                instance_out_compo_1 = torch.stack(instance_out_compo_list, dim=1)
                interaction_decoder_out_compo_1 = torch.stack(interaction_decoder_out_compo_list, dim=1)
                if human_out is not None:
                    human_out_compo_1 = torch.stack(human_out_compo_list, dim=1)
                num_queries.append(interaction_decoder_out_compo_1.shape[1])
            else:
                instance_out_compo_1 = instance_out[:,i,:,:].clone()
                if human_out is not None:
                    human_out_compo_1 = human_out[:,i,:,:].clone()
                interaction_decoder_out_compo_1 = interaction_decoder_out[:,index,:,:].clone()
                num_queries.append(instance_out[:, i, :, :].shape[1])
            if human_out is not None:
                human_out_compo.append(human_out_compo_1)
            instance_out_compo.append(instance_out_compo_1)
            interaction_decoder_out_compo.append(interaction_decoder_out_compo_1)

        num_max = max(num_queries)
        for i in range(half_bs * 2):
            if num_queries[i] != num_max:
                padding = torch.zeros(instance_out_compo[0].shape[0], num_max - num_queries[i],
                                      instance_out_compo[0].shape[2]).to(instance_out_compo[i].device)
                instance_out_compo[i] = torch.cat((instance_out_compo[i], padding), dim=1)
                interaction_decoder_out_compo[i] = torch.cat((interaction_decoder_out_compo[i], padding), dim=1)
                if human_out is not None:
                    human_out_compo[i] = torch.cat((human_out_compo[i], padding), dim=1)
        instance_out_compo = torch.stack(instance_out_compo, dim=1)
        interaction_decoder_out_compo = torch.stack(interaction_decoder_out_compo, dim=1)
        if human_out is not None:
            human_out_compo = torch.stack(human_out_compo, dim=1)

        obj_verb_rep_compo = torch.cat((instance_out_compo, interaction_decoder_out_compo), dim=3)
        if self.fusion_mode:
            obj_verb_rep_compo = self.fusion_module(obj_verb_rep_compo)
        if human_out is None:
            outputs_sub_coord = self.sub_bbox_embed(instance_out_compo).sigmoid()  # [num_decoder_layers(3),B,num_queries(100),4]
        else:
            outputs_sub_coord = self.sub_bbox_embed(human_out_compo).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(instance_out_compo).sigmoid()  # [num_decoder_layers(3),B,num_queries(100),4]
        outputs_obj_class = self.obj_class_embed(
            instance_out_compo)  # [num_decoder_layers(3),B,num_queries(100),num_obj_classes+1(82)]
        if self.use_matching:
            outputs_matching = self.matching_embed(instance_out_compo)
        outputs_verb_class = self.verb_class_embed(obj_verb_rep_compo)
        # [num_decoder_layers(3),B,num_queries(100),num_verb_classes(29)]
        out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}
        uctt_verb = None
        if self.uncertainty:
            uctt_verb = []
            for i in range(obj_verb_rep_compo.shape[0]):
                uctt_i = self.uctt(obj_verb_rep_compo[i])
                uctt_verb.append(uctt_i)
            uctt_verb = torch.stack(uctt_verb)
            out['uctt_verb'] = uctt_verb[-1]
        if self.use_matching:
            out['pred_matching_logits'] = outputs_matching[-1]
        if self.aux_loss:
            if self.use_matching:
                out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                        outputs_sub_coord, outputs_obj_coord,
                                                        outputs_matching, outputs_verb_uctt=uctt_verb)
            else:
                out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                        outputs_sub_coord, outputs_obj_coord,
                                                        outputs_verb_uctt=uctt_verb)
        return out

    def forward_eval(self, samples):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        # pos:[B,hidden_dim(256),15,20]
        src, mask = features[-1].decompose()  # src:[B,2048,15,20],mask:[B,15,20]
        assert mask is not None
        if self.separate:
            human_out, obj_out, interaction_decoder_out = self.transformer(self.input_proj(src), mask,
                                                                               self.query_embed.weight, pos[-1])
            outputs_sub_coord = self.sub_bbox_embed(human_out).sigmoid()
            outputs_obj_coord = self.obj_bbox_embed(obj_out).sigmoid()
            outputs_obj_class = self.obj_class_embed(obj_out)
            if self.use_matching:
                outputs_matching = self.matching_embed(obj_out)
            obj_verb_rep = torch.cat((obj_out, interaction_decoder_out), 3)
        else:
            hopd_out, interaction_decoder_out = self.transformer(self.input_proj(src), mask,
                                                                     self.query_embed.weight, pos[-1])[:2]
            outputs_sub_coord = self.sub_bbox_embed(hopd_out).sigmoid()  # [num_decoder_layers(3),B,num_queries(100),4]
            outputs_obj_coord = self.obj_bbox_embed(hopd_out).sigmoid()  # [num_decoder_layers(3),B,num_queries(100),4]
            outputs_obj_class = self.obj_class_embed(
                hopd_out)  # [num_decoder_layers(3),B,num_queries(100),num_obj_classes+1(82)]
            if self.use_matching:
                outputs_matching = self.matching_embed(hopd_out)
            obj_verb_rep = torch.cat((hopd_out, interaction_decoder_out), 3)
        if self.fusion_mode:
            obj_verb_rep = self.fusion_module(obj_verb_rep)
        outputs_verb_class = self.verb_class_embed(obj_verb_rep)
        # [C(3),B,num_queries(100),num_verb_classes(29)]
        out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}
        uctt_verb = None
        if self.uncertainty:
            uctt_verb = []
            for i in range(obj_verb_rep.shape[0]):
                uctt_i = self.uctt(obj_verb_rep[i])
                uctt_verb.append(uctt_i)
            uctt_verb = torch.stack(uctt_verb)
            out['uctt_verb'] = uctt_verb[-1]
        if self.use_matching:
            out['pred_matching_logits'] = outputs_matching[-1]
        if self.aux_loss:
            if self.use_matching:
                out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                        outputs_sub_coord, outputs_obj_coord,
                                                        outputs_matching, outputs_verb_uctt=uctt_verb)
            else:
                out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                        outputs_sub_coord, outputs_obj_coord,
                                                        outputs_verb_uctt=uctt_verb)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord, outputs_matching=None, outputs_verb_uctt=None):
        min_dec_layers_num = min(self.dec_layers_hopd, self.dec_layers_interaction)
        if self.use_matching:
            if outputs_verb_uctt is not None:
                return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c,
                         'pred_obj_boxes': d, 'pred_matching_logits': e, 'uctt_verb': f}
                        for a, b, c, d, e, f in
                        zip(outputs_obj_class[-min_dec_layers_num: -1], outputs_verb_class[-min_dec_layers_num: -1],
                            outputs_sub_coord[-min_dec_layers_num: -1], outputs_obj_coord[-min_dec_layers_num: -1],
                            outputs_matching[-min_dec_layers_num: -1], outputs_verb_uctt[-min_dec_layers_num: -1])]
            else:
                return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c,
                         'pred_obj_boxes': d, 'pred_matching_logits': e}
                        for a, b, c, d, e in
                        zip(outputs_obj_class[-min_dec_layers_num: -1], outputs_verb_class[-min_dec_layers_num: -1],
                            outputs_sub_coord[-min_dec_layers_num: -1], outputs_obj_coord[-min_dec_layers_num: -1],
                            outputs_matching[-min_dec_layers_num: -1])]
        else:
            if outputs_verb_uctt is not None:
                return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d,
                         'uctt_verb': e}
                        for a, b, c, d, e in
                        zip(outputs_obj_class[-min_dec_layers_num: -1], outputs_verb_class[-min_dec_layers_num: -1],
                            outputs_sub_coord[-min_dec_layers_num: -1], outputs_obj_coord[-min_dec_layers_num: -1],
                            outputs_verb_uctt[-min_dec_layers_num: -1])]
            else:
                return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                    for a, b, c, d in zip(outputs_obj_class[-min_dec_layers_num : -1], outputs_verb_class[-min_dec_layers_num : -1],
                                          outputs_sub_coord[-min_dec_layers_num : -1], outputs_obj_coord[-min_dec_layers_num : -1])]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class MLP_UCTT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_queries):
        super().__init__()
        self.num_layers = num_layers
        self.num_queries = 200
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList()
        for i, dims in enumerate(zip([input_dim] + h, h + [output_dim])):
            self.layers.append(nn.Linear(dims[0], dims[1]))
            self.layers.append(nn.BatchNorm1d(self.num_queries))
            if i < num_layers-1:
                self.layers.append(nn.ReLU())
            else:
                self.layers.append(nn.Tanh())

    def forward(self, x):
        orig_num_queries = x.shape[1]
        if orig_num_queries < self.num_queries:
            padding = torch.zeros(x.shape[0],self.num_queries-orig_num_queries,x.shape[2]).to(x.device)
            x = torch.cat((x,padding),dim=1)
        for i, layer in enumerate(self.layers):
            #print(x.shape)
            x = layer(x)
        x = x[:,:orig_num_queries,:]
        return x


class SetCriterionHOI(nn.Module):

    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, eos_coef, losses, args):
        super().__init__()

        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_obj_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.alpha = args.alpha
        self.uncertainty = args.uncertainty

        if args.dataset_file == 'hico':
            self.obj_nums_init = [1811, 9462, 2415, 7249, 1665, 3587, 1396, 1086, 10369, 800,
                                  287, 77, 332, 2352, 974, 470, 1386, 4889, 1675, 1131,
                                  1642, 185, 92, 717, 2228, 4396, 275, 1236, 1447, 1207,
                                  2949, 2622, 1689, 2345, 1863, 408, 5594, 1178, 562, 1479,
                                  988, 1057, 419, 1451, 504, 177, 1358, 429, 448, 186,
                                  121, 441, 735, 706, 868, 1238, 1838, 1224, 262, 517,
                                  5787, 200, 529, 1337, 146, 272, 417, 1277, 31, 213,
                                  7, 102, 102, 2424, 606, 215, 509, 529, 102, 572]
        elif args.dataset_file == 'vcoco':
            self.obj_nums_init = [5397, 238, 332, 321, 5, 6, 45, 90, 59, 20,
                                  13, 5, 6, 313, 28, 25, 46, 277, 20, 16,
                                  154, 0, 7, 13, 356, 191, 458, 66, 337, 1364,
                                  1382, 958, 1166, 68, 258, 221, 1317, 1428, 759, 201,
                                  190, 444, 274, 587, 124, 107, 102, 37, 226, 16,
                                  30, 22, 187, 320, 222, 465, 893, 213, 56, 322,
                                  306, 13, 55, 834, 23, 104, 38, 861, 11, 27,
                                  0, 16, 22, 405, 50, 14, 145, 63, 9, 11]
        else:
            raise

        self.obj_nums_init.append(3 * sum(self.obj_nums_init))  # 3 times fg for bg init

        if args.dataset_file == 'hico':
            self.verb_nums_init = [67, 43, 157, 321, 664, 50, 232, 28, 5342, 414,
                                   49, 105, 26, 78, 157, 408, 358, 129, 121, 131,
                                   275, 1309, 3, 799, 2338, 128, 633, 79, 435, 1,
                                   905, 19, 319, 47, 816, 234, 17958, 52, 97, 648,
                                   61, 1430, 13, 1862, 299, 123, 52, 328, 121, 752,
                                   111, 30, 293, 6, 193, 32, 4, 15421, 795, 82,
                                   30, 10, 149, 24, 59, 504, 57, 339, 62, 38,
                                   472, 128, 672, 1506, 16, 275, 16092, 757, 530, 380,
                                   132, 68, 20, 111, 2, 160, 3209, 12246, 5, 44,
                                   18, 7, 5, 4815, 1302, 69, 37, 25, 5048, 424,
                                   1, 235, 150, 131, 383, 72, 76, 139, 258, 464,
                                   872, 360, 1917, 1, 3775, 1206, 1]
        elif args.dataset_file == 'vcoco':
            self.verb_nums_init = [4001, 4598, 1989, 488, 656, 3825, 367, 367, 677, 677,
                                   700, 471, 354, 498, 300, 313, 300, 300, 622, 458,
                                   500, 498, 489, 1545, 133, 142, 38, 116, 388]
        else:
            raise

        self.verb_nums_init.append(3 * sum(self.verb_nums_init))

        self.obj_reweight = args.obj_reweight
        self.verb_reweight = args.verb_reweight
        self.use_static_weights = args.use_static_weights
        
        Maxsize = args.queue_size

        if self.obj_reweight:
            self.q_obj = Queue(maxsize=Maxsize)
            self.p_obj = args.p_obj
            self.obj_weights_init = self.cal_weights(self.obj_nums_init, p=self.p_obj)

        if self.verb_reweight:
            self.q_verb = Queue(maxsize=Maxsize)
            self.p_verb = args.p_verb
            self.verb_weights_init = self.cal_weights(self.verb_nums_init, p=self.p_verb)

    def cal_weights(self, label_nums, p=0.5):
        num_fgs = len(label_nums[:-1])
        weight = [0] * (num_fgs + 1)
        num_all = sum(label_nums[:-1])

        for index in range(num_fgs):
            if label_nums[index] == 0: continue
            weight[index] = np.power(num_all/label_nums[index], p)

        weight = np.array(weight)
        weight = weight / np.mean(weight[weight>0])

        weight[-1] = np.power(num_all/label_nums[-1], p) if label_nums[-1] != 0 else 0

        weight = torch.FloatTensor(weight).cuda()
        return weight

    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_obj_logits' in outputs
        src_logits = outputs['pred_obj_logits']  #[B,num_queries,num_obj_classes]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        # indices: [(tensor([97]), tensor([0])), (tensor([43, 61]), tensor([0, 1]))]
        # idx: (tensor([0, 1, 1]), tensor([97, 43, 61]))
        # target_classes_o: tensor([36, 80, 80], device='cuda:0')
        # target_classes: tensor[B,num_queries](81,81,81,...)

        if not self.obj_reweight:
            obj_weights = self.empty_weight
        elif self.use_static_weights:
            obj_weights = self.obj_weights_init
        else:
            obj_label_nums_in_batch = [0] * (self.num_obj_classes + 1)
            for target_class in target_classes:
                for label in target_class:
                    obj_label_nums_in_batch[label] += 1

            if self.q_obj.full(): self.q_obj.get()
            self.q_obj.put(np.array(obj_label_nums_in_batch))
            accumulated_obj_label_nums = np.sum(self.q_obj.queue, axis=0)
            obj_weights = self.cal_weights(accumulated_obj_label_nums, p=self.p_obj)

            aphal = min(math.pow(0.999, self.q_obj.qsize()),0.9)
            obj_weights = aphal * self.obj_weights_init + (1 - aphal) * obj_weights

        loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, obj_weights)  #transpose:[B,num_obj_classes,num_queries]
        losses = {'loss_obj_ce': loss_obj_ce}

        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions):
        pred_logits = outputs['pred_obj_logits']  #[B,num_queries,num_obj_classes+1]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)  #[B]
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}
        return losses

    def loss_verb_labels(self, outputs, targets, indices, num_interactions):
        assert 'pred_verb_logits' in outputs
        src_logits = outputs['pred_verb_logits']  # [B,num_queries,num_verb_classes]
        uncertainty = None
        if 'uctt_verb' in outputs:
            uncertainty = outputs['uctt_verb']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o
        # target_classes_o: [num_hois_in_batch,num_verb_classes]
        # target_classes: [B,num_queries,num_verb_classes]

        if not self.verb_reweight:
            verb_weights = None
        elif self.use_static_weights:
            verb_weights = self.verb_weights_init
        else:
            verb_label_nums_in_batch = [0] * (self.num_verb_classes + 1)
            for target_class in target_classes:
                for label in target_class:
                    label_classes = torch.where(label > 0)[0]
                    if len(label_classes) == 0:
                        verb_label_nums_in_batch[-1] += 1
                    else:
                        for label_class in label_classes:
                            verb_label_nums_in_batch[label_class] += 1

            if self.q_verb.full(): self.q_verb.get()
            self.q_verb.put(np.array(verb_label_nums_in_batch))
            accumulated_verb_label_nums = np.sum(self.q_verb.queue, axis=0)
            verb_weights = self.cal_weights(accumulated_verb_label_nums, p=self.p_verb)

            aphal = min(math.pow(0.999, self.q_verb.qsize()),0.9)
            verb_weights = aphal * self.verb_weights_init + (1 - aphal) * verb_weights

        src_logits = src_logits.sigmoid()
        loss_verb_ce = self._neg_loss(src_logits, target_classes, uncertainty, weights=verb_weights, alpha=self.alpha)

        losses = {'loss_verb_ce': loss_verb_ce}
        return losses

    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions):
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses

    def loss_matching_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_matching_logits' in outputs
        src_logits = outputs['pred_matching_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['matching_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_matching = F.cross_entropy(src_logits.transpose(1, 2), target_classes)
        losses = {'loss_matching': loss_matching}

        if log:
            losses['matching_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_verb_uctt(self, outputs, targets, indices, num_interactions):
        assert 'pred_verb_logits' in outputs
        src_logits = outputs['pred_verb_logits']  # [B,num_queries,num_verb_classes]
        uncertainty = None
        if 'uctt_verb' in outputs:
            uncertainty = outputs['uctt_verb']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o
        # target_classes_o: [num_hois_in_batch,num_verb_classes]
        # target_classes: [B,num_queries,num_verb_classes]
        pos_inds = target_classes.gt(0.5)  # [B,num_queries,num_verb_classes]
        uctt_match_list = []
        for i in range(len(indices)):
            uctt_match = uncertainty[i,indices[i][0],:]
            pos_index = pos_inds[i,indices[i][0],:]
            uctt_match = uctt_match[pos_index]
            uctt_match_list.append(uctt_match)
        uctt_match_batch = torch.cat(uctt_match_list)
        if len(uctt_match_batch):
            uctt_avg = torch.log(torch.mean(torch.exp(uctt_match_batch)))
        else:
            uctt_avg = torch.log(torch.mean(torch.exp(uncertainty)))

        # src_logits = src_logits.sigmoid()*torch.exp(-uncertainty)*pos_inds
        #loss_verb_uctt = torch.nn.functional.binary_cross_entropy_with_logits(src_logits, target_classes)

        src_logits = src_logits * torch.exp(-uncertainty)
        src_logits = src_logits.sigmoid() * pos_inds
        loss_verb_uctt = torch.nn.functional.binary_cross_entropy(src_logits, target_classes) * torch.exp(-uctt_avg)
        losses = {'loss_uctt': uctt_avg,
                  'loss_verb_uctt': uctt_avg+loss_verb_uctt}
        return losses

    def _neg_loss(self, pred, gt, uctt, weights=None, alpha=0.25):
        pos_inds = gt.gt(0.5).float()
        neg_inds = gt.lt(0.5).float()
        '''
        if uctt is not None:
            pred = pred * torch.exp(-uctt)
        pred = pred.sigmoid()
        '''

        loss = 0

        pos_loss = alpha * torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds  # [B,num_queries,num_verb_classes]
        if weights is not None:
            pos_loss = pos_loss * weights[:-1]

        neg_loss = (1 - alpha) * torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds  # [B,num_queries,num_verb_classes]

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        loss_map = {
            'obj_labels': self.loss_obj_labels,
            'obj_cardinality': self.loss_obj_cardinality,
            'verb_labels': self.loss_verb_labels,
            'sub_obj_boxes': self.loss_sub_obj_boxes,
            'matching_labels': self.loss_matching_labels,
            'verb_uctt': self.loss_verb_uctt,
            'uctt': None
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        indices = self.matcher(outputs_without_aux, targets)

        num_interactions = sum(len(t['obj_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            if loss != 'uctt':
                losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'obj_labels':
                        kwargs = {'log': False}
                    if not loss == 'uctt':
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions, **kwargs)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

        return losses


class PostProcessHOI(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.subject_category_id = args.subject_category_id
        self.use_matching = args.use_matching

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_obj_logits = outputs['pred_obj_logits']
        out_verb_logits = outputs['pred_verb_logits']
        out_sub_boxes = outputs['pred_sub_boxes']
        out_obj_boxes = outputs['pred_obj_boxes']

        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        obj_prob = F.softmax(out_obj_logits, -1)
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

        verb_scores = out_verb_logits.sigmoid()

        if self.use_matching:
            out_matching_logits = outputs['pred_matching_logits']
            matching_scores = F.softmax(out_matching_logits, -1)[..., 1]

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for index in range(len(obj_scores)):
            os, ol, vs, sb, ob =  obj_scores[index], obj_labels[index], verb_scores[index], sub_boxes[index], obj_boxes[index]
            sl = torch.full_like(ol, self.subject_category_id)
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

            vs = vs * os.unsqueeze(1)

            if self.use_matching:
                ms = matching_scores[index]
                vs = vs * ms.unsqueeze(1)

            ids = torch.arange(b.shape[0])

            results[-1].update({'verb_scores': vs.to('cpu'), 'sub_ids': ids[:ids.shape[0] // 2],
                                'obj_ids': ids[ids.shape[0] // 2:]})

        return results


def build(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)

    cdn = build_cdn(args)

    if args.compo:
        model = CDNHOICompo(
            backbone,
            cdn,
            num_obj_classes=args.num_obj_classes,
            num_verb_classes=args.num_verb_classes,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
            fusion_mode=args.fusion_mode,
            args=args
        )
    else:
        model = CDNHOI2(
            backbone,
            cdn,
            num_obj_classes=args.num_obj_classes,
            num_verb_classes=args.num_verb_classes,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
            args=args
        )

    matcher = build_matcher(args)
    weight_dict = {}
    weight_dict['loss_obj_ce'] = args.obj_loss_coef
    weight_dict['loss_verb_ce'] = args.verb_loss_coef
    weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
    weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
    weight_dict['loss_sub_giou'] = args.giou_loss_coef
    weight_dict['loss_obj_giou'] = args.giou_loss_coef
    if args.uncertainty:
        weight_dict['loss_verb_uctt'] = 1
        weight_dict['loss_uctt'] = 0
    if args.use_matching:
        weight_dict['loss_matching'] = args.matching_loss_coef

    if args.aux_loss:
        min_dec_layers_num = min(args.dec_layers_hopd, args.dec_layers_interaction)
        aux_weight_dict = {}
        for i in range(min_dec_layers_num - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['obj_labels', 'verb_labels', 'sub_obj_boxes', 'obj_cardinality']
    if args.use_matching:
        losses.append('matching_labels')
    if args.uncertainty:
        losses.append('verb_uctt')
        losses.append('uctt')

    criterion = SetCriterionHOI(args.num_obj_classes, args.num_queries, args.num_verb_classes, matcher=matcher,
                                weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses,
                                args=args)

    criterion.to(device)
    postprocessors = {'hoi': PostProcessHOI(args)}

    return model, criterion, postprocessors
