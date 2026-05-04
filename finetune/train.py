import datetime
import random
import sys
import math
import os
import time
import warnings
import copy
import pickle
import torch.nn.functional as F
import torch
import torch.utils.data
import torchvision
import utils
from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from collections import OrderedDict
from typing import Dict, Callable, Optional, Any, Tuple, Union
import numpy as np
import statistics
import logging
from visdial import VisDial_train, VisDial_test
import torch
import torch.utils.data
from torch.nn.functional import normalize
import h5py
from transformers import AutoProcessor, BlipForImageTextRetrieval
from easydict import EasyDict as edict
from model_components import LinearLayer, BertAttention_
import json

class BlipForRetrieval(BlipForImageTextRetrieval):
    def get_text_features(self,
                          input_ids: torch.LongTensor,
                          attention_mask: Optional[torch.LongTensor] = None,
                          return_dict: Optional[bool] = None,
                          ) -> torch.FloatTensor:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        question_embeds = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state
        return self.text_proj(question_embeds[:, 0, :]), question_embeds[:, 0, :], question_embeds
       
    def get_image_features(
            self,
            pixel_values: torch.FloatTensor,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        with torch.no_grad():
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            image_embeds = vision_outputs[0]

            image_feat = normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        return image_feat








class MAQIU(nn.Module):
    def __init__(self, args, device):
        super(MAQIU, self).__init__()
        self.drop = 0.2
        self.cap_dim = 768
        self.out_dim = 256
        self.txt_history_compress = BertAttention_(edict(hidden_size=self.cap_dim, intermediate_size=self.cap_dim,
                                           hidden_dropout_prob=self.drop, num_attention_heads=4,
                                           attention_probs_dropout_prob=self.drop))
        
        self.txt_history_tokens = nn.Parameter(torch.randn(1, 36, self.cap_dim))

        self.attn_score_mapping = nn.Linear(self.cap_dim, out_features=1, bias=False)
        self.txt_out_mapping = nn.Linear(self.cap_dim,self.out_dim)
        self.candidate_img_mapping = LinearLayer(self.out_dim,self.cap_dim)
        self.candidate_guide_txt_encoder = BertAttention_(edict(hidden_size=self.cap_dim, intermediate_size=self.cap_dim,
                                           hidden_dropout_prob=self.drop, num_attention_heads=4,
                                           attention_probs_dropout_prob=self.drop))
        self.reset_parameters()
        self.blip_backbone = BlipForRetrieval.from_pretrained("blip-itm-large-coco")
        self.processor = AutoProcessor.from_pretrained("blip-itm-large-coco")
        self.img_candidates = {}
        self.candidates_ids = {}
        self.candidates_ids_to_idx = {}
        self.candidates_idx_to_ids = {}

        img_feats_train_ids, img_feats_train = torch.load('../cache/visdial_img_train.pth')
        self.img_candidates['train'] = img_feats_train.to(device)
        print(self.img_candidates['train'].shape)
        self.candidates_ids['train']= img_feats_train_ids
        self.candidates_ids_to_idx['train'] = {}
        self.candidates_idx_to_ids['train'] = {}
        for idx, img_id in enumerate(img_feats_train_ids):
            self.candidates_ids_to_idx['train'][img_id] = idx
            self.candidates_idx_to_ids['train'][idx] = img_id

        val_time = time.time()
        img_feats_val_ids, img_feats_val = torch.load('../cache/visdial_img_val.pth')
        self.img_candidates['val'] = normalize(img_feats_val, dim=-1).to(device)
        print(self.img_candidates['val'].shape)
        self.candidates_ids['val']= img_feats_val_ids
        self.candidates_ids_to_idx['val'] = {}
        self.candidates_idx_to_ids['val'] = {}
        for idx, img_id in enumerate(img_feats_val_ids):
            self.candidates_ids_to_idx['val'][img_id] = idx
            self.candidates_idx_to_ids['val'][idx] = img_id
        print(time.time()-val_time)
        test_time = time.time()
        with open('../Protocol/Search_Space_val_50k.json') as f:
                corpus = json.load(f)
        self.candidates_ids['test'] = []
        self.candidates_ids_to_idx['test'] = {}
        self.candidates_idx_to_ids['test'] = {}
        for idx, path in enumerate(corpus):
            img_id = path.split('/',1)[-1]
            self.candidates_ids['test'].append(img_id)
            self.candidates_ids_to_idx['test'][img_id] = idx
            self.candidates_idx_to_ids['test'][idx] = img_id
        self.img_candidates['test'] = normalize(torch.load('../cache/corpus_blip.pth')[1].to(device),dim=-1)
        print(time.time()-test_time)

        self.data_split = 'train'
        self.candidate_num = args.candidate_num
        self.candidate_info = int(self.candidate_num * 0.1)
        self.sim_min_topk = args.recall_num


    def reset_parameters(self):
        """ Initialize the weights."""

        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(re_init)

    def forward(self, imgs, txts, all_txts, data_ids, dialog_turn, txt_history_tokens=None, init_txt_features=None, memory_bank=None, candidates_imgs=None, candidate_img_labels=None):
        img_feats = imgs.squeeze()
        txts = self.processor(text=txts, padding=True, return_tensors='pt')
        global_txt_features_mapping, global_txt_features, txt_features = self.blip_backbone.get_text_features(**txts)
        cap_attn_mask = txts['attention_mask'].to(txt_features.device)
        if dialog_turn == 0:
            target_ids_idx = [self.candidates_ids_to_idx[self.data_split][i] for i in data_ids]
            init_scores = normalize(global_txt_features_mapping, dim=-1) @ normalize(self.img_candidates[self.data_split],dim=-1).t()
            candidate_scores, candidate_img_idx = torch.topk(init_scores, k=self.candidate_num, dim=-1)
            labels = torch.zeros(global_txt_features_mapping.shape[0],self.candidate_num).int().to(txt_features.device)
            for i, target_id in enumerate(target_ids_idx):
                if target_id not in candidate_img_idx[i].tolist():
                    candidate_img_idx[i][-1] = target_ids_idx[i]
                    candidate_scores[i][-1] = init_scores[i, target_ids_idx[i]]
                    labels[i, -1] = 1
                else:
                    labels[i, candidate_img_idx[i].tolist().index(target_ids_idx[i])] = 1
            
            init_candidate_img_feats = self.img_candidates[self.data_split][candidate_img_idx]       
            
            txt_history_tokens = self.txt_history_compress(self.txt_history_tokens.repeat(txt_features.shape[0],1,1),
                                                           txt_features,txt_features, cap_attn_mask.unsqueeze(1))
            txt_attn_scores = F.softmax(self.attn_score_mapping(txt_history_tokens), dim=1)
            txt_global_feature_ = torch.einsum("blm,bld->bmd", txt_attn_scores,txt_history_tokens).squeeze()
            txt_global_feature = self.txt_out_mapping(txt_global_feature_)
            
            
            sim = torch.einsum('bnd,bd->bn', normalize(init_candidate_img_feats, dim=-1), normalize(txt_global_feature,dim=-1))
            
            sorted_sim, rank = torch.sort(sim, dim = 1, descending = True)
            sorted_candidate_features = torch.gather(init_candidate_img_feats, dim=1, index=rank.unsqueeze(-1).expand(-1, -1, 256))
            sorted_labels = torch.gather(labels, dim=1, index=rank)
            
            sim_ = normalize(txt_global_feature,dim=-1) @ normalize(img_feats,dim=-1).t()
            return (sorted_sim,sim_), txt_history_tokens, (global_txt_features,global_txt_features), sorted_candidate_features, sorted_labels
            
        else:
            
            txt_history_tokens = self.txt_history_compress(txt_history_tokens, txt_features, txt_features, cap_attn_mask.unsqueeze(1))

            candidate_info = candidates_imgs[:,:self.candidate_info]
            candidate_info = self.candidate_img_mapping(candidate_info)

            short_mem = self.txt_history_compress(init_txt_features, txt_features, txt_features, cap_attn_mask.unsqueeze(1))
            short_mem_attn_scores = F.softmax(self.attn_score_mapping(short_mem), dim=1)
            short_mem_global_feature = torch.einsum("blm,bld->bmd", short_mem_attn_scores,short_mem).squeeze()
            mem_bank_k = short_mem_global_feature
            mem_bank_v = global_txt_features
            mem_bank_unit = (mem_bank_k, mem_bank_v)
            
            if dialog_turn >= 3:
                long_mem_attn_scores = F.softmax(self.attn_score_mapping(txt_history_tokens), dim=1)
                long_mem_global_feature = torch.einsum("blm,bld->bmd", long_mem_attn_scores,txt_history_tokens)
                long_mem_global_feature = normalize(long_mem_global_feature, dim=-1)

                memory_bank_k = torch.stack([unit[0] for unit in memory_bank], dim=1)
                memory_bank_v = torch.stack([unit[1] for unit in memory_bank], dim=1)
                candidate_memory = normalize(memory_bank_k, dim=-1).permute(0,2,1)
                
                
                retrieval_memory_sim =  torch.bmm(long_mem_global_feature, candidate_memory).squeeze()
                weight, retrieval_memory_idx = torch.topk(retrieval_memory_sim, dim=-1, k=min(self.sim_min_topk,retrieval_memory_sim.shape[-1]), largest=False)
                weight = 1 - F.softmax(weight, dim=-1)
                    
                all_feats = []
                for i in range(retrieval_memory_idx.shape[-1]):
                    retrieval_memory_idx_ = retrieval_memory_idx[:,i]
                    global_retrieval_qa_features = memory_bank_v[torch.arange(txt_history_tokens.shape[0]), retrieval_memory_idx_]
                    all_feats.append(global_retrieval_qa_features*weight[:,i].unsqueeze(1))
                retrieval_qa_features = torch.sum(torch.stack(all_feats, dim=1), dim=1).unsqueeze(1) 
                retrieval_aug_memory = self.txt_history_compress(txt_history_tokens, retrieval_qa_features, retrieval_qa_features)
                candidate_guided_tokens = self.candidate_guide_txt_encoder(retrieval_aug_memory, candidate_info, candidate_info)
                long_mem = retrieval_aug_memory
            else:
                candidate_guided_tokens = self.candidate_guide_txt_encoder(txt_history_tokens, candidate_info, candidate_info)
                long_mem = txt_history_tokens
            
            txt_attn_scores = F.softmax(self.attn_score_mapping(candidate_guided_tokens), dim=1)
            txt_global_feature = torch.einsum("blm,bld->bmd", txt_attn_scores,candidate_guided_tokens).squeeze()
            txt_global_feature = self.txt_out_mapping(txt_global_feature)

            sim = torch.einsum('bnd,bd->bn', normalize(candidates_imgs, dim=-1), normalize(txt_global_feature,dim=-1))
            sorted_sim, rank = torch.sort(sim, dim = 1, descending = True)
            sorted_candidate_features = torch.gather(candidates_imgs, dim=1, index=rank.unsqueeze(-1).expand(-1, -1, 256))
            sorted_labels = torch.gather(candidate_img_labels, dim=1, index=rank)

            sim_ = normalize(txt_global_feature,dim=-1) @ normalize(img_feats,dim=-1).t()
            return (sorted_sim,sim_), long_mem, mem_bank_unit, sorted_candidate_features, sorted_labels
            

    def test_forward(self, all_txts, data_ids, restrict_pool=False):
        txts = self.processor(text=all_txts[0], padding=True, return_tensors='pt')

        global_txt_features_mapping,global_txt_features, txt_features = self.blip_backbone.get_text_features(**txts)
        cap_attn_mask = txts['attention_mask'].to(txt_features.device)

        dl_recalls = []
        target_idxs = torch.tensor([self.candidates_ids_to_idx[self.data_split][p] for p in data_ids]).unsqueeze(1).to(txt_features.device)
        memory_bank = [(global_txt_features,global_txt_features)]

        txt_history_tokens = self.txt_history_compress(self.txt_history_tokens.repeat(txt_features.shape[0],1,1),
                                                           txt_features,txt_features, cap_attn_mask.unsqueeze(1))
        txt_attn_scores = F.softmax(self.attn_score_mapping(txt_history_tokens), dim=1)
        txt_global_feature = torch.einsum("blm,bld->bmd", txt_attn_scores,txt_history_tokens).squeeze()

        init_txt_features = txt_history_tokens
        txt_global_feature = self.txt_out_mapping(txt_global_feature)

        start_sims = normalize(txt_global_feature,dim=-1) @ self.img_candidates[self.data_split].t()
        start_ranks =  torch.argsort(start_sims, descending=True, dim=1).long()
        start_target_recall = ((start_ranks - target_idxs) == 0).nonzero()[:, 1].unsqueeze(1)
        
        dl_recalls.append(start_target_recall)
        if restrict_pool:
            candidate_img_idx = start_ranks[:,:self.candidate_num]
            candidate_img_feats = self.img_candidates[self.data_split][candidate_img_idx]
        else:
            guide_img_idx = start_ranks[:,:self.candidate_info]
            guide_feats = self.img_candidates[self.data_split][guide_img_idx]
            
                

        for i in range(1,11):
            txts = self.processor(text=all_txts[i], padding=True, return_tensors='pt') 
            global_txt_features_mapping,global_txt_features, txt_features = self.blip_backbone.get_text_features(**txts)
            cap_attn_mask = txts['attention_mask'].to(txt_features.device)

            txt_history_tokens = self.txt_history_compress(txt_history_tokens,txt_features,txt_features, cap_attn_mask.unsqueeze(1))

            short_mem = self.txt_history_compress(init_txt_features, txt_features, txt_features, cap_attn_mask.unsqueeze(1))
            short_mem_attn_scores = F.softmax(self.attn_score_mapping(short_mem), dim=1)
            short_mem_global_feature = torch.einsum("blm,bld->bmd", short_mem_attn_scores,short_mem).squeeze()

            memory_bank_unit = (short_mem_global_feature, global_txt_features)

            candidate_info = self.candidate_img_mapping(guide_feats)
            if i >= 3:
                long_mem_attn_scores = F.softmax(self.attn_score_mapping(txt_history_tokens), dim=1)
                long_mem_global_feature = torch.einsum("blm,bld->bmd", long_mem_attn_scores,txt_history_tokens)
                long_mem_global_feature = normalize(long_mem_global_feature, dim=-1)

                
                memory_bank_k = torch.stack([unit[0] for unit in memory_bank], dim=1)
                memory_bank_v = torch.stack([unit[1] for unit in memory_bank], dim=1)
                candidate_memory = normalize(memory_bank_k, dim=-1).permute(0,2,1)

                retrieval_memory_sim =  torch.bmm(long_mem_global_feature, candidate_memory).squeeze()
                
                weight, retrieval_memory_idx = torch.topk(retrieval_memory_sim, dim=-1, k=min(self.sim_min_topk,retrieval_memory_sim.shape[-1]), largest=False)
                weight = 1 - F.softmax(weight, dim=-1)
                    
                all_feats = []
                for j in range(retrieval_memory_idx.shape[-1]):
                    retrieval_memory_idx_ = retrieval_memory_idx[:,j]
                    global_retrieval_qa_features = memory_bank_v[torch.arange(txt_history_tokens.shape[0]), retrieval_memory_idx_]
                    all_feats.append(global_retrieval_qa_features*weight[:,j].unsqueeze(1))
                retrieval_qa_features = torch.sum(torch.stack(all_feats, dim=1), dim=1).unsqueeze(1) 
                retrieval_aug_memory = self.txt_history_compress(txt_history_tokens, retrieval_qa_features, retrieval_qa_features)
                txt_history_tokens = retrieval_aug_memory
                candidate_guided_tokens = self.candidate_guide_txt_encoder(retrieval_aug_memory, candidate_info, candidate_info)
            else:
                candidate_guided_tokens = self.candidate_guide_txt_encoder(txt_history_tokens, candidate_info, candidate_info)

            txt_attn_scores = F.softmax(self.attn_score_mapping(candidate_guided_tokens), dim=1)
            txt_global_feature = torch.einsum("blm,bld->bmd", txt_attn_scores,candidate_guided_tokens).squeeze()
            txt_global_feature = self.txt_out_mapping(txt_global_feature)

            sims = normalize(txt_global_feature,dim=-1) @ self.img_candidates[self.data_split].t()
            ranks =  torch.argsort(sims, descending=True, dim=1).long()

            target_recall = ((ranks - target_idxs) == 0).nonzero()[:, 1].unsqueeze(1)
            dl_recalls.append(target_recall)
            memory_bank.append(memory_bank_unit)
            guide_img_idx = ranks[:,:self.candidate_info]
            guide_feats = self.img_candidates[self.data_split][guide_img_idx]



        dl_recalls = torch.cat(dl_recalls, dim=1)
        
        return dl_recalls

    def backbone_forward(self, imgs, txts, data_ids):
        txts = self.processor(text=txts, padding=True, return_tensors='pt') 
        global_txt_feats, txt_feats = self.blip_backbone.get_text_features(**txts)
        imgs = imgs.to(global_txt_feats.device).squeeze()
        sims = normalize(global_txt_feats, dim=-1) @ normalize(imgs, dim=-1).t()
        return sims

    def backbone_test_forward(self, all_txts, data_ids,device):
        dl_recalls = []
        target_idxs = torch.tensor([self.candidates_ids_to_idx[self.data_split][p] for p in data_ids]).unsqueeze(1).to(device)
        
        for i in range(11):
            txts = self.processor(text=all_txts[i], padding=True, return_tensors='pt')
            global_txt_features, txt_features = self.blip_backbone.get_text_features(**txts)
            
            sims = normalize(global_txt_features,dim=-1) @ self.img_candidates[self.data_split].t()
            ranks =  torch.argsort(sims, descending=True, dim=1).long()
            target_recall = ((ranks - target_idxs) == 0).nonzero()[:, 1].unsqueeze(1)
            dl_recalls.append(target_recall)
        dl_recalls = torch.cat(dl_recalls, dim=1)
        return dl_recalls



def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)


class Contrastive(torch.nn.Module):
    def __init__(self,device,temp=0.03):
        super(Contrastive, self).__init__()
        self.temp = temp
        self.xent = nn.CrossEntropyLoss()
        self.device = device
    
    def forward(self, sims, _):

        targets = torch.arange(sims.size(0), device=self.device)

        sim_i2t = sims / self.temp
        sim_t2i = sims.t() / self.temp
        
        loss_i2t = self.xent(sim_i2t, targets)
        loss_t2i = self.xent(sim_t2i, targets)

        return (loss_i2t + loss_t2i) / 2


class Contrastive2(torch.nn.Module):
    def __init__(self,temp=0.03):
        super(Contrastive2, self).__init__()
        self.temp = temp
        self.xent = nn.CrossEntropyLoss()
    
    def forward(self, sims, targets):
        targets = targets.argmax(dim=1)
        
        sim_t2i = sims / self.temp
        loss = self.xent(sim_t2i ,targets)
        return loss


def train_one_epoch(model, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None, processor=None, loss_func1=None, loss_func2=None):
    model.train()
    model.module.data_split = 'train'
    metric_logger = utils.MetricLogger(delimiter="  ")
    if args.backbone_ft:
        metric_logger.add_meter("lr_backbone", utils.SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("lr_head", utils.SmoothedValue(window_size=1, fmt="{value}"))
    else:
        metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))
    # with torch.autograd.set_detect_anomaly(True):
    header = f"Epoch: [{epoch}]"
    
    for i, (imgs, txts, data_ids) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        dialog_len = random.randint(1,11)
        
        all_loss = []
        memory_bank = []
        init_txt_feats = None
        for dialog_turn in range(dialog_len):
            dialog_txts = txts[dialog_turn]
            
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                if dialog_turn == 0:
                    sims, txt_history_feature, init_txt_global_feature, sorted_candidate_features, sorted_labels = model(imgs, dialog_txts,txts[1:], data_ids, dialog_turn)
                    init_txt_feats = txt_history_feature
                    memory_bank.append(init_txt_global_feature)
                else:
                    sims, txt_history_feature, short_mem, sorted_candidate_features, sorted_labels = model(imgs, dialog_txts,txts, data_ids, dialog_turn,
                                                                                        txt_history_feature,init_txt_feats,memory_bank,sorted_candidate_features, sorted_labels)
                    memory_bank.append(short_mem)

                sum_loss = loss_func1(sims[0], sorted_labels) + loss_func2(sims[1],sorted_labels)
                all_loss.append(sum_loss)
        
        loss = torch.stack(all_loss).mean()
        
        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

            optimizer.step()

        batch_size = imgs.shape[0]
        if args.backbone_ft:
            metric_logger.update(loss=loss.item(), lr_backbone=optimizer.param_groups[0]["lr"], lr_head=optimizer.param_groups[-1]["lr"])
        else:
            metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

def evaluate(model, data_loader, device, args, log_suffix="", loss_func1=None, loss_func2=None):
    model.eval()
    model.module.data_split = 'val'
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        txt_history = []
        candidate_feat = []
        labels = []

        all_memory_bank = {i:[] for i in range(11)}
        all_init_text_feat = []

        for i in range(11):
            # data_loader.dataset.dialog_len = i
            batch_id = 0
            for imgs, txts,  data_ids in metric_logger.log_every(data_loader, args.print_freq, header):
                start_time = time.time()
                if i == 0:
                    sims, txt_history_feature, txt_history_global_feature, sorted_candidate_features, sorted_labels = model(imgs,txts[i],txts,data_ids,i)
                    txt_history.append(txt_history_feature)
                    candidate_feat.append(sorted_candidate_features)
                    labels.append(sorted_labels)
                    all_memory_bank[i].append(txt_history_global_feature)
                    all_init_text_feat.append(txt_history_feature)
                else:
                    init_txt_feat = all_init_text_feat[batch_id]
                    memory_bank = [all_memory_bank[j][batch_id] for j in range(i)]
                    
                    sims, long_mem, short_mem, sorted_candidate_features, sorted_labels = model(imgs, txts[i], txts, data_ids, i,
                                                                                      txt_history[batch_id],init_txt_feat,memory_bank, candidate_feat[batch_id],labels[batch_id])
                    txt_history[batch_id] = long_mem
                    candidate_feat[batch_id] = sorted_candidate_features
                    labels[batch_id] = sorted_labels
                    all_memory_bank[i].append(short_mem)
                 

                loss = loss_func1(sims[0],sorted_labels) + loss_func2(sims[1],sorted_labels)

                batch_id += 1
                batch_size = imgs.shape[0]
                metric_logger.update(loss=loss.item())
                metric_logger.meters[f"loss{i}"].update(loss.item())
                metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
                num_processed_samples += batch_size
            logging.info("Test loss at dialog_len %d : %f"%(i, metric_logger.meters[f"loss{i}"].global_avg))

    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    return metric_logger.loss.global_avg

def evaluate_test(model, data_loader, device, args, log_suffix="", processor=None):
    model.eval()
    model.module.data_split = 'test'
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"
    num_processed_samples = 0
    with torch.inference_mode():
        dl_recalls = []
        for _, txts, data_ids in metric_logger.log_every(data_loader, args.print_freq, header):
            txts = txts[:]
            dl_recalls_ = model.module.test_forward(txts, data_ids)
            dl_recalls.append(dl_recalls_.cpu())

        dl_recalls = torch.cat(dl_recalls, dim=0)
        hits_results = []
        num_rounds = 11
        min_ranks = []
        for dl in range(num_rounds):
            logging.info(f"Calculate recalls for each dialogues of length {dl}...")
            dialog_recalls = dl_recalls[:,dl]
            if dl == 0:
                min_ranks.append(dialog_recalls)
            else:
                min_ranks.append(torch.minimum(min_ranks[dl-1], dialog_recalls))
            hits_results.append(dialog_recalls)

        hits_results, temp_hits_results = cumulative_hits_per_round(torch.cat(hits_results), num_rounds, hitting_recall=1)
        hits_results = hits_results.tolist()
        temp_hits_results = temp_hits_results.tolist()
        logging.info(f"====== Results for Hits@{10} ====== ")
        for dl in range(num_rounds):
            logging.info(f"\t Dialog Length: {dl}: {round(hits_results[dl], 2)}%")
        logging.info(f"====== Results for Recall@{10} ====== ")
        for dl in range(num_rounds):
            logging.info(f"\t Dialog Length: {dl}: {round(temp_hits_results[dl], 2)}%")
     
    return sum(temp_hits_results) / len(temp_hits_results)
    

def backbone_evaluate(model, data_loader, device, args, log_suffix="", processor=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for i in range(11):
            data_loader.dataset.dialog_len = i
            for imgs, txts, data_ids in metric_logger.log_every(data_loader, args.print_freq, header):
                start_time = time.time()
                txts = model.module.processor(text=txts, padding=True, return_tensors='pt')
                
                txts = txts.to(device)
                # img_features = model.module.get_image_features(imgs)
                txt_features,_ = model.module.blip_backbone.get_text_features(**txts)
                img_features = imgs.squeeze().to(txt_features.device)
                sims = normalize(txt_features, dim=-1) @ normalize(img_features, dim=-1).t()
                loss = Contrastive(sims, args.temp, device)
                # loss = criterion(img_features, txt_features, args.temp, device, args)

                # FIXME need to take into account that the datasets
                # could have been padded in distributed setup
                batch_size = imgs.shape[0]
                metric_logger.update(loss=loss.item())
                metric_logger.meters[f"loss{i}"].update(loss.item())
                metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
                num_processed_samples += batch_size
            logging.info("Test loss at dialog_len %d : %f"%(i, metric_logger.meters[f"loss{i}"].global_avg))
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    return metric_logger.loss.global_avg


def get_first_hitting_time(target_recall, num_rounds, hitting_recall=10):
    """ returns (11, n) tensor with hitting time in each round (0, 11). inf indicate a miss (no hit after 11 rounds) """
    target_recalls = target_recall.view(num_rounds, -1).T
    hits = (target_recalls < hitting_recall)

    final_hits = torch.inf * torch.ones(target_recalls.shape[0])

    hitting_times = []
    temp_hitting_times = []
    for ro_i in range(num_rounds):
        temp_hits = torch.inf * torch.ones(target_recalls.shape[0])
        rh = hits[:, ro_i]
        final_hits[rh] = torch.min(final_hits[rh], torch.ones(final_hits[rh].shape) * ro_i)
        temp_hits[rh] = torch.min(temp_hits[rh], torch.ones(temp_hits[rh].shape) * ro_i)
        hitting_times.append(final_hits.clone())
        temp_hitting_times.append(temp_hits)

    return torch.stack(hitting_times), torch.stack(temp_hitting_times)


def cumulative_hits_per_round(target_recall, num_rounds,  hitting_recall=10):
    """ return calculation of avg number of hits until round x"""
    if type(hitting_recall) is tuple:
        assert len(hitting_recall) == 1
        hitting_recall = hitting_recall[0]

    ht_times, temp_ht_times = get_first_hitting_time(target_recall, num_rounds, hitting_recall)

    return ((ht_times < torch.inf).sum(dim=-1) * 100 / ht_times[0].shape[0]), ((temp_ht_times < torch.inf).sum(dim=-1) * 100 / temp_ht_times[0].shape[0])


def load_data(args):
    # Data loading code
    logging.info("Loading data")

    interpolation = InterpolationMode(args.interpolation)

    logging.info("Loading training data")
    st = time.time()
    name = args.data_path.split('/')[-1]
    
    dataset_train = VisDial_train(args.data_path, 'train')
    dataset_val = VisDial_train(args.data_path, 'val')
    dataset_test = VisDial_test()
    logging.info(f"Took {time.time() - st}")

    logging.info("Creating data loaders")
    logging.info('distributed:')
    logging.info(args.distributed)
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset_train, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        val_sampler = torch.utils.data.SequentialSampler(dataset_val)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset_train, dataset_val, train_sampler, val_sampler, dataset_test, test_sampler
def main(args):
    checkpoint = None
    if args.run_id is None:
        now = time.localtime()
        args.output_dir = os.path.join(args.output_dir,time.strftime("%Y-%m-%d-%H-%M-%S", now))
    else:
        args.output_dir = os.path.join(args.output_dir, args.run_id)
    
    
    if args.output_dir:
        utils.mkdir(args.output_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'training.log')),
            logging.StreamHandler()
        ])
    logger = logging.getLogger()
    

    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    logging.info("Creating model")
    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)
    model_set = {'MAQIU':MAQIU,}
    model = model_set[args.model_name](args, device)
    model.to(device)


    if args.distributed and args.sync_bn:  
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    dataset, dataset_val, train_sampler, val_sampler, dataset_test, test_sampler = load_data(args)
    logging.info(args)     

    collate_fn = None
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=False,
        collate_fn=collate_fn
    )
    data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=args.test_batch_size, sampler=val_sampler, num_workers=args.workers, pin_memory=True
    )

    data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=args.test_batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))

    '''
    TODO: train only text encoder
    '''
    for name, p in model.named_parameters():
        if 'blip_backbone' in name:
            if args.backbone_ft:
                if 'text' in name:
                    print(f'update {name}')
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            else:
                p.requires_grad = False
        else:
            
            print(f'update {name}')
            p.requires_grad = True


    
    parameters = utils.set_weight_decay_blip(
            model)    
   

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=False, find_unused_parameters=True)
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent from other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    

    
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        checkpoint = checkpoint["model_ema"]
        state_dict = {}
        for k, v in checkpoint.items():
            if 'heads' not in k:
                state_dict[k] = v
        msg = model.load_state_dict(state_dict, strict=False)
        logging.info("Load pretrained model with msg: {}".format(msg))

    if args.test_model:
        ckpt = torch.load(os.path.join(args.output_dir, 'best_model.pth'), map_location="cpu")
        state_dict = ckpt['model']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict['module.' + k] = v
        
        msg = model.load_state_dict(new_state_dict, strict=False)
        evaluate_test(model, data_loader_test, device=device, args=args)
        exit()

    logging.info("Start training")
    start_time = time.time()
    best = 99999

    loss_func1 = Contrastive2()
    loss_func_2 = Contrastive(device)

    for epoch in range(args.start_epoch, args.epochs):
        logging.info('epoch %d'%epoch)
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args, model_ema, scaler,loss_func1, loss_func_2)
        lr_scheduler.step()
        
        test_loss = evaluate(model, data_loader_val, device=device, args=args, loss_func1=loss_func1, loss_func2=loss_func2)

        if model_ema:
            ema_test_loss = evaluate(model_ema, data_loader_val, device=device, log_suffix="EMA", processor=processor, args=args)
        else:
            ema_test_loss = 99999
        if args.output_dir and best > min(test_loss, ema_test_loss):
            logging.info(f"Save the checkpoint at epoch {epoch}")
            best = min(test_loss, ema_test_loss)
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            utils.save_on_master(checkpoint, os.path.join(args.output_dir,  f"best_model.pth"))
            del checkpoint
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    ''' Add '''
    parser.add_argument("--torch-seed", type=int, default=42)
    parser.add_argument("--num-exp", type=int)
    parser.add_argument("--save-after-epoch", type=int, default=100)
    parser.add_argument("--pretrained", type=str)
    parser.add_argument("--loss", type=str, default="recall", choices=["contrastive", "recall"])
    parser.add_argument("--temp", type=float, default=1)

    parser.add_argument("--data-path", default="VisDial", type=str, help="dataset path")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=512, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--test-batch-size", default=100, type=int)
    parser.add_argument("--epochs", default=36, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=8, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="adamw", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.0000125, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--lr_scheduler", default="exponentiallr", type=str, help="the lr scheduler")
    parser.add_argument("--lr_warmup_epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr_warmup_method", default="linear", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.93, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="./res", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument("--clip_grad_norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--run_id", default=None, type=str)
    parser.add_argument("--backbone_ft", action='store_true')
    parser.add_argument("--model_name", default='MAQIU', type=str)
    parser.add_argument("--test_model", action='store_true')
    parser.add_argument("--recall_num", default=2, type=int)
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
