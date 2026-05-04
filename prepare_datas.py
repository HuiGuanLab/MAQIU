from transformers import AutoProcessor, BlipForImageTextRetrieval
import os
import json
import torch
from typing import Dict, Callable, Optional, Any, Tuple, Union
import numpy as np
import h5py
from PIL import Image
from torch.nn.functional import normalize


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
        return question_embeds

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
            return image_embeds
            image_feat = normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        return image_feat
device = 'cuda:0'
model = BlipForRetrieval.from_pretrained("blip-itm-large-coco").to(device)
processor = AutoProcessor.from_pretrained("blip-itm-large-coco")
model.to(device)
split = 'train'
root = os.path.join('../images/visdial_1.0_%s' % split)
_data_path = os.path.join(root)
with open(os.path.join(_data_path, f'visdial_1.0_{split}.json'), "r") as f:
    data = json.load(f)

image_id_to_file_name = {}
if split == 'val':
    prefix = 'VisualDialog_val2018_'
    for dialog in data['data']['dialogs']:
        image_id = dialog['image_id']
        file_name = prefix + "%012d.jpg" % image_id
        image_id_to_file_name[image_id] = file_name
else:
    prefix_train = 'COCO_train2014_'
    prefix_val = 'COCO_val2014_'
    for dialog in data['data']['dialogs']:
        image_id = dialog['image_id']
        file_name = prefix_train + "%012d.jpg" % image_id
        if os.path.exists('../images/%s' % file_name):
            image_id_to_file_name[image_id] = file_name
        else:
            image_id_to_file_name[image_id] = prefix_val + "%012d.jpg" % image_id
feat_path = h5py.File('visdial_img_%s.hdf5' % split, 'w')
bsz = 256
for i in range((len(data['data']['dialogs']) // 256) + 1):
    images = []
    file_names = []
    for j in range(i*bsz, (i+1)*bsz):
        try:
            file_name = image_id_to_file_name[data['data']['dialogs'][j]['image_id']]
        except:
            print(j)
            break
        image_file = os.path.join(_data_path, 'images', file_name)
        image = Image.open(image_file)
        image = torch.tensor(processor(image)['pixel_values'][0]).to(device).unsqueeze(0)
        images.append(image)
        file_names.append(file_name)
    images = torch.cat(images,dim=0)
    with torch.no_grad():
        img_feats = model.get_image_features(images).cpu().numpy()
        
    for idx, file_name in enumerate(file_names):
        feat_path.create_dataset(file_name, data=img_feats[idx])
    print(i)



