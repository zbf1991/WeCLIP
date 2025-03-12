import torch
import torch.nn as nn
import torch.nn.functional as F

from .segformer_head import SegFormerHead
import numpy as np
import clip
from clip.clip_text import class_names, new_class_names_coco, BACKGROUND_CATEGORY_COCO
from pytorch_grad_cam import GradCAM
from clip.clip_tool import generate_cam_label, generate_clip_fts, perform_single_coco_cam
import os
from torchvision.transforms import Compose, Normalize
from .Decoder.TransDecoder import DecoderTransformer
from WeCLIP_Plus.PAR import PAR

def Normalize_clip():
    return Compose([
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result



def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights.t()


def _refine_cams(ref_mod, images, cams, valid_key):
    images = images.unsqueeze(0)
    cams = cams.unsqueeze(0)

    refined_cams = ref_mod(images.float(), cams.float())
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label.squeeze(0)


class WeCLIP_Plus(nn.Module):
    def __init__(self, num_classes=None, clip_model=None, dino_model=None, dino_fts_dim=768, decoder_layers=3,
                 embedding_dim=256, in_channels=512, dataset_root_path=None, clip_flag=16, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.dino_fts_fuse_dim = dino_fts_dim #384 for vit-s, 768for vit-b
        self.clip_flag = clip_flag

        self.encoder, _ = clip.load(clip_model, device=device)

        for name, param in self.encoder.named_parameters():
            if clip_flag == 14 and '23' not in name:
                param.requires_grad=False
            if clip_flag == 16 and "11" not in name:
                 param.requires_grad=False

        for name, param in self.encoder.named_parameters():
            print(name, param.requires_grad)

        self.dino_encoder = torch.hub.load('facebookresearch/dinov2', dino_model)


        for name, param in self.dino_encoder.named_parameters():
            param.requires_grad = False

        self.in_channels = in_channels

        self.decoder_fts_fuse = SegFormerHead(in_channels=self.in_channels,embedding_dim=self.embedding_dim,
                                              num_classes=self.num_classes, index=1) #index=11
        
        self.dino_decoder_fts_fuse = SegFormerHead(in_channels=[self.dino_fts_fuse_dim, self.dino_fts_fuse_dim, self.dino_fts_fuse_dim, self.dino_fts_fuse_dim], embedding_dim=self.embedding_dim,
                                              num_classes=self.num_classes, index=1)
        
        self.decoder = DecoderTransformer(width=self.embedding_dim, layers=decoder_layers, heads=8, output_dim=self.num_classes)

        self.bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY_COCO, ['a clean origami {}.'],
                                               self.encoder)  # ['a rendering of a weird {}.'], model)
        self.fg_text_features = zeroshot_classifier(new_class_names_coco, ['a clean origami {}.'],
                                               self.encoder)  # ['a rendering of a weird {}.'], model) (20, 512)

        self.root_path = os.path.join(dataset_root_path, 'SegmentationClass')
        self.target_layers = [self.encoder.visual.transformer.resblocks[-1].ln_1]
        self.grad_cam = GradCAM(model=self.encoder, target_layers=self.target_layers, reshape_transform=reshape_transform, clip_flag=clip_flag)
        self.cam_bg_thres = 1
        self.encoder.eval()
        self.par = PAR(num_iter=20, dilations=[1,2,4,8,12,24]).cuda() #1,2,4,8,12,24
        self.iter_num = 0
        self.require_all_fts = True


    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)
        for param in list(self.decoder_fts_fuse.parameters()):
            param_groups[3].append(param)
        for param in list(self.dino_decoder_fts_fuse.parameters()):
            param_groups[3].append(param)

        return param_groups
    


    def forward(self, img, img_names='2007_000032', mode='train'):
        all_img_tokens_list = []
        cam_list = []
        b, c, h, w = img.shape
        self.encoder.eval()
        self.iter_num += 1

        fts_all, attn_weight_list = generate_clip_fts(img, self.encoder, require_all_fts=True, clip_flag=self.clip_flag)

        with torch.no_grad():
            dino_img_h, dino_img_w = (h//14)*14, (w//14)*14
            dino_img = F.interpolate(img, size=(dino_img_h, dino_img_w), mode='bilinear', align_corners=False)
            dino_ftses = self.dino_encoder.forward_features(dino_img)
            dino_fts = dino_ftses['x_norm_patchtokens']

        fts_all_stack = torch.stack(fts_all, dim=0) # (11, hw, b, c)
        attn_weight_stack = torch.stack(attn_weight_list, dim=0).permute(1, 0, 2, 3)
        if self.require_all_fts==True:
            cam_fts_all = fts_all_stack[-1].unsqueeze(0).permute(2, 1, 0, 3) #(1, hw, 1, c)
        else:
            cam_fts_all = fts_all_stack.permute(2, 1, 0, 3)

        all_img_tokens = fts_all_stack[:, 1:, ...]
        img_tokens_channel = all_img_tokens.size(-1)
        all_img_tokens = all_img_tokens.permute(0, 2, 3, 1)
        all_img_tokens = all_img_tokens.reshape(-1, b, img_tokens_channel, h//self.clip_flag, w //self.clip_flag) #(11, b, c, h, w)

        all_img_tokens = all_img_tokens[-1].unsqueeze(0)

        fts = self.decoder_fts_fuse(all_img_tokens)
        _, _, fts_h, fts_w = fts.shape #24

        if isinstance(dino_fts, list):
            for d_i, dino_fts_single in enumerate(dino_fts):
                dino_fts_single = dino_fts_single.reshape([b, dino_img_h // 14, dino_img_w // 14, -1]).permute(0, 3, 1, 2)
                dino_fts[d_i] = dino_fts_single

            dino_fts = torch.stack(dino_fts)
            dino_fts = self.dino_decoder_fts_fuse(dino_fts)
            dino_h, dino_w = dino_img_h // 14, dino_img_w // 14

        else:
            dino_fts = dino_fts.reshape([b, dino_img_h//14, dino_img_w//14, -1]).permute(0,3,1,2)
            _, _, dino_h, dino_w = dino_fts.shape #32
            dino_fts = self.dino_decoder_fts_fuse(dino_fts.unsqueeze(0))
        
        dino_fts = F.interpolate(dino_fts, size=(fts_h, fts_w), mode='bilinear', align_corners=False)


        seg_clip, seg_attn_weight_list_clip = self.decoder(fts)
        seg_dino, seg_attn_weight_list_dino = self.decoder(dino_fts)


        clip_dino_fts = torch.cat([fts, dino_fts], dim=1)

        seg_dino_prob = F.softmax(0.5*seg_dino+0.5*seg_clip, dim=1)
        seg_dino_prob = seg_dino_prob.detach()


        attn_fts = F.interpolate(clip_dino_fts, size=(fts_h, fts_w), mode='bilinear', align_corners=False)
        f_b, f_c, f_h, f_w = attn_fts.shape
        attn_fts_flatten = attn_fts.reshape(f_b, f_c, f_h*f_w)
        attn_pred = attn_fts_flatten.transpose(2, 1).bmm(attn_fts_flatten)
        attn_pred = torch.sigmoid(attn_pred)

        if self.training:

            for i, img_name in enumerate(img_names):
                img_path = os.path.join(self.root_path, 'train', str(img_name) + '.png')
                img_i = img[i]
                cam_fts = cam_fts_all[i]
                cam_attn = attn_weight_stack[i]

                seg_attn = attn_pred.unsqueeze(0)[:, i, :, :]


                # if self.iter_num > 40000 or mode == 'val':
                #     require_seg_trans = True
                #     seg_dino_cam = seg_dino_prob[i]
                # else:
                require_seg_trans = True #False
                seg_dino_cam = seg_dino_prob[i]

                cam_refined_list, keys, w, h = perform_single_coco_cam(img_path, img_i, cam_fts, cam_attn, seg_attn,
                                                                              self.bg_text_features, self.fg_text_features,
                                                                              self.grad_cam,
                                                                              mode=mode,
                                                                              require_seg_trans=require_seg_trans,
                                                                              seg_dino_cam=seg_dino_cam,
                                                                              clip_flag=self.clip_flag
                                                                              )

                cam_dict = generate_cam_label(cam_refined_list, keys, w, h)

                cams = cam_dict['refined_cam'].cuda()

                bg_score = torch.pow(1 - torch.max(cams, dim=0, keepdims=True)[0], self.cam_bg_thres).cuda()

                cams = torch.cat([bg_score, cams], dim=0).cuda()

                valid_key = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
                valid_key = torch.from_numpy(valid_key).cuda()

                with torch.no_grad():
                    cam_labels = _refine_cams(self.par, img[i], cams, valid_key)

                cam_list.append(cam_labels)

            all_cam_labels = torch.stack(cam_list, dim=0)

        if self.training:
            return seg_clip, seg_dino, all_cam_labels, attn_pred
        else:
            return seg_clip, seg_dino

        
    

if __name__=="__main__":

    pretrained_weights = torch.load('pretrained/mit_b1.pth')
    wetr = WeCLIP_Plus('mit_b1', num_classes=20, embedding_dim=256, pretrained=True)
    wetr._param_groups()
    dummy_input = torch.rand(2,3,512,512)
    wetr(dummy_input)