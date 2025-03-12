import torch
import torch.nn as nn
import torch.nn.functional as F

from .segformer_head import SegFormerHead
from . import mix_transformer
import numpy as np
import clip
from clip.clip_text import class_names, new_class_names, BACKGROUND_CATEGORY
from pytorch_grad_cam import GradCAM
from clip.clip_tool import perform_single_voc, perform_single_inference, generate_cam_label, generate_clip_fts, perform_single_voc_fast, perform_single_voc_fast_select
import os
from torchvision.transforms import Compose, Normalize
from .Decoder.TransDecoder import DecoderTransformer
from wetr.PAR import PAR

# from dinov2.models.vision_transformer import vit_base
# dinov2_vitb14 = vit_base(patch_size=14, img_size=518) 
# state_dict_vitb14 = torch.load("dinov2_vitb14_pretrain.pth")
# dinov2_vitb14.load_state_dict(state_dict_vitb14, strict=False)


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
    # print('1111', images.shape, cams.shape)

    refined_cams = ref_mod(images.float(), cams.float())
    # refined_cams = F.interpolate(refined_cams, size=orig_size, mode="bilinear", align_corners=False)
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label.squeeze(0)


class Wetr_CLIP(nn.Module):
    def __init__(self, num_classes=None, clip_model=None, embedding_dim=256, in_channels=512, dataset_root_path=None, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.dino_fts_fuse_dim = 384 #384 for vit-s, 768for vit-b

        self.encoder, _ = clip.load(clip_model, device=device)

        for name, param in self.encoder.named_parameters():
            if "11" not in name:
                param.requires_grad=False

        for name, param in self.encoder.named_parameters():
            print(name, param.requires_grad)

        self.dino_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        # self.dino_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

        for name, param in self.dino_encoder.named_parameters():
            param.requires_grad = False

        # DINOv2
        # dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        # dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        # dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        # dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
        #
        # # DINOv2 with registers
        # dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        # dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        # dinov2_vitl14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        # dinov2_vitg14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')

        self.in_channels = in_channels

        self.decoder_fts_fuse = SegFormerHead(in_channels=self.in_channels,embedding_dim=self.embedding_dim,
                                              num_classes=self.num_classes, index=11)
        
        self.dino_decoder_fts_fuse = SegFormerHead(in_channels=[self.dino_fts_fuse_dim, self.dino_fts_fuse_dim, self.dino_fts_fuse_dim, self.dino_fts_fuse_dim], embedding_dim=self.embedding_dim,
                                              num_classes=self.num_classes, index=1)
        
        self.decoder = DecoderTransformer(width=self.embedding_dim+256, layers=3, heads=8, output_dim=self.num_classes)

        self.bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY, ['a clean origami {}.'],
                                               self.encoder)  # ['a rendering of a weird {}.'], model)
        self.fg_text_features = zeroshot_classifier(new_class_names, ['a clean origami {}.'],
                                               self.encoder)  # ['a rendering of a weird {}.'], model) (20, 512)


        self.target_layers = [self.encoder.visual.transformer.resblocks[-1].ln_1]
        self.grad_cam = GradCAM(model=self.encoder, target_layers=self.target_layers, reshape_transform=reshape_transform)
        self.root_path = os.path.join(dataset_root_path, 'JPEGImages')
        self.cam_bg_thres = 1
        self.encoder.eval()
        self.par = PAR(num_iter=20, dilations=[1,2,4,8,12,24]).cuda()
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

        fts_all, attn_weight_list = generate_clip_fts(img, self.encoder, require_all_fts=True)

        with torch.no_grad():
            dino_img_h, dino_img_w = (h//14)*14, (w//14)*14
            dino_img = F.interpolate(img, size=(dino_img_h, dino_img_w), mode='bilinear', align_corners=False)
            # dino_fts = self.dino_encoder(dino_img)
            dino_ftses = self.dino_encoder.forward_features(dino_img)
            dino_fts = dino_ftses['x_norm_patchtokens']
            # print(dino_fts.shape)

        fts_all_stack = torch.stack(fts_all, dim=0) # (11, hw, b, c)
        attn_weight_stack = torch.stack(attn_weight_list, dim=0).permute(1, 0, 2, 3)
        if self.require_all_fts==True:
            cam_fts_all = fts_all_stack[-1].unsqueeze(0).permute(2, 1, 0, 3) #(1, hw, 1, c)
        else:
            cam_fts_all = fts_all_stack.permute(2, 1, 0, 3)

        all_img_tokens = fts_all_stack[:, 1:, ...]
        img_tokens_channel = all_img_tokens.size(-1)
        all_img_tokens = all_img_tokens.permute(0, 2, 3, 1)
        all_img_tokens = all_img_tokens.reshape(-1, b, img_tokens_channel, h//16, w //16) #(11, b, c, h, w)

        fts = self.decoder_fts_fuse(all_img_tokens)
        # attn_fts = fts.clone()
        _, _, fts_h, fts_w = fts.shape #24
        
        # dino_fts = dino_fts.reshape([b, h//14, w//14, -1]).permute(0,3,1,2)
        dino_fts = dino_fts.reshape([b, dino_img_h//14, dino_img_w//14, -1]).permute(0,3,1,2)
        _, _, dino_h, dino_w = dino_fts.shape #32
        
        dino_fts = self.dino_decoder_fts_fuse(dino_fts.unsqueeze(0))
        
        dino_fts = F.interpolate(dino_fts, size=(fts_h, fts_w), mode='bilinear', align_corners=False)
        
        clip_dino_fts = torch.cat([fts, dino_fts], dim=1)
        # print(clip_dino_fts.shape)
        # print('fts', fts_h, fts_w, 'dino', dino_h, dino_w)
        
        seg, seg_attn_weight_list = self.decoder(clip_dino_fts)
        
        # attn_fts = fts.clone()
        attn_fts = F.interpolate(clip_dino_fts, size=(fts_h, fts_w), mode='bilinear', align_corners=False)
        f_b, f_c, f_h, f_w = attn_fts.shape
        attn_fts_flatten = attn_fts.reshape(f_b, f_c, f_h*f_w)
        attn_pred = attn_fts_flatten.transpose(2, 1).bmm(attn_fts_flatten)
        attn_pred = torch.sigmoid(attn_pred)

        for i, img_name in enumerate(img_names):
            img_path = os.path.join(self.root_path, str(img_name)+'.jpg')
            img_i = img[i]
            cam_fts = cam_fts_all[i]
            cam_attn = attn_weight_stack[i]
            seg_attn = attn_pred.unsqueeze(0)[:, i, :, :]
            
            if self.iter_num > 15000 or mode=='val':
                require_seg_trans = True
            else:
                require_seg_trans = False
                
            # cam_refined_list, keys, w, h = perform_single_voc_fast(img_path, img_i, cam_fts, cam_attn, seg_attn,
            #                                                        self.bg_text_features, self.fg_text_features,
            #                                                        self.grad_cam,
            #                                                        mode=mode,
            #                                                        require_seg_trans=require_seg_trans)
            cam_refined_list, keys, w, h = perform_single_voc_fast_select(img_path, img_i, cam_fts, cam_attn, seg_attn,
                                                                   self.bg_text_features, self.fg_text_features,
                                                                   self.grad_cam,
                                                                   mode=mode,
                                                                   require_seg_trans=require_seg_trans)

            # fts_all, cam_refined_list, keys, w, h = perform_single_voc(img_path, img_i, cam_fts, cam_attn, seg_attn,
            #                                                        self.bg_text_features, self.fg_text_features,
            #                                                        self.grad_cam,
            #                                                        mode=mode,
            #                                                        require_seg_trans=require_seg_trans)

            # fts_all, cam_refined_list, keys, w, h = perform_single(img_path, img_i, cam_fts, cam_attn, seg_attn,self.encoder,
            #                                                        self.bg_text_features, self.fg_text_features,
            #                                                        self.grad_cam,
            #                                                        mode=mode,
            #                                                        require_all_fts=self.require_all_fts,
            #                                                        require_seg_trans=require_seg_trans)

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

        return seg, all_cam_labels, attn_pred

        
    

if __name__=="__main__":

    pretrained_weights = torch.load('pretrained/mit_b1.pth')
    wetr = WeTr('mit_b1', num_classes=20, embedding_dim=256, pretrained=True)
    wetr._param_groups()
    dummy_input = torch.rand(2,3,512,512)
    wetr(dummy_input)