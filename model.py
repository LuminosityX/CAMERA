import torch
import torch.nn as nn
from torch.nn import init
import torchvision.models as models
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
import math
import torch.optim as optim

from models import TextEncoder
from models import PositionEncoder
from models import AGSA
from models import Summarization
from models import MultiViewMatching
from loss import TripletLoss, DiversityRegularization

#########----PositionEncoder----##############
'''
def absoluteEncode(boxes, imgs_wh):
    # boxes -- (bs, num_regions, 4), imgs_wh -- (bs, 2) 
    x, y, w, h = boxes[:, :, 0], boxes[:, :, 1], boxes[:, :, 2] - boxes[:, :, 0], boxes[:, :, 3] - boxes[:, :, 1]
    expand_wh = torch.cat([imgs_wh, imgs_wh], dim=1).unsqueeze(dim=1)    #(bs, 1, 4)
    ratio_wh = (w / h).unsqueeze(dim=-1)  #(bs, num_r, 1)
    ratio_area = (w * h) / (imgs_wh[:, 0] * imgs_wh[:, 1]).unsqueeze(-1) #(bs, num_r)
    ratio_area = ratio_area.unsqueeze(-1) #(bs, num_r, 1)
    boxes = torch.stack([x, y, w, h], dim=2)
    boxes = boxes / expand_wh   #(bs, num_r, 4)
    res = torch.cat([boxes, ratio_wh, ratio_area], dim=-1)  #(bs, num_r, 6)
    return res

class PositionEncoder(nn.Module):
    # Relative position Encoder
    
    def __init__(self, embed_dim, posi_dim=6):             # 2048
        super(PositionEncoder, self).__init__()
        self.proj = nn.Linear(posi_dim, embed_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, boxes, imgs_wh):
        # boxes -- (bs, num_regions, 4), imgs_wh -- (bs, num_regions, 2)
        bs, num_regions = boxes.size()[:2]
        posi = absoluteEncode(boxes, imgs_wh)   #(bs, num_r, 6)

        x = self.proj(posi)                     #(bs, num_r, 2048)
        x = self.sigmoid(x)
        return x
'''
##############################################

#########----Summarization----##############
'''
class Summarization(nn.Module):
    # Multi-View Summarization Module 
    def __init__(self, embed_size, smry_k):   # 2048 12
        super(Summarization, self).__init__()       
        # dilation conv
        out_c = [256, 128, 128, 128, 128, 128, 128]   # all plus is 1024 
        k_size = [1, 3, 3, 3, 5, 5, 5]
        dila = [1, 1, 2, 3, 1, 2, 3]
        pads = [0, 1, 2, 3, 2, 4, 6]     # 填充0的层数，刚好使得得到的所有kenel的结果的长度都依旧是36
        convs_dilate = [nn.Conv1d(embed_size, out_c[i], k_size[i], dilation=dila[i], padding=pads[i]) \    # 虽然是一维卷积，但其实还是在2维tensor上进行的，只是这里将tensor当成了词向量，第一个参数就是词向量维度。所以其实真正的kenel size是该维度乘以相关大小
                        for i in range(len(out_c))]
        self.convs_dilate = nn.ModuleList(convs_dilate)
        self.convs_fc = nn.Linear(1024, smry_k)

    def forward(self, rgn_emb):
        x = rgn_emb.transpose(1, 2)    #(bs, dim, num_r)                  # 要求输入的维度是这样
        x = [F.relu(conv(x)) for conv in self.convs_dilate]
        x = torch.cat(x, dim=1) #(bs, 1024, num_r)
        x = x.transpose(1, 2)   #(bs, num_r, 1024)
        smry_mat = self.convs_fc(x)    #(bs, num_r, k)                    # 得到了K个view的权重
        return smry_mat
'''
##############################################

def l2norm(X, dim=1):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

class EncoderImagePrecompSelfAttn(nn.Module):

    def __init__(self, img_dim, embed_size, head, smry_k, drop=0.0):
        super(EncoderImagePrecompSelfAttn, self).__init__()
        self.embed_size = embed_size

        self.fc = nn.Linear(img_dim, embed_size)
        self.init_weights()
        self.position_enc = PositionEncoder(embed_size) 
        self.agsa = AGSA(1, embed_size, h=head, is_share=False, drop=drop)
        self.mvs = Summarization(embed_size, smry_k)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images, boxes, imgs_wh):
        """Extract image feature vectors."""
        fc_img_emd = self.fc(images)
        fc_img_emd = l2norm(fc_img_emd)  #(bs, num_regions, dim)
        posi_emb = self.position_enc(boxes, imgs_wh)    #(bs, num_regions, num_regions, dim) 这里应该是[b, 36, 2048]

        # Adaptive Gating Self-Attention
        self_att_emb = self.agsa(fc_img_emd, posi_emb)    #(bs, num_regions, dim)
        self_att_emb = l2norm(self_att_emb)
        # Multi-View Summarization
        smry_mat = self.mvs(self_att_emb)                 # 得到的view权重
        L = F.softmax(smry_mat, dim=1)
        img_emb_mat = torch.matmul(L.transpose(1, 2), self_att_emb) #(bs, k, dim)

        return F.normalize(img_emb_mat, dim=-1), smry_mat

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecompSelfAttn, self).load_state_dict(new_state)
        
#########----MultiViewMatching----##############
'''
class MultiViewMatching(nn.Module):
    def __init__(self, ):
        super(MultiViewMatching, self).__init__()

    def forward(self, imgs, caps):
        # caps -- (num_caps, dim), imgs -- (num_imgs, r, dim)
        num_caps  = caps.size(0)
        num_imgs, r = imgs.size()[:2]
        
        if num_caps == num_imgs:                             # train 与 test 会有区别
            scores = torch.matmul(imgs, caps.t()) #(num_imgs, r, num_caps)
            scores = scores.max(1)[0]  #(num_imgs, num_caps)
        else:                                                # test 会出现的问题
            scores = []
            score_ids = []
            for i in range(num_caps):
                cur_cap = caps[i].unsqueeze(0).unsqueeze(0)  #(1, 1, dim)
                cur_cap = cur_cap.expand(num_imgs, -1, -1)   #(num_imgs, 1, dim)
                cur_score = torch.matmul(cur_cap, imgs.transpose(-2, -1)).squeeze()    #(num_imgs, r)
                cur_score = cur_score.max(1, keepdim=True)[0]   #(num_imgs, 1)
                scores.append(cur_score)
            scores = torch.cat(scores, dim=1)   #(num_imgs, num_caps)

        return scores  
'''
##############################################

   
class CAMERA(object):
    def __init__(self, opt):
        # Build Models
        self.opt = opt
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImagePrecompSelfAttn(opt.img_dim, opt.embed_size, \       # 2048 2048
                                    opt.head, opt.smry_k, drop=opt.drop)                # 64 12
        self.txt_enc = TextEncoder(opt.bert_config_file, opt.init_checkpoint, \
                                    opt.embed_size, opt.head, drop=opt.drop)            # 2048 64

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        self.mvm = MultiViewMatching()
        # Loss and Optimizer
        self.crit_ranking = TripletLoss(margin=opt.margin, max_violation=opt.max_violation)
        self.crit_div = DiversityRegularization(opt.smry_k, opt.batch_size)

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        params = filter(lambda p: p.requires_grad, params)

        self.params = params
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])


    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, batch_data, volatile=False):
        """Compute the image and caption embeddings
        """
        images, boxes, imgs_wh, input_ids, lengths, ids, attention_mask, token_type_ids = batch_data
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            boxes = boxes.cuda()
            imgs_wh = imgs_wh.cuda()
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            token_type_ids = token_type_ids.cuda()

        # Forward
        cap_emb = self.txt_enc(input_ids, attention_mask, token_type_ids, lengths) # [bs, dim]
        img_emb, smry_mat = self.img_enc(images, boxes, imgs_wh)                   # [bs, k/12, dim]

        return img_emb, cap_emb, smry_mat                                          # smry_mat 未softmax前的权重


    def train_emb(self, epoch, batch_data, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        self_att_emb, cap_emb, smry_mat = self.forward_emb(batch_data)
        bs = self_att_emb.size(0)
        # bidirectional triplet ranking loss
        sim_mat = self.mvm(self_att_emb, cap_emb)
        ranking_loss = self.crit_ranking(sim_mat)
        self.logger.update('Rank', ranking_loss.item(), bs)
        # diversity regularization
        div_reg = self.crit_div(smry_mat)
        self.logger.update('Div', div_reg.item(), bs)
        # total loss
        loss = ranking_loss + div_reg * self.opt.smry_lamda
        self.logger.update('Le', loss.item(), bs) 

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            if isinstance(self.params[0], dict):
                params = []
                for p in self.params:
                    params.extend(p['params'])
                clip_grad_norm(params, self.grad_clip)
            else:
                clip_grad_norm(self.params, self.grad_clip)

        self.optimizer.step()

