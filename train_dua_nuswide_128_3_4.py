import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from metric_duapre import compress_wiki, compress, calculate_map, calculate_top_map, p_topK
# import datasetspre as datasets
import settingsnuspre as settings
from models3pre3 import ImgNet, TxtNet, DeTxtNet, DeImgNet, GenHash, GetTxtNet, GetImgNet, FuseTransEncoder, GetITNet
from load_data import get_loader_flickr, get_loader_nus, get_loader_coco
import os.path as osp
import sys

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda:0', temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())
    def forward(self, emb_i, emb_j):
        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)
        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        sim_ij = torch.diag(similarity_matrix, self.batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss
class Session:
    def __init__(self, train_loader, test_loader, database_loader, train_dataset, test_dataset, database_dataset, data_train, a, a2, a3):

        self.logger = settings.logger

        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.cuda.set_device(settings.GPU_ID)
        # train_dataset = dataloader['train']
        # test_dataset = dataloader['query']
        # database_dataset = dataloader['database']

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.database_dataset = database_dataset
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.database_loader = database_loader
        self.I_tr, self.T_tr, self.L_tr = data_train

        # txt_feat_len = datasets.txt_feat_len
        txt_feat_len = self.T_tr.shape[1]
        img_feat_len = self.I_tr.shape[1]
        self.ContrastiveLoss = ContrastiveLoss(batch_size=settings.BATCH_SIZE, device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu"))

        self.CodeNet_I = ImgNet(code_len=settings.CODE_LEN, img_feat_len=img_feat_len)
        self.FeatNet_I = ImgNet(code_len=settings.CODE_LEN, img_feat_len=img_feat_len)
        self.CodeNet_T = TxtNet(code_len=settings.CODE_LEN, txt_feat_len=txt_feat_len)


        self.GetNet_I = GetImgNet(code_len=settings.CODE_LEN, img_feat_len=img_feat_len)
        self.GetNet_T = GetTxtNet(code_len=settings.CODE_LEN, txt_feat_len=txt_feat_len)
        self.GetNet = GetITNet(code_len=settings.CODE_LEN, img_feat_len=5000)

        self.DeCodeNet_I = DeImgNet(code_len=settings.CODE_LEN, img_feat_len=img_feat_len)
        self.DeCodeNet_T = DeTxtNet(code_len=settings.CODE_LEN, txt_feat_len=txt_feat_len)
        self.genHash = GenHash(code_len=settings.CODE_LEN, txt_feat_len=settings.BATCH_SIZE)
        # self.Txt2Img = Txt2Img(txt_code_len=settings.CODE_LEN, img_code_len=settings.CODE_LEN)
        # self.Img2Txt = Img2Txt(img_code_len=settings.CODE_LEN, txt_code_len=settings.CODE_LEN)
        num_layers, nhead = 2,  4

        self.FuseTrans = FuseTransEncoder(num_layers, txt_feat_len + img_feat_len, nhead).cuda()

        # self.optimizer_TextMlp = optim.Adam(paramsText, lr=1e-3, betas=(0.5, 0.999))

        if settings.DATASET == "WIKI":
            self.opt_I = torch.optim.SGD([{'params': self.CodeNet_I.fc_encode.parameters(), 'lr': settings.LR_IMG},
                                          {'params': self.CodeNet_I.alexnet.classifier.parameters(),
                                           'lr': settings.LR_IMG}],
                                         momentum=settings.MOMENTUM, weight_decay=settings.WEIGHT_DECAY)
            self.opt_DeI = torch.optim.SGD([{'params': self.DeCodeNet_I.fc_encode.parameters(), 'lr': settings.LR_IMG}],
                                           momentum=settings.MOMENTUM, weight_decay=settings.WEIGHT_DECAY)

        # if settings.DATASET == "MIRFlickr" or settings.DATASET == "NUSWIDE":
        #     self.opt_I = torch.optim.SGD(self.CodeNet_I.parameters(), lr=settings.LR_IMG, momentum=settings.MOMENTUM,
        #                                  weight_decay=settings.WEIGHT_DECAY)
        #     self.opt_GI = torch.optim.SGD(self.GetNet_I.parameters(), lr=settings.LR_IMG, momentum=settings.MOMENTUM,
        #                                  weight_decay=settings.WEIGHT_DECAY)
        #     self.opt_DeI = torch.optim.SGD(self.DeCodeNet_I.parameters(), lr=settings.LR_IMGTXT,
        #                                    momentum=settings.MOMENTUM, weight_decay=settings.WEIGHT_DECAY)
        #
        # self.optimizer_FuseTrans = torch.optim.Adam(self.FuseTrans.parameters(), lr=1e-4, betas=(0.5, 0.999))
        #
        # self.opt_T = torch.optim.SGD(self.CodeNet_T.parameters(), lr=settings.LR_TXT, momentum=settings.MOMENTUM,
        #                              weight_decay=settings.WEIGHT_DECAY)
        # self.opt_GT = torch.optim.SGD(self.GetNet_T.parameters(), lr=settings.LR_TXT, momentum=settings.MOMENTUM,
        #                              weight_decay=settings.WEIGHT_DECAY)
        # self.gen_H = torch.optim.SGD(self.genHash.parameters(), lr=settings.LR_TXT, momentum=settings.MOMENTUM,
        #                              weight_decay=settings.WEIGHT_DECAY)
        # self.opt_DeT = torch.optim.SGD(self.DeCodeNet_T.parameters(), lr=settings.LR_IMGTXT, momentum=settings.MOMENTUM,
        #                                weight_decay=settings.WEIGHT_DECAY)
        if settings.DATASET == "MIRFlickr" or settings.DATASET == "NUSWIDE":
            self.opt_I = torch.optim.Adam(self.CodeNet_I.parameters(), lr=settings.LR_IMG, betas=(0.5, 0.999))
            self.opt_GI = torch.optim.Adam(self.GetNet_I.parameters(), lr=settings.LR_IMG, betas=(0.5, 0.999))
            self.opt_DeI = torch.optim.Adam(self.DeCodeNet_I.parameters(), lr=settings.LR_IMGTXT, betas=(0.5, 0.999))

        self.optimizer_FuseTrans = torch.optim.Adam(self.FuseTrans.parameters(), lr=1e-4, betas=(0.5, 0.999))

        self.opt_T = torch.optim.Adam(self.CodeNet_T.parameters(), lr=settings.LR_TXT, betas=(0.5, 0.999))
        self.opt_GT = torch.optim.Adam(self.GetNet_T.parameters(), lr=settings.LR_TXT, betas=(0.5, 0.999))
        self.gen_H = torch.optim.Adam(self.genHash.parameters(), lr=settings.LR_TXT, betas=(0.5, 0.999))
        self.opt_DeT = torch.optim.Adam(self.DeCodeNet_T.parameters(), lr=settings.LR_IMGTXT, betas=(0.5, 0.999))
        self.opt_G = torch.optim.Adam(self.GetNet.parameters(), lr=settings.LR_TXT, betas=(0.5, 0.999))

        # self.opt_T2I = torch.optim.SGD(self.Txt2Img.parameters(), lr=settings.LR_IMGTXT, momentum=settings.MOMENTUM,
        #                                weight_decay=settings.WEIGHT_DECAY)
        # self.opt_I2T = torch.optim.SGD(self.Img2Txt.parameters(), lr=settings.LR_IMGTXT, momentum=settings.MOMENTUM,
        #                                weight_decay=settings.WEIGHT_DECAY)
        img_norm = F.normalize(torch.Tensor(self.I_tr)).cuda()
        txt_norm = F.normalize(torch.Tensor(self.T_tr)).cuda()
        # S_I = img_norm.mm(img_norm.t()).cuda()
        # # S_I = self.cal_similarity(F_I, F_I)
        # # S_I = S_I * 2 - 1
        #
        # S_T = txt_norm.mm(txt_norm.t()).cuda()
        # # S_T = self.cal_similarity(F_T, F_T)
        # # S_T = S_T * 2 - 1
        # a4 = 0.2
        # a5 = 0.8
        # a6 = 0.8
        # S_tilde = a6 * S_I.cuda() + (1 - a6) * S_T.cuda()  # + 0.1 * S_A
        # # S_tilde2 = settings.BETA * S_I - (1 - settings.BETA) * S_T
        # # self.gs = (1 - 0.3) * S_tilde.cuda() + 0.3 * (S_I.mm(S_T.t()).cuda()) / 5000
        # # self.S = (1 - a5) * S_tilde.cuda() + a5 * (
        # #             a4 * S_T.mm(S_T.t()).cuda() + (1 - a4) * S_T.mm(S_T.t()).cuda()) / 5000
        # self.S = (1 - a5) * S_tilde.cuda() + a5 * (a4 * S_T.mm(S_T.t()).cuda() + (1 - a4) * S_I.mm(S_T.t()).cuda()) / 5000
        # # self.S = self.cal_similarity(img_norm, txt_norm, a1, a2, a3)
        # # self.gs = S_tilde.mm(img_norm)
        # # self.gs = torch.cat((self.S.mm(txt_norm).cuda(), self.S.mm(img_norm).cuda()), 1)
        #
        # self.gsT = self.S.mm(txt_norm).cuda()
        # self.gsI = self.S.mm(img_norm).cuda()

        # self.gsT = S_T.mm(txt_norm).cuda()
        # self.gsI = S_I.mm(img_norm).cuda()
        # self.gs = S_I.mm(img_norm).cuda()

        # self.gs = a1 * S_I.cuda() + (1 - a1) * S_T.cuda()
        # self.gs = torch.cat((S_I, S_T), 1)
        # self.gs = self.cal_similarity(img_norm, txt_norm, a1, a2, a3)
        # self.gs = 0.7 * img_norm.mm(img_norm.t()) + 0.3 * txt_norm.mm(txt_norm.t())


        # self.img, self.F_T, self.labels, _ = self.train_loader
        # self.gs = self.cal_similarity(self.img, self.F_T)
        self.gs = self.cal_similarity(img_norm, txt_norm, a, a, a3)
        self.S = self.gs

    def train(self, epoch, l1, l2, l3, l4, l5, l6, l7):
        self.CodeNet_I.cuda().train()
        self.CodeNet_T.cuda().train()
        self.DeCodeNet_I.cuda().train()
        self.DeCodeNet_T.cuda().train()
        self.genHash.cuda().train()
        # self.FuseTrans.cuda().train()
        # self.Txt2Img.cuda().train()
        # self.Img2Txt.cuda().train()
        # self.GetNet_I.cuda().train()
        # self.GetNet_T.cuda().train()
        self.GetNet.cuda().train()

        self.CodeNet_I.set_alpha(1)
        self.CodeNet_T.set_alpha(1)
        self.DeCodeNet_I.set_alpha(1)
        self.DeCodeNet_T.set_alpha(1)
        self.genHash.set_alpha(1)
        # self.Txt2Img.set_alpha(1)
        # self.Img2Txt.set_alpha(1)
        # self.GetNet_I.set_alpha(1)
        # self.GetNet_T.set_alpha(1)
        self.GetNet.set_alpha(1)

        # self.logger.info('Epoch [%d/%d], alpha for ImgNet: %.3f, alpha for TxtNet: %.3f' % (epoch + 1, settings.NUM_EPOCH, self.CodeNet_I.alpha, self.CodeNet_T.alpha))

        for No, (F_I, F_T, _, index_) in enumerate(self.train_loader): #No, (img, txt, _, index_)
            # img = Variable(img.cuda())
            F_T = Variable(torch.FloatTensor(F_T.numpy()).cuda())
            F_I = Variable(torch.FloatTensor(F_I.numpy()).cuda())
            # labels = Variable(labels.cuda())

            self.opt_I.zero_grad()
            self.opt_T.zero_grad()
            self.opt_DeI.zero_grad()
            self.opt_DeT.zero_grad()
            self.gen_H.zero_grad()
            # self.opt_I2T.zero_grad()
            # self.opt_T2I.zero_grad()
            # self.opt_GI.zero_grad()
            # self.opt_GT.zero_grad()
            self.opt_G.zero_grad()
            # self.optimizer_FuseTrans.zero_grad()

            # F_I, _, _ = self.FeatNet_I(img)
            _, _, code_I = self.CodeNet_I(F_I.cuda())
            _, _, code_T = self.CodeNet_T(F_T.cuda())
            # _, hid_T, code_I_ = self.Txt2Img(code_T.cuda())
            # _, hid_T, code_T_ = self.Img2Txt(code_I.cuda())
            _, hid_I, FI_ = self.DeCodeNet_I(code_T.cuda())
            _, hid_T, FT_ = self.DeCodeNet_T(code_I.cuda())
            F_I1 = F_I
            F_T1 = F_T

            B_I = code_I
            B_T = code_T
            F_I = F.normalize(F_I.cuda())
            S_I = F_I.mm(F_I.t()).cuda()
            # S_I = self.cal_similarity(F_I, F_I)
            S_I = S_I * 2 - 1

            F_T = F.normalize(F_T.cuda())
            # S_T = F_T.mm(F_T.t()).cuda()
            # # S_T = self.cal_similarity(F_T, F_T)
            # S_T = S_T.cuda() * 2 - 1
            # a4 = 0.2
            # a6 = 0.8
            # S_tilde = a6 * S_I.cuda() + (1 - a6) * S_T.cuda()  # + 0.1 * S_A
            # # S_tilde2 = settings.BETA * S_I - (1 - settings.BETA) * S_T
            # a5 = 0.8
            # # A = S_I.mm(S_T.t()).cuda()
            # # B = S_tilde.mm(S_tilde.t()).cuda()
            # # S = (1 - a5) * S_tilde.cuda() + a5 * (a4 * S_T.mm(S_T.t()).cuda() + (1 - a4) * S_T.mm(S_T.t()).cuda()) / settings.BATCH_SIZE
            # S = (1 - a5) * S_tilde.cuda() + a5 * (a4 * S_T.mm(S_T.t()).cuda() + (1 - a4) * S_I.mm(S_T.t()).cuda()) / settings.BATCH_SIZE
            # S = (1 - a5) * S_tilde.cuda() + a5 * (S_I.mm(S_T.t()).cuda()) / settings.BATCH_SIZE
            # S = (1 - a5) * S_tilde.cuda() + a5 * (S_tilde.mm(S_tilde.t()).cuda()) / settings.BATCH_SIZE
            # S = (1 - settings.ETA) * S_tilde + settings.ETA * 2 * (
            #         settings.BETA * settings.BETA * S_I.mm(S_I) + (1 - settings.BETA) * (
            #         1 - settings.BETA) * S_T.mm(S_T)) / settings.BATCH_SIZE
            # S = S_tilde
            S = self.gs[index_, :][:, index_].cuda()
            # H, B = self.genHash(torch.cat((S.mm(B_I), S.mm(B_T)), 1))
            # H, B = self.genHash(self.gs[index_, :].cuda())
            # gs = torch.cat((self.gsI[index_, :].cuda(), self.gsT[index_, :].cuda()), 1)
            # gsI, gsT = self.FuseTrans(gs)

            # _, _, Get_I = self.GetNet_I(self.gsI[index_, :].cuda())
            # _, _, Get_T = self.GetNet_T(self.gsT[index_, :].cuda())
            # _, _, Get_I = self.GetNet_I(F_I1.cuda(), S)
            # _, _, Get_T = self.GetNet_T(F_T1.cuda(), S)
            _, _, Get = self.GetNet(self.S[index_, :].cuda())
            # _, _, Get_T = self.GetNet_T(self.S[index_, :].cuda(), S)

            # _, _, Get_I = self.GetNet_I(gsI.cuda())
            # _, _, Get_T = self.GetNet_T(gsT.cuda())

            # B = torch.sign(H)
            # G = torch.cat((Get_I, Get_T), 1).cuda()
            GF = torch.cat((Get, code_T), 1).cuda()
            H, B = self.genHash(torch.cat((code_I, GF), 1).cuda())

            # _, hid_I, FI_ = self.DeCodeNet_I(S.mm(code_T.cuda()))
            # _, hid_T, FT_ = self.DeCodeNet_T(S.mm(code_I.cuda()))
            # _, hid_I, FI_ = self.DeCodeNet_I(B_T.cuda())
            # _, hid_T, FT_ = self.DeCodeNet_T(B_I.cuda())
            B_I = F.normalize(code_I.cuda())
            B_T = F.normalize(code_T.cuda())

            # GB_I = F.normalize(Get_I.cuda())
            # GB_T = F.normalize(Get_T.cuda())
            GB = F.normalize(Get.cuda())

            F_I = F.normalize(F_I.cuda())
            F_T = F.normalize(F_T.cuda())

            FI_ = F.normalize(FI_.cuda())
            FT_ = F.normalize(FT_.cuda())


            # B_I = F.normalize(code_I)
            # B_T = F.normalize(code_T)
            # B_I_ = F.normalize(code_I_)
            # B_T_ = F.normalize(code_T_)

            BI_BI = B_I.mm(B_I.t()).cuda()
            BT_BT = B_T.mm(B_T.t()).cuda()
            BI_BT = B_I.mm(B_T.t()).cuda()
            # BI_H = B_I.mm(H.t()).cuda()
            # BT_H = B_T.mm(H.t()).cuda()
            H_H = H.mm(H.t()).cuda()
            GI_GT = GB.mm(GB.t()).cuda()

            # BI_BI_ = B_I_.mm(B_I_.t())
            # BT_BT_ = B_T_.mm(B_T_.t())


            loss6 = 0 # self.crossview_contrastive_Loss(code_I, code_T)
            a1 = 1


            loss1 = F.mse_loss(BT_BT.cuda(), a1 * S.cuda()) + F.mse_loss(BI_BI.cuda(), a1 * S.cuda())
            loss2 = F.mse_loss(BI_BT.cuda(), a1 * S.cuda()) #+ F.mse_loss(B_T.mm(B_I.t()), 0.8 * S) #+ 0.2 * F.mse_loss(BI_BT, B_T.mm(B_T.t()))  + 0.2 * F.mse_loss(B_T.mm(B_I.t()), B_I.mm(B_I.t()))   # + F.mse_loss(B_T, B_I)
            # FI = F.normalize(FI)
            # FT = F.normalize(FT)
            # loss3 = F.mse_loss(FT.mm(FT.t()), S) + F.mse_loss(FI.mm(FI.t()), S)
            # loss4 = F.mse_loss(FI.mm(FT.t()), S)  # + F.mse_loss(B_T, B_I)
            loss3 = F.mse_loss(B_I.cuda(), B.cuda()) + F.mse_loss(B_T.cuda(), B.cuda()) #+ F.mse_loss(B_T.cuda(), B_I.cuda())


            # loss4 = F.mse_loss(H_H.cuda(), S.cuda()) #+F.mse_loss(H_H.cuda(), S.cuda()) #+ F.mse_loss(BT_H.cuda(), S.cuda()) + F.mse_loss(BI_H.cuda(), S.cuda()) #+ F.mse_loss(B.mm(B.t()).cuda(), S.cuda())
            loss4 = F.mse_loss(GI_GT.cuda(), S.cuda())
            loss5 = F.mse_loss(B_T.cuda(), B_I.cuda())
            # loss4 = F.mse_loss(code_T, B) + F.mse_loss(code_I, B) + F.mse_loss(H, B)#
            # l41 = F.mse_loss(code_T, B)
            # l42 = F.mse_loss(code_I, B)
            # l43 = F.mse_loss(H, B)

            # code = 0.8 * code_I + 0.2 * code_T
            loss71 = F.mse_loss(FI_.cuda(), F_I.cuda()) + F.mse_loss(FT_.cuda(), F_T.cuda())
            # loss72 = F.mse_loss(FI_.mm(FI_.t()).cuda(), F_I.mm(F_I.t()).cuda()) + F.mse_loss(FT_.mm(FT_.t()).cuda(), F_T.mm(F_T.t()).cuda())
            # loss72 = F.mse_loss(FI_.mm(FI_.t()), F_I.mm(F_I.t())) + F.mse_loss(FT_.mm(FT_.t()), F_T.mm(F_T.t()))
            # loss7 =
            # loss4 = F.mse_loss(FI_.mm(FI_.t()),S) + F.mse_loss(F_T_.mm(F_T_.t()), S)
            # loss5 = 0
            # if epoch > 20:
            #     loss5 = F.mse_loss(BI_BI_, S) + F.mse_loss(BT_BT_, S)
            # loss5 = F.mse_loss(BI_BI_, S) + F.mse_loss(BT_BT_, S)

            # loss5 = F.mse_loss(B_T.mm(B_I_.t()), S) + F.mse_loss(B_I.mm(B_T_.t()), S) + F.mse_loss(B_I.mm(B_I_.t()), S) + F.mse_loss(B_T.mm(B_T_.t()), S)

            # loss11 = F.mse_loss(BI1_BI1, S)
            # loss21 = F.mse_loss(BI1_BT1, S)
            # loss31 = F.mse_loss(BT1_BT1, S)
            # l7 = 1
            # l1 = 0.1
            loss = l1 * loss1 + l2 * loss2 + l6 * loss71 + l3 * loss3 + l4 * loss4 + l7*loss5# + l6 * loss72+ l4 * loss4# + l3 * loss3+ l1 * loss3 + l2 * loss4 # + l4 * loss72 + l7 * l3 * loss3 + l3 * loss5 + 0.1 * loss6.item()  # + l7 * loss7 #+ l7 * loss6.item()# + (- loss21) + loss4

            loss.backward()
            self.opt_I.step()
            self.opt_T.step()
            self.opt_DeI.step()
            self.opt_DeT.step()
            self.gen_H.step()
            # self.opt_I2T.step()
            # self.opt_T2I.step()
            # self.opt_GI.step()
            # self.opt_GT.step()
            self.opt_G.step()
            # self.optimizer_FuseTrans.step()

            # if (No + 1) % (self.T_tr.shape[0] // settings.BATCH_SIZE) == 0:
            #     print(loss.item()/134)
            #     self.logger.info('Epoch [%d/%d], Iter [%d/%d] Loss1: %.4f Loss2: %.4f Loss3: %.4f Loss4: %.4f Loss71: %.4f Loss72: %.4f Loss6: %.4f Total Loss: %.4f'
            #         % (epoch + 1, settings.NUM_EPOCH, No + 1, self.T_tr.shape[0] // settings.BATCH_SIZE,
            #             loss1, loss2, loss3, loss4, loss71, loss72, loss6, loss))
            # if (idx + 1) % (len(self.train_dataset) // settings.BATCH_SIZE / settings.EPOCH_INTERVAL) == 0:
            #     self.logger.info('Epoch [%d/%d], Iter [%d/%d] Loss1: %.4f Loss2: %.4f Loss3: %.4f Total Loss: %.4f'
            #         % (epoch + 1, settings.NUM_EPOCH, idx + 1, len(self.train_dataset) // settings.BATCH_SIZE,
            #             loss1, loss2, loss3, loss))

    def eval(self, l1, l2, l3, l4, l5, l6, l7):
        # self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')

        # Change model to 'eval' mode (BN uses moving mean/var).
        self.CodeNet_I.eval().cuda()
        self.CodeNet_T.eval().cuda()

        if settings.DATASET == "WIKI":
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress_wiki(self.database_loader, self.test_loader,
                                                                   self.CodeNet_I, self.CodeNet_T, self.database_dataset,
                                                                   self.test_dataset)

        if settings.DATASET == "MIRFlickr" or settings.DATASET == "NUSWIDE":
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.CodeNet_I,
                                                              self.CodeNet_T,
                                                              self.database_dataset, self.test_dataset)

        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)
        # MAP_I2TA = calculate_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L)
        # MAP_T2IA = calculate_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L)
        # K = [5000]
        # MAP_I2T5 = p_topK(qu_BI, re_BT, qu_L, re_L, K)
        # MAP_T2I5 = p_topK(qu_BT, re_BI, qu_L, re_L, K)

        # self.logger.info('MAP of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2T, MAP_T2I))
        self.logger.info('MAP: %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f' % (l1, l2, l3, l4, l5, l6, l7, MAP_I2T, MAP_T2I, 0, 0, MAP_I2T+MAP_T2I))
        # self.logger.info('--------------------------------------------------------------------')
        # K = [1, 200, 400, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
        # MAP_I2T0 = p_topK(qu_BI, re_BT, qu_L, re_L, K)
        # MAP_T2I0 = p_topK(qu_BT, re_BI, qu_L, re_L, K)
        # self.logger.info(MAP_I2T0)
        # self.logger.info(MAP_T2I0)
        return MAP_I2T+MAP_T2I

    def save_checkpoints(self, step, file_name='latest.pth'):
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        obj = {
            'ImgNet': self.CodeNet_I.state_dict(),
            'TxtNet': self.CodeNet_T.state_dict(),
            'step': step,
        }
        torch.save(obj, ckp_path)
        self.logger.info('**********Save the trained model successfully.**********')

    def load_checkpoints(self, file_name='latest.pth'):
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.logger.error('********** No checkpoint %s!*********' % ckp_path)
            return
        self.CodeNet_I.load_state_dict(obj['ImgNet'])
        self.CodeNet_T.load_state_dict(obj['TxtNet'])
        self.logger.info('********** The loaded model has been trained for %d epochs.*********' % obj['step'])

    def cal_similarity2(self, F_I, F_T):
        a1 = settings.BETA
        a2 = 0.6
        K = 3000
        batch_size = F_I.size(0)
        size = batch_size
        top_size = K

        F_I = F.normalize(F_I)
        S_I = F_I.mm(F_I.t())
        F_T = F.normalize(F_T)
        S_T = F_T.mm(F_T.t())

        S1 = a1 * S_I + (1 - a1) * S_T

        m, n1 = S1.sort()
        S1[torch.arange(size).view(-1, 1).repeat(1, top_size).view(-1), n1[:, :top_size].contiguous().view(-1)] = 0.

        S2 = 2.0 / (1 + torch.exp(-S1)) - 1 + torch.eye(S1.size(0)).cuda()
        S2 = (S2 + S2.t())/2
        S = a2 * S1 + (1 - a2) * S2

        return S

    def cal_similarity(self, F_I, F_T, l1, l4, l3):
        a1 = 0.4
        a2 = 0.4
        a3 = 0.3
        a4 = 2.0
        l2 = 0.8
        knn_number = 440  # l3

        # print(l3)
        scale = 1
        batch_size = F_I.size(0)

        F_I = F.normalize(F_I).cuda()
        S_I = F_I.mm(F_I.t()).cuda()
        F_T = F.normalize(F_T).cuda()
        S_T = F_T.mm(F_T.t()).cuda()
        a4 = 0.2
        a6 = 0.8
        S_pair = a6 * S_I.cuda() + (1 - a6) * S_T.cuda()  # + 0.1 * S_A
        # S_tilde2 = settings.BETA * S_I - (1 - settings.BETA) * S_T
        a5 = 0.8
        # A = S_I.mm(S_T.t()).cuda()
        # B = S_tilde.mm(S_tilde.t()).cuda()
        # S = (1 - a5) * S_tilde.cuda() + a5 * (a4 * S_T.mm(S_T.t()).cuda() + (1 - a4) * S_T.mm(S_T.t()).cuda()) / settings.BATCH_SIZE
        Sp = (1 - a5) * S_pair.cuda() + a5 * (
                a4 * S_T.mm(S_T.t()).cuda() + (1 - a4) * S_I.mm(S_T.t()).cuda()) / batch_size
        # S_I1 = a4 * S_I
        # for i in range(S_I.shape[0]):
        #     S_I1[:, i] = S_I[i, i] - S_I1[:, i]
        # for i in range(S_I.shape[0]):
        #     S_I1[i, :] = S_I[i, i] + S_I1[i, :]
        # S_I1 = (torch.exp(-1 * S_I1))
        # S_I = l2 * S_I + (1 - l2) * S_I1
        # # S_I = S_I * 2 - 1
        # S_T1 = a4 * S_T
        # for i in range(S_T.shape[0]):
        #     S_T1[:, i] = S_T[i, i] - S_T1[:, i]
        # for i in range(S_T.shape[0]):
        #     S_T1[i, :] = S_T[i, i] + S_T1[i, :]
        #
        # S_T1 = (torch.exp(-1 * S_T1))
        # S_T = l2 * S_T + (1 - l2) * S_T1
        # # S_T = S_T * 2 - 1
        #
        # S_pair = a1 * S_T + (1 - a1) * S_I

        # pro = S_T * a1 + S_I * (1. - a1)
        # S = (1 - a2) * (
        #     S_pair) + a2 * S_pair.mm(S_pair) / batch_size
        # m1, n1 = S_pair.sort()
        # S_pair = S_pair / m1[:, 0] * 0.001
        # S_pair = torch.tanh(S_pair * 7) #+ 0.1
        # m, n = S_pair.sort()
        # S_pair2 = torch.log(S_pair) + 1.5
        # b1 = torch.mul(S_pair, S_pair)
        # b2 = torch.mul(b1, S_pair)
        # S = S_pair + b1 + b2

        # m1, n1 = S_pair.sort()
        # m2, n1 = b1.sort()
        # m3, n1 = torch.mul(b1, S_pair2).sort()
        # m4, n1 = S_I.sort()
        # m5, n1 = S_T.sort()
        # S = S * settings.MU
        pro = Sp  # * settings.MU
        size = batch_size
        top_size = knn_number
        m, n1 = S_pair.sort()
        pro[torch.arange(size).view(-1, 1).repeat(1, top_size).view(-1), n1[:, :top_size].contiguous().view(
            -1)] = 0.
        pro[torch.arange(size).view(-1), n1[:, -1:].contiguous().view(
            -1)] = 0.
        # pro = pro / pro.sum(1).view(-1, 1)
        # pro_dis = pro.mm(pro.t())
        pro_dis = pro * scale
        S = pro_dis
        S = S * 2.0 - 1
        # for i in range(batch_size):
        #     S[i] = S[i] / S[i][i]
        return S

    def crossview_contrastive_Loss(self, view1, view2, lamb=9.0, EPS=sys.float_info.epsilon):
        """Contrastive loss for maximizng the consistency"""
        _, k = view1.size()
        # bn, k = view1.size()
        assert (view2.size(0) == _ and view2.size(1) == k)

        p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)
        p_i_j = p_i_j.sum(dim=0)
        p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
        p_i_j = p_i_j / p_i_j.sum()  # normalise
        # p_i_j = compute_joint(view1, view2)
        assert (p_i_j.size() == (k, k))

        p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
        p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

        p_i_j[(p_i_j < EPS).data] = EPS
        p_j[(p_j < EPS).data] = EPS
        p_i[(p_i < EPS).data] = EPS

        loss = - p_i_j * (torch.log(p_i_j) \
                          - (lamb + 1) * torch.log(p_j) \
                          - (lamb + 1) * torch.log(p_i))

        loss = loss.sum()

        return loss


def main():
    # if settings.DATASET == "WIKI":
    #     train_dataset = datasets.WIKI(root=settings.DATA_DIR, train=True, transform=datasets.wiki_train_transform)
    #     test_dataset = datasets.WIKI(root=settings.DATA_DIR, train=False, transform=datasets.wiki_test_transform)
    #     database_dataset = datasets.WIKI(root=settings.DATA_DIR, train=True,
    #                                      transform=datasets.wiki_test_transform)

    if settings.DATASET == "MIRFlickr":
        dataloader, data_train = get_loader_flickr(settings.BATCH_SIZE)
        train_dataset = dataloader['train']
        test_dataset = dataloader['query']
        database_dataset = dataloader['database']
    if settings.DATASET == "NUSWIDE":
        dataloader, data_train = get_loader_nuswide(settings.BATCH_SIZE)
        train_dataset = dataloader['train']
        test_dataset = dataloader['query']
        database_dataset = dataloader['database']
        # train_dataset = datasets.MIRFlickr(train=True, transform=datasets.mir_train_transform)
        # test_dataset = datasets.MIRFlickr(train=False, database=False, transform=datasets.mir_test_transform)
        # database_dataset = datasets.MIRFlickr(train=False, database=True, transform=datasets.mir_test_transform)

    # if settings.DATASET == "NUSWIDE":
    #     train_dataset = datasets.NUSWIDE(train=True, transform=datasets.nus_train_transform)
    #     test_dataset = datasets.NUSWIDE(train=False, database=False, transform=datasets.nus_test_transform)
    #     database_dataset = datasets.NUSWIDE(train=False, database=True, transform=datasets.nus_test_transform)

    # Data Loader (Input Pipeline)
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    #                                            batch_size=settings.BATCH_SIZE,
    #                                            shuffle=True,
    #                                            num_workers=settings.NUM_WORKERS,
    #                                            drop_last=True)
    #
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    #                                           batch_size=settings.BATCH_SIZE,
    #                                           shuffle=False,
    #                                           num_workers=settings.NUM_WORKERS)
    #
    # database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
    #                                               batch_size=settings.BATCH_SIZE,
    #                                               shuffle=False,
    #                                               num_workers=settings.NUM_WORKERS)
    train_loader = train_dataset

    test_loader = test_dataset

    database_loader = database_dataset
    #flickr 0.100000, 10.000000, 1.000000, 0.010000, 0.000100, 1.000000, 1.000000, 0.940705, 0.894480
    #nus 1.000000, 10.000000, 10.000000, 1.000000, 0.300000, 1.000000, 0.010000, 0.860363, 0.810671
    # {0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000} {0.8, 0.9, 1.1, 1.2} {0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2, 5, 10}0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10
    for i in range(100):  # {10}:
        # l1 = 10**(-i+3)
        # l1 = (i+4)*9
        # l1 = 2 ** (-1 * i + 7)

        # l1 = 2**((-1)**i*i+9)
        l1 = i*2 + 66
        for j in range(1):  # {10}:#10
            l2 = 10**(-j-0)
            # l2 = 2 ** (-1 * j + 6)  # 6-9
            # l2 = 2**((-1)**j*j+3)
            # l2 = 0.02*j + 0.1
            for k in range(1):  # range(10):#{100}:
                # l3 = 10**(-k+1)
                # l3 = 2**((-1)**k*k + 4)
                # l3 = 2 ** (-1 * k - 1)
                l3 = 2*k + 10
                for y in range(1):  # {0.1, 0.01, 0.001}: #0.001, 0.01, 0.1
                    l4 = 10
                    # l4 = 10**(-y+1) #0.001-100
                    # l4 = 2**((-1)**y*y+5)
                    # l4 = 2 ** (y - 0)*0.1
                    for i1 in range(1):  # {0.3}: #0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10
                        l5 = i1 * 0.1
                        for j1 in range(1):  # {0.45}:
                            # l6 = j1*0.1 + 5.8
                            l6 = 10**(-j1+2)
                            # l6 = 2 ** (j1 - 1)
                            # l6 = 2**((-1)**j1*j1+0)
                            for k1 in range(1):  # {0.1}: # 0.01, 0.1
                                # l7 = 10**(-k1+4)
                                # l7 = 2 ** (-k1 + 11)
                                # l7 = 2**((-1)**k1*k1+8)
                                # l7 = 2**((-1)**k1*k1+0)*1000
                                l7 = -8*k1 + 2048
                                for y1 in {150}:
                                    l8 = y1
                                    sess = Session(train_loader, test_loader, database_loader, train_dataset,
                                                   test_dataset, database_dataset, data_train, l5, l5, l7)

                                    if settings.EVAL == True:
                                        sess.load_checkpoints()
                                        sess.eval()

                                    else:
                                        for epoch in range(settings.NUM_EPOCH):
                                            # train the Model
                                            sess.train(epoch, l1, l2, l3, l4, l5, l6, l7)
                                            # eval the Model
                                            # if (epoch + 1) % 20 == 0 and epoch < 25:
                                            #     A = sess.eval(l1, l2, l3, l4, l5, l6, l7)
                                            #     # print(A)
                                            #     # if A < 1.72:
                                            #     #     break
                                            if (epoch + 1) % settings.EVAL_INTERVAL == 0 and epoch > 10:
                                                A = sess.eval(l1, l2, l3, l4, l5, l6, l7)
                                                # print(A)
                                                if A < 1.68:
                                                    break
                                            # save the model
                                            # if epoch + 1 == settings.NUM_EPOCH:
                                            #     sess.save_checkpoints(step=epoch + 1)


if __name__ == '__main__':
    main()
