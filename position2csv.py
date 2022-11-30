import os
import torch
import numpy as np
import argparse
from get_pbb2d_test import GetPBB, nms, iou
# from split_combine import SplitComb
import time
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.io as scio

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

upsample_size = [320, 320]

with open(os.path.join('res_csv/', "res_coronal.csv"), 'w') as f:
    f.write("coordX,coordY,coordZ,dx,dy,probability\n")

    for model_num in [50]:

        parser = argparse.ArgumentParser(description="MRS Demo")
        parser.add_argument("--cuda", action="store_true", help="use cuda?")
        parser.add_argument("--model", default="checkpoint_coronal_implant_2d/model_epoch_" + str(model_num) + ".pth",
                            type=str, help="model path")
        # parser.add_argument("--model", default="checkpoint_256/model_epoch_100.pth", type=str, help="model path")
        parser.add_argument("--gpus", default="1", type=str, help="gpu ids (default: 0)")

        opt = parser.parse_args()
        cuda = opt.cuda

        if cuda:
            print("=> use gpu id: '{}'".format(opt.gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
            if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

        model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]
        device = torch.device('cuda:0')
        get_pbb = GetPBB()
        clean_data_path = 'C:/Users/sssnow/Desktop/test_data/coronal/'
        ori_info = scio.loadmat(clean_data_path + '2_' + str(1))
        ori_data = ori_info['slice']
        ori_data = ori_data[np.newaxis, :]
        ori_data = np.transpose(ori_data, (0, 2, 1))

        for slice_num in range(1, 241):
            # for slice_num in [70]:
            info = np.load('test_data_3plane/coronal/' + str(slice_num) + '.npy', allow_pickle=True)
            data = info.tolist()
            # For Self Generating data
            patch = data[0]
            label1 = data[1]
            label2 = data[2]
            coord32 = data[3]

            img = np.copy(patch)

            patch = torch.from_numpy(patch).to(torch.float)
            label1 = torch.from_numpy(np.array(label1)).to(torch.float)
            label2 = torch.from_numpy(np.array(label2)).to(torch.float)
            coord32 = torch.from_numpy(np.array(coord32)).to(torch.float)

            patch = torch.unsqueeze(patch, 0)
            label1 = torch.unsqueeze(label1, 0)
            label2 = torch.unsqueeze(label2, 0)
            coord32 = torch.unsqueeze(coord32, 0)

            model = model.to(device)
            patch = patch.to(device)
            coord32 = coord32.to(device)

            pred32 = model(patch, coord32)

            thresh = -3

            pred32 = pred32[0].data.cpu().numpy()

            pbb32, mask32 = get_pbb(pred32, thresh, ismask=True, stride=4)

            pbb = pbb32

            pbb = pbb[pbb[:, 0] > 1]
            pbb = nms(pbb, 0.1)
            label1 = label1[0].data.cpu().numpy()
            label2 = label2[0].data.cpu().numpy()

            resize_ratio2 = upsample_size[1] / ori_data.shape[2]
            resize_ratio1 = upsample_size[0] / ori_data.shape[1]

            if not pbb is None:
                for single_pbb in pbb:
                    f.write("%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n" % (slice_num-1, (single_pbb[1])/resize_ratio1,
                                                                 (single_pbb[2])/resize_ratio2,
                                                                 single_pbb[3]/resize_ratio1,
                                                                 single_pbb[4]/resize_ratio2, single_pbb[0]))


with open(os.path.join('res_csv/', "res_sagittal.csv"), 'w') as f:
    f.write("coordX,coordY,coordZ,dx,dy,probability\n")

    for model_num in [50]:

        parser = argparse.ArgumentParser(description="MRS Demo")
        parser.add_argument("--cuda", action="store_true", help="use cuda?")
        parser.add_argument("--model", default="checkpoint_sagittal_implant_2d/model_epoch_"+ str(model_num) +".pth", type=str, help="model path")
        # parser.add_argument("--model", default="checkpoint_256/model_epoch_100.pth", type=str, help="model path")
        parser.add_argument("--gpus", default="1", type=str, help="gpu ids (default: 0)")

        opt = parser.parse_args()
        cuda = opt.cuda

        if cuda:
            print("=> use gpu id: '{}'".format(opt.gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
            if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

        model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]
        device = torch.device('cuda:0')
        get_pbb = GetPBB()

        clean_data_path = 'C:/Users/sssnow/Desktop/test_data/sagittal/'
        ori_info = scio.loadmat(clean_data_path + '2_' + str(1))
        ori_data = ori_info['slice']
        ori_data = ori_data[np.newaxis, :]
        ori_data = np.transpose(ori_data, (0, 2, 1))

        for slice_num in range(1, 241):
        # for slice_num in [70]:
            info = np.load('test_data_3plane/sagittal/' + str(slice_num) + '.npy', allow_pickle=True)
            data = info.tolist()
            # For Self Generating data
            patch = data[0]
            label1 = data[1]
            label2 = data[2]
            coord32 = data[3]

            img = np.copy(patch)

            patch = torch.from_numpy(patch).to(torch.float)
            label1 = torch.from_numpy(np.array(label1)).to(torch.float)
            label2 = torch.from_numpy(np.array(label2)).to(torch.float)
            coord32 = torch.from_numpy(np.array(coord32)).to(torch.float)

            patch = torch.unsqueeze(patch, 0)
            label1 = torch.unsqueeze(label1, 0)
            label2 = torch.unsqueeze(label2, 0)
            coord32 = torch.unsqueeze(coord32, 0)

            model = model.to(device)
            patch = patch.to(device)
            coord32 = coord32.to(device)

            pred32 = model(patch, coord32)

            thresh = -3

            pred32 = pred32[0].data.cpu().numpy()

            pbb32, mask32 = get_pbb(pred32, thresh, ismask=True, stride=4)

            pbb = pbb32

            pbb = pbb[pbb[:, 0] > 1]
            pbb = nms(pbb, 0.1)
            label1 = label1[0].data.cpu().numpy()
            label2 = label2[0].data.cpu().numpy()

            resize_ratio2 = upsample_size[1] / ori_data.shape[2]
            resize_ratio1 = upsample_size[0] / ori_data.shape[1]

            if not pbb is None:
                for single_pbb in pbb:
                    f.write("%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n" % (
                    (single_pbb[1]) / resize_ratio1, slice_num - 1, (single_pbb[2]) / resize_ratio2,
                    single_pbb[3] / resize_ratio1, single_pbb[4] / resize_ratio2, single_pbb[0]))


with open(os.path.join('res_csv/', "res_axial.csv"), 'w') as f:
    f.write("coordX,coordY,coordZ,dx,dy,probability\n")

    for model_num in [26]:

        parser = argparse.ArgumentParser(description="MRS Demo")
        parser.add_argument("--cuda", action="store_true", help="use cuda?")
        parser.add_argument("--model", default="checkpoint_axial_implant_2d/model_epoch_"+ str(model_num) +".pth", type=str, help="model path")
        # parser.add_argument("--model", default="checkpoint_256/model_epoch_100.pth", type=str, help="model path")
        parser.add_argument("--gpus", default="1", type=str, help="gpu ids (default: 0)")

        opt = parser.parse_args()
        cuda = opt.cuda

        if cuda:
            print("=> use gpu id: '{}'".format(opt.gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
            if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

        model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]
        device = torch.device('cuda:0')
        get_pbb = GetPBB()

        clean_data_path = 'C:/Users/sssnow/Desktop/test_data/axial/'
        ori_info = scio.loadmat(clean_data_path + '2_' + str(1))
        ori_data = ori_info['slice']
        ori_data = ori_data[np.newaxis, :]
        ori_data = np.transpose(ori_data, (0, 2, 1))

        for slice_num in range(1, 162):
        # for slice_num in [70]:
            info = np.load('test_data_3plane/axial/' + str(slice_num) + '.npy', allow_pickle=True)
            data = info.tolist()
            # For Self Generating data
            patch = data[0]
            label1= data[1]
            label2 = data[2]
            coord32 = data[3]

            img = np.copy(patch)

            patch = torch.from_numpy(patch).to(torch.float)
            label1 = torch.from_numpy(np.array(label1)).to(torch.float)
            label2 = torch.from_numpy(np.array(label2)).to(torch.float)
            coord32 = torch.from_numpy(np.array(coord32)).to(torch.float)

            patch = torch.unsqueeze(patch, 0)
            label1 = torch.unsqueeze(label1, 0)
            label2 = torch.unsqueeze(label2, 0)
            coord32 = torch.unsqueeze(coord32, 0)

            model = model.to(device)
            patch = patch.to(device)
            coord32 = coord32.to(device)

            pred32 = model(patch, coord32)

            thresh = -3

            pred32 = pred32[0].data.cpu().numpy()

            pbb32, mask32 = get_pbb(pred32, thresh, ismask=True, stride=4)

            pbb = pbb32

            pbb = pbb[pbb[:, 0] > -1]
            pbb = nms(pbb, 0.9)
            label1 = label1[0].data.cpu().numpy()
            label2 = label2[0].data.cpu().numpy()

            resize_ratio2 = upsample_size[1] / ori_data.shape[2]
            resize_ratio1 = upsample_size[0] / ori_data.shape[1]

            if not pbb is None:
                for single_pbb in pbb:
                    f.write("%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n" % (
                        (single_pbb[1]) / resize_ratio1, (single_pbb[2]) / resize_ratio2, slice_num - 1,
                        single_pbb[3] / resize_ratio1, single_pbb[4] / resize_ratio2, single_pbb[0]))

