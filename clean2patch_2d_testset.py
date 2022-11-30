import os
import numpy as np
import random
import scipy.io as scio
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

class Crop(object):
    def __init__(self):
        self.crop_size = [320, 320]
        self.upsample_size = [320, 320]
        self.bound_size = 30
        self.stride1 = 4
        # self.stride2 = 2
        # self.stride3 = 4
        # self.stride4 = 8
        self.pad_value = 0.233

    def __call__(self, imgs, target, target2, bboxes):
        self.pad_value = imgs[0, 0, 0]
        crop_size = data.shape[1:]
        img_resized = np.array(Image.fromarray(imgs[0, :, :]).resize((self.upsample_size[0], self.upsample_size[1]), Image.Resampling.BICUBIC))
        img_resized = img_resized[np.newaxis, :]
        resize_ratio2 = self.upsample_size[1]/imgs.shape[2]
        resize_ratio1 = self.upsample_size[0]/imgs.shape[1]

        target_resized = np.array([target[0]*resize_ratio1, target[1]*resize_ratio2, target[2]*resize_ratio1, target[3]*resize_ratio2])
        target_resized = target_resized.astype('int')
        bboxes_resized = (resize_ratio1 * bboxes)
        bboxes_resized = bboxes_resized.astype('int')

        target2_resized = np.array([target2[0]*resize_ratio1, target2[1]*resize_ratio2, target2[2]*resize_ratio1, target2[3]*resize_ratio2])
        target2_resized = target2_resized.astype('int')

        imgs = img_resized
        target = target_resized
        target2 = target2_resized
        bboxes = bboxes_resized
        crop_size = self.crop_size
        bound_size = self.bound_size
        # target = np.copy(target)
        # bboxes = np.copy(bboxes)

        start = []
        for i in range(2):
            r = 0.5 * target[i + 2]
            hw = target[i] + r
            s = np.floor(hw - r) + 1 - bound_size
            e = np.ceil(hw + r) + 1 + bound_size - crop_size[i]
            if s > e:
                start.append(np.random.randint(e, s))  # !
            else:
                start.append(int(hw) - crop_size[i] / 2 + np.random.randint(-bound_size / 2, bound_size / 2))
        start = [0, 0]

        normstart = np.array(start).astype('float32') / np.array(imgs.shape[1:]) - 0.5
        normsize = np.array(crop_size).astype('float32') / np.array(imgs.shape[1:])
        xx, yy= np.meshgrid(
            np.linspace(normstart[0], normstart[0] + normsize[0], crop_size[0] // self.stride1),
            np.linspace(normstart[1], normstart[1] + normsize[1], crop_size[1] // self.stride1), indexing='ij')
        coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...]], 0).astype('float32')

        pad = []
        pad.append([0, 0])
        for i in range(2):
            leftpad = max(0, -start[i])
            rightpad = max(0, start[i] + crop_size[i] - imgs.shape[i + 1])
            pad.append([leftpad, rightpad])


        crop = imgs[:,
               max(start[0], 0):min(start[0] + crop_size[0], imgs.shape[1]),
               max(start[1], 0):min(start[1] + crop_size[1], imgs.shape[2])]
        crop = np.pad(crop, pad, 'constant', constant_values=self.pad_value)

        for i in range(2):
            target[i] = target[i] - start[i]
        for i in range(2):
            target2[i] = target2[i] - start[i]
        for i in range(len(bboxes)):
            for j in range(2):
                bboxes[i][j] = bboxes[i][j] - start[j]

        return crop, target, target2, bboxes, coord

class LabelMapping(object):
    def __init__(self):
        self.stride1 = 1
        self.stride2 = 2
        self.stride3 = 4
        self.stride4 = 8



    def __call__(self, input_size, target, bboxes, stride):
        anchors = np.array([10.0, 30.0, 60.])

        output_size = []
        for i in range(2):
            assert (input_size[i] % stride == 0)
            output_size.append(input_size[i] / stride)

        label = -1 * np.ones([32, 32, 3, 5])
        offset = (stride - 1) / 2

        oh = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)

        for bbox in bboxes:
            for i, anchor in enumerate(anchors):
                ih, iw = select_samples(bbox, anchor, oh, ow)
                label[ih, iw, i, 0] = 0

        ih, iw, ia = [], [], []
        for i, anchor in enumerate(anchors):
            iih, iiw = select_samples(target, anchor, oh, ow)

            ih.append(iih)
            iw.append(iiw)
            ia.append(i * np.ones((len(iih),), np.int64))

        ih = np.concatenate(ih, 0)
        iw = np.concatenate(iw, 0)
        ia = np.concatenate(ia, 0)

        flag = True
        if len(ih) == 0:
            pos = []
            for i in range(2):
                pos.append(max(0, int(np.round((target[i] - offset) / stride))))
            idx = np.argmin(np.abs(np.log(max(target[2], target[3]) / anchors)))
            pos.append(idx)
            flag = False
        else:
            idx = random.sample(range(len(ih)), 1)[0]
            pos = [ih[idx], iw[idx], ia[idx]]

        dh = (target[0] + 0.5 * target[2] - oh[pos[0]]) / anchors[pos[2]]
        dw = (target[1] + 0.5 * target[3] - ow[pos[1]]) / anchors[pos[2]]
        ddh = np.log(target[2] / anchors[pos[2]])
        ddw = np.log(target[3] / anchors[pos[2]])
        label[pos[0], pos[1], pos[2], :] = [1, dh, dw, ddh, ddw]
        return label

def select_samples(bbox, anchor, oh, ow):
    h, w, dh, dw = bbox
    h = h + 0.5 * dh
    w = w + 0.5 * dw
    max_overlap = min(dh, dw, anchor)
    min_overlap = np.power(max(dh, dw, anchor), 3) / max_overlap / max_overlap

    if min_overlap > max_overlap:
        return np.zeros((0,), np.int64), np.zeros((0,), np.int64)
    else:
        s = h - 0.5 * np.abs(dh - anchor) - (max_overlap - min_overlap)
        e = h + 0.5 * np.abs(dh - anchor) + (max_overlap - min_overlap)
        mh = np.logical_and(oh >= s, oh <= e)
        ih = np.where(mh)[0]

        s = w - 0.5 * np.abs(dw - anchor) - (max_overlap - min_overlap)
        e = w + 0.5 * np.abs(dw - anchor) + (max_overlap - min_overlap)
        mw = np.logical_and(ow >= s, ow <= e)
        iw = np.where(mw)[0]

        if len(ih) == 0 or len(iw) == 0:
            return np.zeros((0,), np.int64), np.zeros((0,), np.int64)

        lh, lw = len(ih), len(iw)

        ih = ih.reshape((1, -1, 1))
        iw = iw.reshape((1, 1, -1))

        ih = np.tile(ih, (1, lw)).reshape((-1))
        iw = np.tile(iw, (lh, 1)).reshape((-1))

    return ih, iw

if __name__ == '__main__':

    clean_data_path = 'C:/Users/sssnow/Desktop/test_data/coronal/'
    # idcs_list = [f.split('.')[0] for f in os.listdir(clean_data_path)]
    # idcs = sorted(set(idcs_list), key=idcs_list.index)
    # filenames = [os.path.join(clean_data_path, '%s.mat' % idx) for idx in idcs]
    test_th = -1
    for count in range(1, 241):

        info = scio.loadmat(clean_data_path + '2_' + str(count))
        data = info['slice']
        label = info['slice_location1']
        label2 = info['slice_location2']
        # label = np.array([label[0, 0] + 0.5 * label[0, 2], label[0, 1] + 0.5 * label[0, 3],
        #                   min(label[0, 2], label[0, 3])]).astype(int)

        label = np.array([label[0, 0], label[0, 1], label[0, 2], label[0, 3]]).astype(int)
        bboxes = label.copy()
        bboxes = bboxes[np.newaxis, :]

        label2 = np.array([label2[0, 0], label2[0, 1], label2[0, 2], label2[0, 3]]).astype(int)
        bboxes2 = label2.copy()
        bboxes2 = bboxes2[np.newaxis, :]

        data = data[np.newaxis, :]
        data = np.transpose(data, (0, 2, 1))
        # data = np.swapaxes(data, 1, 3)
        ori_data = data

        # if sum(label) != 0 or sum(label2) != 0:
        #     ax = plt.subplot(1, 1, 1)
        #     plt.imshow(data[0], 'gray')
        #     plt.axis('off')
        #
        #     rect = patches.Rectangle((label2[1], label2[0]), label2[3], label2[2], linewidth=2,
        #                              edgecolor='red', facecolor='none')
        #     ax.add_patch(rect)
        #     plt.show()

        label_mapping = LabelMapping()
        crop = Crop()

        sample, target, target2, bboxes, coord32 = crop(data, label.squeeze(), label2.squeeze(), bboxes)
        # label32 = label_mapping(sample.shape[1:], target, bboxes, stride=4)

        # if target[0] != 0 or target2[0] != 0:
        #     ax = plt.subplot(1, 1, 1)
        #     plt.imshow(sample[0], 'gray')
        #     plt.axis('off')
        #     if target[0] != 0:
        #         rect = patches.Rectangle((target[1], target[0]), target[3], target[2], linewidth=1,
        #                                  edgecolor='red', facecolor='none')
        #         ax.add_patch(rect)
        #     if target2[0] != 0:
        #         rect2 = patches.Rectangle((target2[1], target2[0]), target2[3], target2[2], linewidth=1,
        #                                  edgecolor='red', facecolor='none')
        #         ax.add_patch(rect2)
        #     plt.show()


        data = [sample, target, target2, coord32]


        if count > test_th:
            np.save('test_data_3plane/coronal/' + str(count) + '.npy', data)
        else:
            np.save('coronal_sagittal_2d/' + str(count) + '.npy', data)

        print(count+1)
        print(ori_data.shape[1:])
    print('Prepare Done')