import numpy as np

class GetPBB(object):
    def __init__(self):
        self.stride = 4
        self.anchors = np.array([10.0, 30.0, 60.])

    def __call__(self, output, thresh=-3, ismask=False, stride=4):
        stride = self.stride
        anchors = self.anchors
        output = np.copy(output)
        offset = (float(stride) - 1) / 2
        output_size = output.shape

        oh = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)


        output[:, :, :, 1] = oh.reshape((-1, 1, 1)) + output[:, :, :, 1] * anchors.reshape((1, 1, -1))
        output[:, :, :, 2] = ow.reshape((1, -1, 1)) + output[:, :, :, 2] * anchors.reshape((1, 1, -1))
        output[:, :, :, 3] = np.exp(output[:, :, :, 3]) * anchors.reshape((1, 1, -1))

        mask = output[..., 0] > thresh
        xx, yy, aa = np.where(mask)

        output = output[xx, yy, aa]
        if ismask:
            return output, [xx, yy, aa]
        else:
            return output

def nms(output, nms_th):
    if len(output) == 0:
        return output

    output = output[np.argsort(-output[:, 0])]
    bboxes = [output[0]]

    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j in range(len(bboxes)):
            if iou(bbox[1:4], bboxes[j][1:4]) >= nms_th:
                flag = -1
                break
        if flag == 1:
            bboxes.append(bbox)

    bboxes = np.asarray(bboxes, np.float32)
    return bboxes

def iou(box0, box1):

    r0 = box0[2] / 2
    s0 = box0[:2] - r0
    e0 = box0[:2] + r0

    r1 = box1[2] / 2
    s1 = box1[:2] - r1
    e1 = box1[:2] + r1

    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))

    intersection = overlap[0] * overlap[1]
    union = box0[2] * box0[2] + box1[2] * box1[2] - intersection
    return intersection / union
