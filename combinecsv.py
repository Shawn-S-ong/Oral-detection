import pandas as pd
import numpy as np
import nibabel as nib

def nms(output, nms_th=0.5):
    if len(output) == 0:
        return output

    # output = output.transpose(1, 0)

    output = output[np.argsort(-output[:, 5])]
    bboxes = [output[0]]

    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j in range(len(bboxes)):
            if iou(bbox[0:5], bboxes[j][0:5]) <= nms_th:
                flag = -1
                break
        if flag == 1:
            bboxes.append(bbox)

    bboxes = np.asarray(bboxes, np.float32)
    return bboxes

def iou(box0, box1):
    s0 = [box0[0]-0.5*(box0[3]+box0[4]), box0[1]-0.5*(box0[3]+box0[4]), box0[2]-0.5*(box0[3]+box0[4])]
    e0 = [box0[0]+0.5*(box0[3]+box0[4]), box0[1]+0.5*(box0[3]+box0[4]), box0[2]+0.5*(box0[3]+box0[4])]

    s1 = [box1[0]-0.5*(box1[3]+box1[4]), box1[1]-0.5*(box1[3]+box1[4]), box1[2]-0.5*(box1[3]+box1[4])]
    e1 = [box1[0]+0.5*(box1[3]+box1[4]), box1[1]+0.5*(box1[3]+box1[4]), box1[2]+0.5*(box1[3]+box1[4])]

    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))

    intersection = overlap[0] * overlap[1] * overlap[2]
    union = (e0[0] - s0[0]) * (e0[1] - s0[1]) * (e0[2] - s0[2]) + (e1[0] - s1[0]) * \
            (e1[1] - s1[1]) * (e1[2] - s1[2]) - intersection
    return intersection / union


axial_data = pd.read_csv('res_csv/res_axial.csv')
coronal_data = pd.read_csv('res_csv/res_coronal.csv')
sagittal_data = pd.read_csv('res_csv/res_sagittal.csv')

axial_data = axial_data.to_numpy()
coronal_data = coronal_data.to_numpy()
sagittal_data = sagittal_data.to_numpy()

# axial_data_idcs = axial_data[:, 5] > 4
# axial_data = axial_data[axial_data_idcs]
axial_data = axial_data[:, :5]

# coronal_data_idcs = coronal_data[:, 5] > 4
# coronal_data = coronal_data[coronal_data_idcs]
coronal_data = coronal_data[:, :5]

# sagittal_data_idcs = sagittal_data[:, 5] > 4
# sagittal_data = sagittal_data[sagittal_data_idcs]
sagittal_data = sagittal_data[:, :5]


heatmap1 = np.zeros([240, 240, 161])
for single_data in axial_data:
    location = single_data.astype('int')
    if location[0]+location[3]<240 and location[1]+location[4]<240 and location[2]<161:
        heatmap1[location[1]-location[4]:location[1]+location[4], location[0]-location[3]:location[0]+location[3], location[2]] += 1

heatmap2 = np.zeros([240, 240, 161])
for single_data in coronal_data:
    location = single_data.astype('int')
    if location[0]<240 and location[1]+location[3]<240 and location[2]+location[4]<161:
        heatmap2[location[0], location[2]-location[4]:location[2]+location[4], location[1]-location[3]:location[1]+location[3]] += 1

heatmap3 = np.zeros([240, 240, 161])
for single_data in sagittal_data:
    location = single_data.astype('int')
    if location[0]+location[3]<240 and location[1]<240 and location[2]+location[4]<161:
        heatmap3[location[2]-location[4]:location[2]+location[4], location[1], location[0]-location[3]:location[0]+location[3]] += 1

heatmap = heatmap1 * heatmap2 * heatmap3
heatmap = heatmap/np.max(heatmap)

nib_heatmap = nib.Nifti1Image(heatmap, affine=np.eye(4))
nib.save(nib_heatmap, "heatmap_final.nii")
a=1