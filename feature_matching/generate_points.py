import os
import sys


import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from . import hubconf
from tqdm import tqdm
from sklearn.cluster import KMeans


def find_foreground_patches(mask_np,size):
    fore_patchs = []
    back_patchs = []

    for i in range(0, mask_np.shape[0] - 14, 14):
        for j in range(0, mask_np.shape[1] - 14, 14):
            if np.all(mask_np[i:i + 14, j:j + 14] != [0, 0, 0]):
                fore_patchs.append((i, j))

            if np.all(mask_np[i:i + 14, j:j + 14] == [0, 0, 0]):
                back_patchs.append((i, j))

    fore_index = np.empty((len(fore_patchs), 1), dtype=int)
    back_index = np.empty((len(back_patchs), 1), dtype=int)

    for i in range(len(fore_patchs)):
        row = fore_patchs[i][0] / 14
        col = fore_patchs[i][1] / 14
        fore_index[i] = row * (size/14) + col

    for i in range(len(back_patchs)):
        row = back_patchs[i][0] / 14
        col = back_patchs[i][1] / 14
        back_index[i] = row * (size/14) + col

    return fore_patchs, fore_index, back_patchs, back_index
    # return fore_patchs


# Calculate the center pixel point of each patch
def calculate_center_points(indices,size):
    # print(indices)
    center_points = []
    indices = indices.cpu().numpy()

    for i in range(len(indices)):
        row_index = indices[i] // (size/14)
        col_index = indices[i] % (size/14)
        center_x = col_index * 14 + 14 // 2
        center_y = row_index * 14 + 14 // 2
        center_points.append([center_x, center_y])

    return center_points


# Mapping the extracted coordinates back to the coordinate points corresponding to the original image size
def map_to_ori_size(resized_coordinates, original_size,size):
    # Assuming that the height of the original image is H and the width is W
    original_height, original_width = original_size

    # Calculate height and width scaling
    scale_height = original_height / size
    scale_width = original_width / size

    if isinstance(resized_coordinates, tuple):
        # If the input is a single coordinate (x, y)
        resized_x, resized_y = resized_coordinates
        original_x = resized_x * scale_width
        original_y = resized_y * scale_height
        return original_x, original_y
    elif isinstance(resized_coordinates, list):
        # If the input is a list of coordinates [(x1, y1), (x2, y2), ...]
        original_coordinates = [[round(x * scale_width), round(y * scale_height)] for x, y in resized_coordinates]
        return original_coordinates
    else:
        raise ValueError("Unsupported input format. Please provide a tuple or list of coordinates.")


def forward_matching(images_inner, index, device, dino,size):
    transform = T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    #     # images_inner = [Image.open(image_path).resize((560, 560)) for image_path in image_paths]
    imgs_tensor = torch.stack([transform(img)[:3] for img in images_inner]).to(device)
    with torch.no_grad():
        features_dict = dino.forward_features(imgs_tensor)
        features = features_dict['x_norm_patchtokens']
    fore_index = torch.tensor(index)
    distances = torch.cdist(features[0][fore_index].squeeze(1), features[1])
    min_values, min_indices = distances.min(dim=1)
    # print('min_indices:',len(min_indices))
    # print(min_indices)
    max_values, max_indices = distances.max(dim=1)
    for_center_points = calculate_center_points(min_indices,size)

    return images_inner, features, min_indices, max_indices


def backward_matching(features, index, min_indices, device, dino,size):
    re_distances = torch.cdist(features[1][min_indices], features[0])
    re_min_values, re_min_indices = re_distances.min(dim=1)
    #re1_center_points = calculate_center_points(re_min_indices,size)

    # Remove indexes in min_indices that make re_min_indices not in fore_index after reverse matching
    ng_min_indices = []
    for i in range(len(re_min_indices)):
        if re_min_indices[i].item() not in index.squeeze(1):
            ng_min_indices.append(i)
    #print("min_indices:", re_min_indices)
    #print("ng_min_indices:", ng_min_indices)
    #print("torch.tensor.ng_min_indices:", torch.tensor(ng_min_indices))

    # Record deleted indexes
    ng_indices_initial = min_indices[ng_min_indices]
    ng_distances=torch.cdist(features[1][ng_indices_initial].squeeze(1),features[0])
    hard_min_values,hard_min_indices=ng_distances.min(dim=1)
    mean_distance = torch.mean(hard_min_values)
    #print(ng_distances)
    ng_indices_final = torch.where(hard_min_values<= mean_distance)
    #print(ng_indices_final)
    indices = []
    for i in ng_indices_final[0]:
        indices.append(ng_indices_initial[i])

    #ng_indices=ng_indices_initial[ng_min_indices[ng_indices_final]]
    #ng_indices=torch.cat(indices)
    ng_indices=torch.stack(indices)
    #print(ng_indices)
    #ng_indices=






    ng_points = calculate_center_points(torch.tensor(ng_indices),size)
    # print("ng_points:",ng_points)


    # Delete Index
    filtered_min_indices = min_indices
    #print(len(filtered_min_indices))
    filtered_min_indices[ng_min_indices] = -1
    filtered_min_indices = filtered_min_indices[filtered_min_indices != -1]
    #print(len(filtered_min_indices))



    return ng_indices, filtered_min_indices, re_min_indices


def negative_points(features, max_indices, index, re_min_indices, device, dino,size):
    re_ng_distances = torch.cdist(features[1][max_indices], features[0])
    re_min_values, re_ng_min_indices = re_ng_distances.min(dim=1)
    indices = []
    for i in range(len(re_min_indices)):
        if re_ng_min_indices[i].item() in index.squeeze(1):
            indices.append(i)


    filtered_ng_indices = max_indices
    filtered_ng_indices[indices] = -1
    filtered_ng_indices = filtered_ng_indices[filtered_ng_indices != -1]

    #close_patch_indices = torch.where(row_mean_distance <= mean_distance)


    re2_max_center_points = calculate_center_points(filtered_ng_indices,size)
    return filtered_ng_indices



def self_matching(features, filtered_min_indices, device, dino):
    self_distances = torch.cdist(features[1][filtered_min_indices], features[1][filtered_min_indices])
    mean_distance = torch.mean(self_distances)
    row_mean_distance = torch.mean(self_distances, dim=1)

    close_patch_indices = torch.where(row_mean_distance <= mean_distance)
    filtered_min_indices = filtered_min_indices[close_patch_indices[0]]

    return filtered_min_indices


def loading_dino(device):
    dino = hubconf.dinov2_vitg14()
    dino.to(device)

    return dino


def generate(mask, image_inner, device, dino,size):
    # print('---------Start generating the initial prompting scheme---------')
    mask = np.array(mask)  # standardization
    mask_np = np.repeat(mask[:, :, np.newaxis], 3, axis=2)  # Expanded to 3 channels
    fore_patchs, fore_index, back_patchs, back_index = find_foreground_patches(mask_np,size)
    # dino = hubconf.dinov2_vitg14()
    # dino.to(device)

    # foward matching
    images_inner, features, min_indices, max_indices = forward_matching(image_inner, fore_index, device, dino,size)
    images_inner_back, features_back, min_indices_back, max_indices_back = forward_matching(image_inner, back_index,
                                                                                            device, dino,size)

    # backward matching
    ng_indices, filtered_min_indices, re_min_indices = backward_matching(features, fore_index, min_indices, device,
                                                                         dino,size)
    ng_indices_back, filtered_min_indices_back, re_min_indices_back = backward_matching(features, back_index,
                                                                                        min_indices_back, device, dino,size)

    # negative sample
    ng_indices = negative_points(features, max_indices, fore_index, re_min_indices, device, dino,size)
    ng_indices_back = negative_points(features, max_indices_back, back_index, re_min_indices_back, device, dino,size)

    # self matching
    #filtered_min_indices = self_matching(features, filtered_min_indices, device, dino)
    #filtered_min_indices_back = self_matching(features, filtered_min_indices_back, device, dino)
    #filtered_min_indices_back = torch.cat([ng_indices, filtered_min_indices_back], dim=0)


    #filtered_min_indices_back=torch.cat([ng_indices_back, filtered_min_indices_back], dim=0)
    #filtered_min_indices=min_indices
    final_pos_points = calculate_center_points(filtered_min_indices,size)
    final_neg_points = calculate_center_points(filtered_min_indices_back,size)
    final_hard_points = calculate_center_points(ng_indices,size)

    # fin_center_points,max_center_points=process_image_pair(image_paths, fore_index,output_dir)

    # Convert points to tuples and remove duplicate points
    final_pos_points = set(tuple(point) for point in final_pos_points)
    final_neg_points = set(tuple(point) for point in final_neg_points)
    image = images_inner[1]
    final_pos_points_map = map_to_ori_size(list(final_pos_points), [image.size[1], image.size[0]],size)
    #print([image.size[1], image.size[0]],size)
    final_neg_points_map = map_to_ori_size(list(final_neg_points), [image.size[1], image.size[0]],size)
    final_hard_points_map = map_to_ori_size(list(final_hard_points), [image.size[1], image.size[0]],size)

    #print(final_pos_points_map,final_neg_points_map)
    return features,final_pos_points_map, final_neg_points_map, final_hard_points_map

