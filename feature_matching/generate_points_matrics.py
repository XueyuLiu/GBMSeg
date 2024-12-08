import os
import sys

# current_path = os.path.abspath(os.path.dirname(__file__))
# sys.path.append(os.path.join(current_path,'dinov2'))

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


def convert_to_rgb(image):
    if image.mode == 'RGBA':
        return image.convert('RGB')
    return image

def forward_matching(images_inner, index, device, dino,size):
    transform = T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    #     # images_inner = [Image.open(image_path).resize((560, 560)) for image_path in image_paths]
    imgs_tensor = torch.stack([transform(convert_to_rgb(img))[:3] for img in images_inner]).to(device)
    print("img_tensor input shape:", imgs_tensor.shape)
    with torch.no_grad():
        features_dict = dino.forward_features(imgs_tensor)
        features = features_dict['x_norm_patchtokens']
        print(features.shape)
    fore_index = torch.tensor(index)
    distances = torch.cdist(features[0][fore_index].squeeze(1), features[1])
    min_values, min_indices = distances.min(dim=1)


    #print(self_distances)
    #print(self_distances[0].shape)



    return images_inner, features, min_indices




def loading_dino(device):
    dino = hubconf.dinov2_vitg14()
    dino.to(device)

    return dino


def distance_calculate(features,indices_pos,indices_back,size):
    final_pos_points = torch.tensor(calculate_center_points(indices_pos,size))
    final_neg_points = torch.tensor(calculate_center_points(indices_back,size))
    print(f"Shape of features[1][indices_pos]: {features[1][indices_pos].shape}")
    print(f"Shape of features[1][indices_back]: {features[1][indices_back].shape}")

    feature_pos_distances = torch.cdist(features[1][indices_pos], features[1][indices_pos])
    #feature_neg_distances = torch.cdist(features[1][initial_indices_back], features[1][initial_indices_back])
    feature_cross_distances= torch.cdist(features[1][indices_pos], features[1][indices_back])

    physical_pos_distances=torch.cdist(final_pos_points,final_pos_points)
    #physical_neg_distances=torch.cdist(final_pos_points,final_pos_points)
    physical_cross_distances=torch.cdist(final_pos_points,final_neg_points)

    return feature_pos_distances,feature_cross_distances,physical_pos_distances,physical_cross_distances


def points_generate(indices_pos,indices_neg):
    final_pos_points = calculate_center_points(indices_pos,size)
    final_neg_points = calculate_center_points(indices_neg,size)



    #final_hard_points = calculate_center_points(ng_indices,size)

    # fin_center_points,max_center_points=process_image_pair(image_paths, fore_index,output_dir)

    # Convert points to tuples and remove duplicate points
    final_pos_points = set(tuple(point) for point in final_pos_points)
    final_neg_points = set(tuple(point) for point in final_neg_points)
    image = images_inner[1]
    final_pos_points_map = map_to_ori_size(list(final_pos_points), [image.size[1], image.size[0]],size)
    final_neg_points_map = map_to_ori_size(list(final_neg_points), [image.size[1], image.size[0]],size)

    return final_pos_points_map,final_neg_points_map


def generate_indices(mask, image_inner, device, dino,size):

    # print('---------Start generating the initial prompting scheme---------')
    mask = np.array(mask)  # standardization
    mask_np = np.repeat(mask[:, :, np.newaxis], 3, axis=2)  # Expanded to 3 channels
    fore_patchs, fore_index, back_patchs, back_index = find_foreground_patches(mask_np,size)
    # dino = hubconf.dinov2_vitg14()
    # dino.to(device)
    #print(fore_patchs,back_patchs)
    # foward matching
    images_inner, features, initial_indices= forward_matching(image_inner, fore_index, device, dino,size)
    images_inner_back, features_back, initial_indices_back= forward_matching(image_inner, back_index,
                                                                                            device, dino,size)



    return features,initial_indices, initial_indices_back