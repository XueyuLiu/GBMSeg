import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import hubconf
from tqdm import tqdm
from sklearn.cluster import KMeans




def find_foreground_patches(mask_np):
    fore_patchs = []
    back_patchs=[]

    for i in range(0, mask_np.shape[0] - 14, 14):
        for j in range(0, mask_np.shape[1] - 14, 14):
            if np.all(mask_np[i:i+14, j:j+14] != [0, 0, 0]):
                fore_patchs.append((i, j))

            if np.all(mask_np[i:i+14, j:j+14] == [0, 0, 0]):
                back_patchs.append((i, j))

    fore_index = np.empty((len(fore_patchs), 1), dtype=int)
    back_index = np.empty((len(back_patchs), 1), dtype=int)

    for i in range(len(fore_patchs)):
        row = fore_patchs[i][0] / 14
        col = fore_patchs[i][1] / 14
        fore_index[i] = row * 40 + col

    for i in range(len(back_patchs)):
        row = back_patchs[i][0] / 14
        col = back_patchs[i][1] / 14
        back_index[i] = row * 40 + col

    return fore_patchs, fore_index, back_patchs, back_index
    # return fore_patchs

# Calculate the center pixel point of each patch
def calculate_center_points(indices):
    #print(indices)
    center_points = []
    indices = indices.cpu().numpy()
    
    for i in range(len(indices)):
        row_index = indices[i] // 40
        col_index = indices[i] % 40
        center_x = col_index * 14 + 14 // 2
        center_y = row_index * 14 + 14 // 2
        center_points.append([center_x, center_y])

    return center_points

# Mapping the extracted coordinates back to the coordinate points corresponding to the original image size
def map_to_ori_size(resized_coordinates, original_size):

    # Assuming that the height of the original image is H and the width is W
    original_height, original_width = original_size

    # Calculate height and width scaling
    scale_height = original_height / 560
    scale_width = original_width / 560

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



def forward_matching(image_paths,index):
    transform = T.Compose([
        T.Resize(560),
        T.CenterCrop(560),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    images_inner = [Image.open(image_path).resize((560, 560)) for image_path in image_paths]
    imgs_tensor = torch.stack([transform(img)[:3] for img in images_inner]).to(device)
    with torch.no_grad():
        features_dict = dino.forward_features(imgs_tensor)
        features = features_dict['x_norm_patchtokens']
    fore_index = torch.tensor(index)
    distances = torch.cdist(features[0][fore_index].squeeze(1), features[1])
    min_values, min_indices = distances.min(dim=1)
    #print('min_indices:',len(min_indices))
    #print(min_indices)
    max_values, max_indices = distances.max(dim=1)
    for_center_points = calculate_center_points(min_indices)


    #max_center_points = calculate_center_points(max_indices)
    return images_inner,features, min_indices,max_indices


def backward_matching(features,index,min_indices):
    re_distances = torch.cdist(features[1][min_indices], features[0])
    re_min_values, re_min_indices = re_distances.min(dim=1)
    re1_center_points = calculate_center_points(re_min_indices)


  
    #Remove indexes in min_indices that make re_min_indices not in fore_index after reverse matching
    ng_min_indices = []
    for i in range(len(re_min_indices)):
        if re_min_indices[i].item() not in index.squeeze(1):
            ng_min_indices.append(i)
    # print("ng_min_indices:", ng_min_indices)
    # print("torch.tensor.ng_min_indices:", torch.tensor(ng_min_indices))
    
    #Record deleted indexes
    ng_indices=min_indices[ng_min_indices]
    ng_points=calculate_center_points(torch.tensor(ng_indices))
    # print("ng_points:",ng_points)

    
    #Delete Index
    filtered_min_indices = min_indices
    filtered_min_indices[ng_min_indices] = -1
    filtered_min_indices = filtered_min_indices[filtered_min_indices != -1]

    #re2_center_points = calculate_center_points(filtered_min_indices)


    return ng_indices,filtered_min_indices,re_min_indices

def negative_points(max_indices,index,re_min_indices):
    re_ng_distances = torch.cdist(features[1][max_indices], features[0])
    re_min_values, re_ng_min_indices = re_ng_distances.min(dim=1)
    indices = []
    for i in range(len(re_min_indices)):
        if re_ng_min_indices[i].item() in index.squeeze(1):
            indices.append(i)
    filtered_ng_indices = max_indices
    filtered_ng_indices[indices] = -1
    filtered_ng_indices = filtered_ng_indices[filtered_ng_indices != -1]
    re2_max_center_points = calculate_center_points(filtered_ng_indices)
    return filtered_ng_indices


def cluster(features,all_features,indices,label):
    kmeans= KMeans(n_clusters=2)    
    kmeans.fit(all_features.cpu().numpy())
    #print(kmeans.labels_)
    center_t1=torch.tensor(kmeans.cluster_centers_[0]).unsqueeze(0).to(device)
    center_t2=torch.tensor(kmeans.cluster_centers_[1]).unsqueeze(0).to(device)
    if label=='positive':
        if torch.mean(torch.cdist(center_t1,features[1][indices])[0])>torch.mean(torch.cdist(center_t2,features[1][indices])[0]):
            center_p=center_t2
            center_n=center_t1
        else:
            center_p=center_t1
            center_n=center_t2
        cluster_indices = torch.where(torch.cdist(center_n,features[1][indices])[0] >= torch.cdist(center_p,features[1][indices])[0])
        clus_center_indices = indices[cluster_indices[0]]
    elif label=='negative':
        if torch.mean(torch.cdist(center_t1,features[1][indices])[0])<torch.mean(torch.cdist(center_t2,features[1][indices])[0]):
            center_p=center_t2
            center_n=center_t1
        else:
            center_p=center_t1
            center_n=center_t2
        cluster_indices = torch.where(torch.cdist(center_n,features[1][indices])[0] <= torch.cdist(center_p,features[1][indices])[0])
        clus_center_indices = indices[cluster_indices[0]]

    return clus_center_indices


def self_matching(features,filtered_min_indices):
    self_distances = torch.cdist(features[1][filtered_min_indices], features[1][filtered_min_indices])
    mean_distance = torch.mean(self_distances)
    row_mean_distance = torch.mean(self_distances, dim=1)

    close_patch_indices = torch.where(row_mean_distance <= mean_distance)
    filtered_min_indices = filtered_min_indices[close_patch_indices[0]]

    return filtered_min_indices


if __name__ == "__main__":

    print('---------Start generating the initial prompting scheme---------')
    image_prompt_dir = '../data/reference_image' # 'img_prompt'
    mask_path = '../data/reference_mask'   # 'mask_prompt'
    image_dir = '../data/target_image'  # 'test'
    output_file = '../data/prompt_initial.txt'
    output_file_hard='../data/prompt_hard.txt'

    mask_files = os.listdir(mask_path)
    mask = os.path.join(mask_path, mask_files[0])

    image_files = os.listdir(image_dir)
    image_prompt_files = os.listdir(image_prompt_dir)

    mask_files = os.listdir(mask_path)
    mask = Image.open(os.path.join(mask_path, mask_files[0]))
    mask = mask.resize((560, 560))  
    mask = np.array(mask)  # standardization
    mask_np = np.repeat(mask[:, :, np.newaxis], 3, axis=2)  # Expanded to 3 channels

    fore_patchs,fore_index, back_patchs,back_index= find_foreground_patches(mask_np)
    img_prompt = Image.open(os.path.join(image_prompt_dir,image_prompt_files[0]))
    img_prompt = img_prompt.resize((560, 560))



    dino = hubconf.dinov2_vitg14()
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    dino.to(device)




    # Open the output file to write data
    with open(output_file, 'w') as file:
        for image_file in tqdm(image_files):
            image_paths = [os.path.join(image_prompt_dir, image_prompt_files[0]), os.path.join(image_dir, image_file)]
            image_paths_b = [os.path.join(image_prompt_dir, image_prompt_files[0]), os.path.join(image_dir, image_file)]

            #foward matching
            images_inner,features, min_indices,max_indices=forward_matching(image_paths,fore_index)
            images_inner_back,features_back, min_indices_back,max_indices_back=forward_matching(image_paths,back_index)

            #backward matching
            ng_indices,filtered_min_indices,re_min_indices=backward_matching(features,fore_index,min_indices)
            ng_indices_back,filtered_min_indices_back,re_min_indices_back=backward_matching(features,back_index,min_indices_back)

            #negative sample
            ng_indices=negative_points(max_indices,fore_index,re_min_indices)
            ng_indices_back=negative_points(max_indices_back,back_index,re_min_indices_back)


            
            #self matching
            filtered_min_indices=self_matching(features,filtered_min_indices)
            filtered_min_indices_back=self_matching(features,filtered_min_indices_back)
            
            filtered_min_indices_back=torch.cat([ng_indices,filtered_min_indices_back],dim=0)
            #cluster matching
            all_indices=torch.cat([filtered_min_indices,filtered_min_indices_back],dim=0)
            
            #clus_indices=cluster(features,features[1][all_indices],filtered_min_indices_back,'negative')
            #final_neg_points=calculate_center_points(clus_indices)


            final_pos_points=calculate_center_points(min_indices)
            final_neg_points=calculate_center_points(min_indices_back)

            #fin_center_points,max_center_points=process_image_pair(image_paths, fore_index,output_dir)

            # Convert points to tuples and remove duplicate points
            final_pos_points = set(tuple(point) for point in final_pos_points)
            final_neg_points = set(tuple(point) for point in final_neg_points)
            image = Image.open(image_paths[1])
            final_pos_points_map = map_to_ori_size(list(final_pos_points), [image.size[1],image.size[0]])
            final_neg_points_map = map_to_ori_size(list(final_neg_points), [image.size[1],image.size[0]])

            # Write the image file name and corresponding fin_center_points_map to the file
            file.write(f"{os.path.basename(image_paths[1])};{final_pos_points_map};{final_neg_points_map}\n")


    # Open the output file of hard sampling to write data
    with open(output_file_hard, 'w') as file:
        for image_file in tqdm(image_files):
            image_paths = [os.path.join(image_prompt_dir, image_prompt_files[0]), os.path.join(image_dir, image_file)]
            image_paths_b = [os.path.join(image_prompt_dir, image_prompt_files[0]), os.path.join(image_dir, image_file)]

            #foward matching
            images_inner,features, min_indices,max_indices=forward_matching(image_paths,fore_index)
            images_inner_back,features_back, min_indices_back,max_indices_back=forward_matching(image_paths,back_index)

            #backward matching
            ng_indices,filtered_min_indices,re_min_indices=backward_matching(features,fore_index,min_indices)
            ng_indices_back,filtered_min_indices_back,re_min_indices_back=backward_matching(features,back_index,min_indices_back)

            #negative sample
            ng_indices=negative_points(max_indices,fore_index,re_min_indices)
            ng_indices_back=negative_points(max_indices_back,back_index,re_min_indices_back)


            
            #self matching
            filtered_min_indices=self_matching(features,filtered_min_indices)
            filtered_min_indices_back=self_matching(features,filtered_min_indices_back)
            
            filtered_min_indices_back=torch.cat([ng_indices,filtered_min_indices_back],dim=0)
            #cluster matching
            all_indices=torch.cat([filtered_min_indices,filtered_min_indices_back],dim=0)
            
            #clus_indices=cluster(features,features[1][all_indices],filtered_min_indices_back,'negative')
            #final_neg_points=calculate_center_points(clus_indices)


            final_pos_points=calculate_center_points(min_indices)
            final_neg_points=calculate_center_points(ng_indices)

            #fin_center_points,max_center_points=process_image_pair(image_paths, fore_index,output_dir)

            # Convert points to tuples and remove duplicate points
            final_pos_points = set(tuple(point) for point in final_pos_points)
            final_neg_points = set(tuple(point) for point in final_neg_points)
            image = Image.open(image_paths[1])
            final_pos_points_map = map_to_ori_size(list(final_pos_points), [image.size[1],image.size[0]])
            final_neg_points_map = map_to_ori_size(list(final_neg_points), [image.size[1],image.size[0]])

            # Write the image file name and corresponding fin_center_points_map to the file
            file.write(f"{os.path.basename(image_paths[1])};{final_pos_points_map};{final_neg_points_map}\n")
