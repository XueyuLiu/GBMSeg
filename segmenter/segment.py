import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm

def show_mask(mask, ax, random_color=False):
    """
    Displays a mask on the given axis, with an option for random or fixed color.

    Args:
        mask (ndarray): The binary mask to display.
        ax (matplotlib.axes.Axes): The axis to display the mask on.
        random_color (bool): Whether to use a random color for the mask.
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)  # Random color
    else:
        color = np.array([254/255, 215/255, 26/255, 0.8])  # Fixed color (yellowish)
    
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=350):
    """
    Displays positive and negative points on the given axis.

    Args:
        coords (ndarray): The coordinates of the points.
        labels (ndarray): The labels (1 for positive, 0 for negative).
        ax (matplotlib.axes.Axes): The axis to display the points on.
        marker_size (int): The size of the markers for the points.
    """
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='blue', marker='o', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='o', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    """
    Displays a bounding box on the given axis.

    Args:
        box (list): Coordinates of the box in the form [x0, y0, x1, y1].
        ax (matplotlib.axes.Axes): The axis to display the box on.
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def prepare_input(ps_points, ng_points):
    """
    Prepares the input points and labels for the SamPredictor.

    Args:
        ps_points (list): List of positive points.
        ng_points (list): List of negative points.

    Returns:
        tuple: (input_points, input_labels) where input_points is a stacked array of positive and negative points,
               and input_labels is an array of corresponding labels (1 for positive, 0 for negative).
    """
    if ps_points is not None and ng_points is not None:
        ps_points = np.array(ps_points)
        ng_points = np.array(ng_points)
        input_point = np.vstack((ps_points, ng_points))
        input_label = np.concatenate((np.ones(ps_points.shape[0]), np.zeros(ng_points.shape[0])))
    else:
        ps_points = np.array(ps_points)
        input_point = ps_points
        input_label = np.ones(ps_points.shape[0])

    return input_point, input_label

def save_max_contour_area(mask):
    """
    Finds and returns the largest contour in a binary mask.

    Args:
        mask (ndarray): Binary mask image.

    Returns:
        ndarray: Image with the largest contour area filled.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filled_image = np.zeros_like(mask)
    
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(filled_image, [max_contour], -1, 255, thickness=cv2.FILLED)
        
    return filled_image

def refine_mask(mask):
    """
    Refines the mask by keeping only contours that are at least 30% of the largest contour's area.

    Args:
        mask (ndarray): Binary mask image.

    Returns:
        ndarray: Refined mask with selected contours.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    largest_contour = contours[0]
    min_area = 0.3 * cv2.contourArea(largest_contour)
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]

    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)
    cv2.drawContours(contour_mask, filtered_contours, -1, 255, -1)

    return contour_mask

def process_image(image, ps_points, ng_points, sam, max_contour=False, boxes=None):
    """
    Processes an image with given points and SamPredictor to generate segmentation masks.

    Args:
        image (PIL.Image): The input image to process.
        ps_points (list): Positive points for the segmentation.
        ng_points (list): Negative points for the segmentation.
        sam (SamPredictor): The SamPredictor model to use for mask prediction.
        max_contour (bool): Whether to apply contour refinement.
        boxes (optional): Bounding boxes for the image.

    Returns:
        ndarray: The final mask after processing.
    """
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    input_point, input_label = prepare_input(ps_points, ng_points)

    # Predict the mask
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
        box=None
    )
    
    # Process each mask
    for i, mask in enumerate(masks):
        mask_image = (mask * 255).astype(np.uint8)
        if max_contour:
            mask_image = save_max_contour_area(mask_image)

        return mask_image  # Returning the processed mask

def loading_seg(model_type, device):
    """
    Loads the SAM model based on the specified type.

    Args:
        model_type (str): Type of the SAM model ('vitb', 'vitl', or 'vith').
        device (torch.device): Device to load the model on.

    Returns:
        sam (SamPredictor): The loaded SAM model.
    """
    sam_checkpoint = {
        'vit_b': "./segmenter/checkpoint/sam_vit_b_01ec64.pth",
        'vit_l': "./segmenter/checkpoint/sam_vit_l_0b3195.pth",
        'vit_h': "./segmenter/checkpoint/sam_vit_h_4b8939.pth"
    }[model_type]
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    return sam

def seg_main(image, pos_prompt, neg_prompt, device, max_contour=False, seg_model=None, boxes=None):
    """
    Main function to handle the segmentation task for a given image.

    Args:
        image (PIL.Image): The input image to segment.
        pos_prompt (list): Positive points for segmentation.
        neg_prompt (list): Negative points for segmentation.
        device (torch.device): Device for running the model.
        max_contour (bool): Whether to apply contour refinement.
        seg_model (SamPredictor): The SAM model to use for segmentation.
        boxes (optional): Bounding boxes for the image.

    Returns:
        ndarray: The final processed mask.
    """
    mask = process_image(image, pos_prompt, neg_prompt, seg_model, max_contour, boxes)
    return mask
