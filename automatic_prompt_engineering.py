import torch
import torch.nn.functional as F
import math
import numpy as np

def euc_distance(point1, point2):
    """
    Computes the Euclidean distance between two points.
    
    Args:
        point1 (list or ndarray): Coordinates of the first point.
        point2 (list or ndarray): Coordinates of the second point.
        
    Returns:
        float: The Euclidean distance between the two points.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))


def refine_prompt(radius, prompt_all, prompt_other):
    """
    Refines the list of prompts by considering the proximity of points within a given radius.

    Args:
        radius (float): The radius within which points are considered similar.
        prompt_all (list of list): A list of all prompt points.
        prompt_other (list of list): A list of other prompt points to be compared with.

    Returns:
        list: A list of refined prompts based on distance criteria.
    """
    judge_prompt = set()  # Use set for faster membership checking
    final_prompt = []

    for prompt in prompt_all:
        if tuple(prompt) not in judge_prompt:
            list_prompt = [prompt]
            for prompt_t in prompt_all:
                if tuple(prompt_t) not in judge_prompt:
                    dis = euc_distance(prompt, prompt_t)
                    if dis <= radius and prompt_t != prompt:
                        judge_prompt.add(tuple(prompt_t))  # Add to judge set to avoid duplicates
                        if prompt in list_prompt:
                            list_prompt.remove(prompt)
                        list_prompt.append(prompt_t)
                        list_prompt.append(prompt)
            
            # Find closest prompt from 'prompt_other' for each in list_prompt
            temp_name = []
            temp_min_dis = []
            for name in list_prompt:
                distances = [euc_distance(name, name_other) for name_other in prompt_other]
                temp_min_dis.append(np.mean(distances))
                temp_name.append(name)

            # Map distances to the prompt names and select the one with the max distance
            temp_dic = dict(zip(temp_min_dis, temp_name))
            temp = temp_dic[max(temp_min_dis)]
            final_prompt.append(temp)
    
    return final_prompt


def refine_prompt_converse(prompt_all, prompt_other, radius_pn):
    """
    Filters out prompts from 'prompt_all' that are too close to any points in 'prompt_other'.
    
    Args:
        prompt_all (list of list): The list of all prompts.
        prompt_other (list of list): The list of other prompts to compare distances with.
        radius_pn (float): The minimum distance threshold.
        
    Returns:
        list: A list of refined prompts that are sufficiently far from 'prompt_other'.
    """
    final_prompt = []
    for prompt in prompt_all:
        distances = [euc_distance(prompt, prompt_o) for prompt_o in prompt_other]
        if min(distances) >= radius_pn:
            final_prompt.append(prompt)

    return final_prompt


def APE(positive_prompt, negative_prompt, hard_prompt, window_size, window_size_n, window_size_hard, min_radius, hard_sample):
    """
    The APE function refines positive and negative prompts based on certain criteria.
    It integrates hard negatives if needed and applies distance thresholds.

    Args:
        positive_prompt (list of list): The positive prompts.
        negative_prompt (list of list): The negative prompts.
        hard_prompt (list of list): The hard negative prompts.
        window_size (float): The radius for refining positive prompts.
        window_size_n (float): The radius for refining negative prompts.
        window_size_hard (float): The radius for refining hard negatives.
        min_radius (float): Minimum distance threshold for filtering negative prompts.
        hard_sample (bool): Whether to include hard negatives in the final list.

    Returns:
        tuple: Refined positive and negative prompts.
    """
    # Refine negative hard prompts
    negative_hard = refine_prompt(window_size_hard, hard_prompt, positive_prompt)
    
    # Refine final negative and positive prompts
    negative_final = refine_prompt_converse(negative_prompt, positive_prompt, min_radius)
    positive_final = refine_prompt(window_size, positive_prompt, negative_prompt)
    negative_final = refine_prompt(window_size_n, negative_final, positive_final)

    # Add hard negatives if required
    if hard_sample:
        negative_final.extend(negative_hard)
    
    # Ensure that there is at least one negative and one positive prompt
    if not negative_final:
        negative_final.append([10, 10])  # Default negative point
    if not positive_final:
        positive_final.append([250, 250])  # Default positive point
    
    return positive_final, negative_final
