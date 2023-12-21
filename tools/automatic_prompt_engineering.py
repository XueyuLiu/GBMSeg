import os
from tqdm import tqdm
import random
import math
import numpy as np

def euc_distance(point1,point2):
    distance=0.0
    for i in range(len(point1)):
        distance+=(point1[i]-point2[i])**2
    return math.sqrt(distance)


def refine_prompt(radius,prompt_all,prompt_other):
    judge_prompt=[]
    final_prompt=[]
    prompt_all_list=tuple(eval(prompt_all))
    prompt_other_list=tuple(eval(prompt_other))
    delate_list=[]
    for prompt in prompt_all_list:
        if tuple(prompt) not in judge_prompt:
            list_prompt=[]
            list_prompt.append(prompt)
            for prompt_t in prompt_all_list:
                if tuple(prompt_t) not in judge_prompt:
                    dis=euc_distance(prompt,prompt_t)
                    if dis<=radius and prompt_t!=prompt:
                        judge_prompt.append(prompt_t)
                        if prompt in list_prompt:
                            list_prompt.remove(prompt)
                        list_prompt.append(prompt_t)
                        list_prompt.append(prompt)
            judge_prompt.append(prompt)
            temp_name=[]
            temp_min_dis=[]
            for name in list_prompt:
                temp_dis=[]
                temp_name.append(name)
                for name_other in prompt_other_list:
                    temp_dis.append(euc_distance(name,name_other))
                temp_min_dis.append(np.mean(temp_dis))
            temp_dic=dict(zip(temp_min_dis,temp_name))
            temp=temp_dic[max(temp_min_dis)]
            final_prompt.append(temp)
            judge_prompt=set(tuple(point) for point in judge_prompt)
            judge_prompt=list(judge_prompt)

    return final_prompt
    

def refine_prompt_converse(prompt_all,prompt_other,radius_pn):
    final_prompt=[]
    prompt_all_list=tuple(prompt_all)
    prompt_other_list=tuple(prompt_other)
    for prompt in prompt_all_list:
        distance=[]
        for prompt_other in prompt_other_list:
            distance.append(euc_distance(prompt,prompt_other))
        if min(distance)>=radius_pn:
           final_prompt.append(prompt)

    return final_prompt


if __name__ == "__main__":
    path_prompt='../data/prompt_initial.txt'
    path_prompt_n='../data/prompt_hard.txt'
    output_file='../data/prompt_final.txt'

    window_size=0  #Search radius for positive prompt point discrate sampling
    window_size_n=64     #Search radius for negativa prompt point discrate sampling
    min_radius=128    #Search radius for positive prompt point independent sampling

    print('---------Start automatic prompt engineering---------')
    with open(path_prompt, 'r') as file:
        lines = file.readlines()
        with open(path_prompt_n,'r') as file_n:
            lines_n=file_n.readlines()
            with open(output_file, 'w') as file2:
                for line in tqdm(lines):
                    for line_n in lines_n:
                        parts = line.strip().split(';')
                        parts_n = line_n.strip().split(';')
                        if parts_n[0]==parts[0]:
                            negative_hard=refine_prompt(window_size_n,parts_n[2],parts_n[1])
                            positive_final=refine_prompt(window_size,parts[1],parts[2])    
                            negative_final=refine_prompt(window_size_n,parts[2],parts[1])
                            negative_final=refine_prompt_converse(negative_final,positive_final,min_radius)
                            #negative_all=[]
                            #print(len(negative_final))
                            negative_final.extend(negative_hard)
                            #print(len(negative_final))
                            if len(positive_final)==0:
                                print(parts[0])
                            file2.write(f"{parts[0]};{positive_final};{negative_final}\n")


