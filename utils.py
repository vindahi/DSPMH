import os
import random
import numpy as np
import torch
import logging
import os.path as osp

def seed_setting(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def zero2eps(x):
    x[x == 0] = 1
    return x

def normalize(affinity):
    col_sum = zero2eps(np.sum(affinity, axis=1)[:, np.newaxis])
    row_sum = zero2eps(np.sum(affinity, axis=0))
    out_affnty = affinity/col_sum
    in_affnty = np.transpose(affinity/row_sum)
    return in_affnty, out_affnty

def affinity_tag_multi(tag1: np.ndarray, tag2: np.ndarray):
    aff = np.matmul(tag1, tag2.T)
    affinity_matrix = np.float32(aff)
    affinity_matrix = 1 / (1 + np.exp(-affinity_matrix))
    affinity_matrix = 2 * affinity_matrix - 1
    in_aff, out_aff = normalize(affinity_matrix)

    return in_aff, out_aff, affinity_matrix

def calculate_map(qu_B, re_B, qu_L, re_L):
    num_query = qu_L.shape[0]
    map = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        count = np.linspace(1, tsum, tsum) # [1,2, tsum]
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query
    return map


def calculate_top_map(qu_B, re_B, qu_L, re_L, topk):
    num_query = qu_L.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, int(tsum))
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap



def calculate_hamming(B1, B2):
    leng = B2.shape[1] 
    distH = 0.5 * (leng - np.dot(B1, B2.transpose()))
    return distH


def logger(fileName='log'):
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)

    log_name = str(fileName) + '.txt'
    log_dir = './logs'
    txt_log = logging.FileHandler(osp.join(log_dir, log_name))
    txt_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    txt_log.setFormatter(formatter)
    logger.addHandler(txt_log)

    stream_log = logging.StreamHandler()
    stream_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_log.setFormatter(formatter)
    logger.addHandler(stream_log)

    return logger