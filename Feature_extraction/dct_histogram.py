## rgb cipherimage retrieval - server
import numpy as np
import pandas as pd

from Encryption_algorithm.JPEG.jacdecColorHuffman import jacdecColor
from Encryption_algorithm.JPEG.jdcdecColorHuffman import jdcdecColor
from Encryption_algorithm.JPEG.Quantization import *
from Encryption_algorithm.JPEG.invzigzag import invzigzag
from Encryption_algorithm.JPEG.zigzag import zigzag
from Encryption_algorithm.encryption_utils import loadEncBit, loadImgSizes
import tqdm

# 64 histograms for each cipherimage
DC_bin_interval = [i for i in range(-2080, 2080, 64)]
AC_bin_interval = [i for i in range(-2080, 2080, 64)]
DC_histgram_dimension = len(DC_bin_interval) - 1
AC_histgram_dimension = len(AC_bin_interval) - 1


def extract_feature(dc, ac, size, type, QF, N=8):
    _, acarr = jacdecColor(ac, type)
    _, dcarr = jdcdecColor(dc, type)
    acarr = np.array(acarr)
    dcarr = np.array(dcarr)

    if type == 'Y':
        row, col = size
        row = int(32 * np.ceil(row / 32))
        col = int(32 * np.ceil(col / 32))
    else:
        row, col = size
        row = int(16 * np.ceil(row / 32))
        col = int(16 * np.ceil(col / 32))

    Eob = np.where(acarr == 999)
    Eob = Eob[0]
    count = 0
    kk = 0
    ind1 = 0
    allblock8 = np.zeros([8, 8, int(row * col / (8 * 8))])
    allblock8_number = 0
    for m in range(0, row, N):
        for n in range(0, col, N):
            ac = acarr[ind1: Eob[count]]
            ind1 = Eob[count] + 1
            count = count + 1
            acc = np.append(dcarr[kk], ac)
            az = np.zeros(64 - acc.shape[0])
            acc = np.append(acc, az)
            temp = invzigzag(acc, 8, 8)
            temp = iQuantization(temp, QF, type)
            allblock8[:, :, allblock8_number] = temp
            kk = kk + 1
            allblock8_number = allblock8_number + 1

    allcoe = np.zeros([allblock8.shape[2], 64])
    for j in range(0, allblock8.shape[2]):
        temp = allblock8[:, :, j]
        allcoe[j, :] = zigzag(temp)

    hist_img = np.zeros(AC_histgram_dimension * 63 + DC_histgram_dimension)
    for j in range(0, 64):
        if j == 0:
            hist_tmp = np.zeros([1, DC_histgram_dimension])
            tmp = allcoe[:, j]
            hist_t = np.histogram(tmp, bins=DC_bin_interval)
            hist_t = hist_t[0]
            hist_tmp[0, :] = hist_t
            hist_tmp = hist_tmp.T
            hist_norm = hist_tmp / np.sum(hist_tmp)
            hist_img[0:DC_histgram_dimension] = hist_norm[:, 0]
        else:
            hist_tmp = np.zeros([1, AC_histgram_dimension])
            tmp = allcoe[:, j]
            hist_t = np.histogram(tmp, bins=AC_bin_interval)
            hist_t = hist_t[0]
            hist_tmp[0, :] = hist_t
            hist_tmp = hist_tmp.T
            hist_norm = hist_tmp / np.sum(hist_tmp)
            hist_img[DC_histgram_dimension + (
                    j - 1) * AC_histgram_dimension:DC_histgram_dimension + j * AC_histgram_dimension] = hist_norm[:,
                                                                                                        0]
    return hist_img


def extract_all_component_feature(QF=90):
    feature_save_path = '../data/features'
    img_size = loadImgSizes()
    image_num = len(img_size)
    hist64_rgb_Y = np.zeros([AC_histgram_dimension * 63 + DC_histgram_dimension, image_num])
    hist64_rgb_Cb = np.zeros([AC_histgram_dimension * 63 + DC_histgram_dimension, image_num])
    hist64_rgb_Cr = np.zeros([AC_histgram_dimension * 63 + DC_histgram_dimension, image_num])
    for k in tqdm.tqdm(range(image_num)):
        dcallY, acallY, dcallCb, acallCb, dcallCr, acallCr = loadEncBit('../data/JPEGBitStream', k)
        hist64_rgb_Y[:, k] = extract_feature(dcallY.astype(np.int8), acallY.astype(np.int8), img_size[k], "Y", QF)
        hist64_rgb_Cb[:, k] = extract_feature(dcallCb.astype(np.int8), acallCb.astype(np.int8), img_size[k], "C",
                                              QF)
        hist64_rgb_Cr[:, k] = extract_feature(dcallCr.astype(np.int8), acallCr.astype(np.int8), img_size[k], "C",
                                              QF)
    hist64_rgb_Y = hist64_rgb_Y.T.reshape([-1, 64, 64])
    hist64_rgb_Cb = hist64_rgb_Cb.T.reshape([-1, 64, 64])
    hist64_rgb_Cr = hist64_rgb_Cr.T.reshape([-1, 64, 64])
    features = np.concatenate([hist64_rgb_Y, hist64_rgb_Cb, hist64_rgb_Cr], axis=2).reshape(image_num, -1)
    data = pd.DataFrame(features)
    data.to_csv(feature_save_path + '/' + 'DCTHsitfeats.csv'.format(QF), index=False, header=0)
