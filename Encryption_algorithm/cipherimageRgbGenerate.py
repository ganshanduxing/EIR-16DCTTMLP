## generate cipherimage
import numpy as np
from JPEG.jacdecColorHuffman import jacdecColor
from JPEG.jdcdecColorHuffman import jdcdecColor
from JPEG.invzigzag import invzigzag
import cv2
from JPEG.rgbandycbcr import ycbcr2rgb, rgb2ycbcr
import glob
import tqdm
from JPEG.DCT import idctJPEG
from JPEG.Quantization import iQuantization
from encryption_utils import loadEncBit, loadImgSizes


def deEntropy(acall, dcall, row, col, type, N=8, QF=100):
    accof = acall
    dccof = dcall
    kk, acarr = jacdecColor(accof, type)
    kk, dcarr = jdcdecColor(dccof, type)
    acarr = np.array(acarr)
    dcarr = np.array(dcarr)

    Eob = np.where(acarr == 999)
    Eob = Eob[0]
    count = 0
    kk = 0
    ind1 = 0
    xq = np.zeros([row, col])
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
            temp = idctJPEG(temp)
            xq[m:m + N, n:n + N] = temp + 128
            kk = kk + 1
    return xq


def Gen_cipher_images(QF, Image_num=1):
    cipherimage_Y = []
    cipherimage_Cb = []
    cipherimage_Cr = []

    img_size = loadImgSizes()

    for k in tqdm.tqdm([i for i in range(Image_num)]):
        dcallY, acallY, dcallCb, acallCb, dcallCr, acallCr = loadEncBit('../data/JPEGBitStream', k)
        row, col = img_size[k]
        row = int(32 * np.ceil(row / 32))
        col = int(32 * np.ceil(col / 32))
        cipherimage_Y.append(deEntropy(acallY, dcallY, row, col, 'Y', QF=QF))
        cipherimage_Cb.append(deEntropy(acallCb, dcallCb, int(row / 2), int(col / 2), 'C', QF=QF))
        cipherimage_Cr.append(deEntropy(acallCr, dcallCr, int(row / 2), int(col / 2), 'C', QF=QF))

    np.save("../data/cipherimageNPYFiles/cipherimage_Y.npy", cipherimage_Y)
    np.save("../data/cipherimageNPYFiles/cipherimage_Cb.npy", cipherimage_Cb)
    np.save("../data/cipherimageNPYFiles/cipherimage_Cr.npy", cipherimage_Cr)

    srcFiles = glob.glob('../data/plainimages/*.jpg')
    cipherimage_all = []
    for k in tqdm.tqdm([i for i in range(Image_num)]):
        row, col = img_size[k]
        row = int(32 * np.ceil(row / 32))
        col = int(32 * np.ceil(col / 32))
        cipherimage = np.zeros([row, col, 3])
        cipher_Y = cipherimage_Y[k]
        cipher_cb = cipherimage_Cb[k]
        cipher_cr = cipherimage_Cr[k]
        cipher_cb = cv2.resize(cipher_cb,
                               (col, row),
                               interpolation=cv2.INTER_CUBIC)
        cipher_cr = cv2.resize(cipher_cr,
                               (col, row),
                               interpolation=cv2.INTER_CUBIC)
        cipherimage[:, :, 0] = cipher_Y
        cipherimage[:, :, 1] = cipher_cb
        cipherimage[:, :, 2] = cipher_cr
        cipherimage = np.round(cipherimage)
        cipherimage = cipherimage.astype(np.uint8)
        cipherimage = ycbcr2rgb(cipherimage)
        cipherimage_all.append(cipherimage)
        merged = cv2.merge([cipherimage[:, :, 2], cipherimage[:, :, 1], cipherimage[:, :, 0]])
        cv2.imwrite('../data/cipherimages/{}'.format(srcFiles[k].split("/")[-1].split("\\")[-1]), merged,
                    [int(cv2.IMWRITE_JPEG_QUALITY), QF])

    # print('{} pictures completed.'.format(k+1))
