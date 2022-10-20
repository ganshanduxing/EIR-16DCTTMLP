## laod plain-images and secret keys
import numpy as np
import scipy.io as scio
from encryption_utils import ksa
from encryption_utils import prga
from encryption_utils import yates_shuffle
import tqdm
from encryption_utils import loadImageSet, loadImageFiles
from JPEG.rgbandycbcr import rgb2ycbcr
import cv2
import copy
from JPEG.jdcencColor import jdcencColor
from JPEG.zigzag import zigzag
from JPEG.invzigzag import invzigzag
from JPEG.jacencColor import jacencColor
from JPEG.Quantization import *
from cipherimageRgbGenerate import Gen_cipher_images
import hashlib


def coe_distribution_Same_Posit(d, key):
    [s, s] = d.shape
    n = int((s * s) / (8 * 8))
    block8 = np.zeros([int(8 * 8), n])
    b8number = [0] * n
    x = zigzag(d)
    order_cases = np.array([[1, 2, 3, 4],
                            [1, 2, 4, 3],
                            [1, 3, 2, 4],
                            [1, 3, 4, 2],
                            [1, 4, 2, 3],
                            [1, 4, 3, 2],
                            [2, 1, 3, 4],
                            [2, 1, 4, 3],
                            [2, 3, 1, 4],
                            [2, 3, 4, 1],
                            [2, 4, 3, 1],
                            [2, 4, 1, 3],
                            [3, 1, 2, 4],
                            [3, 1, 4, 2],
                            [3, 2, 1, 4],
                            [3, 2, 4, 1],
                            [3, 4, 1, 2],
                            [3, 4, 2, 1],
                            [4, 1, 2, 3],
                            [4, 1, 3, 2],
                            [4, 2, 1, 3],
                            [4, 2, 3, 1],
                            [4, 3, 1, 2],
                            [4, 3, 2, 1]]) - 1
    coeindex = 0
    while coeindex < int(s * s):
        b8index_per = order_cases[int('0b' + key[0:7], 2) % 24]
        for i in range(0, n):
            block8[b8number[b8index_per[i]], i] = x[coeindex]
            b8number[b8index_per[i]] = b8number[b8index_per[i]] + 1
            coeindex = coeindex + 1
        key = key[2:]
    b_count = 0
    for r in range(0, s, 8):
        for c in range(0, s, 8):
            d[r:r + 8, c:c + 8] = invzigzag(block8[:, b_count], 8, 8)
            b_count = b_count + 1
    return d


def encryption_each_component(image_component, keys, type, row, col, N, QF):
    # generate block permutation vector
    block8_number = int((row * col) / (8 * 8))
    data = [i for i in range(0, block8_number)]
    p_blockY = yates_shuffle(data, keys)
    keys = keys[64:]

    allblock8 = np.zeros([8, 8, int(row * col / (8 * 8))])
    allblock8_number = 0
    for m in range(0, row, N):
        for n in range(0, col, N):
            t = image_component[m:m + N, n:n + N] - 128
            y = cv2.dct(t)  # DCT
            y = coe_distribution_Same_Posit(y, keys)
            keys = keys[2:]
            for i in range(0, N, 8):
                for j in range(0, N, 8):
                    temp = Quantization(y[i:i + 8, j:j + 8], QF, type=type)  # Quanlity
                    allblock8[:, :, allblock8_number] = temp
                    allblock8_number = allblock8_number + 1

    # block permutation
    permuted_blocks = copy.copy(allblock8)
    for i in range(len(p_blockY)):
         permuted_blocks[:, :, i] = allblock8[:, :, p_blockY[i]]

    # Huffman coding
    dccof = []
    accof = []
    for i in range(0, allblock8_number):
        temp = copy.copy(permuted_blocks[:, :, i])
        if i == 0:
            dc = temp[0, 0]
            dc_component = jdcencColor(dc, type)
            dccof = np.append(dccof, dc_component)
        else:
            dc = temp[0, 0] - dc
            dc_component = jdcencColor(dc, type)
            dccof = np.append(dccof, dc_component)
            dc = temp[0, 0]
        acseq = []
        aczigzag = zigzag(temp)
        eobi = 0
        for j in range(63, -1, -1):
            if aczigzag[j] != 0:
                eobi = j
                break
        if eobi == 0:
            acseq = np.append(acseq, [999])
        else:
            acseq = np.append(acseq, aczigzag[1: eobi + 1])
            acseq = np.append(acseq, [999])
        ac_component = jacencColor(acseq, type)
        accof = np.append(accof, ac_component)

    return dccof, accof


def encryption(img, keyY, keyCb, keyCr, QF, N=8):
    # N: block size
    # QF: quality factor
    row, col, _ = img.shape
    plainimage = rgb2ycbcr(img)
    plainimage = plainimage.astype(np.float64)
    Y = plainimage[:, :, 0]
    Cb = plainimage[:, :, 1]
    Cr = plainimage[:, :, 2]

    for i in range(0, int(32 * np.ceil(col / 32) - col)):
        Y = np.c_[Y, Y[:, -1]]
        Cb = np.c_[Cb, Cb[:, -1]]
        Cr = np.c_[Cr, Cr[:, -1]]

    for i in range(0, int(32 * np.ceil(row / 32) - row)):
        Y = np.r_[Y, [Y[-1, :]]]
        Cb = np.r_[Cb, [Cb[-1, :]]]
        Cr = np.r_[Cr, [Cr[-1, :]]]

    [row, col] = Y.shape

    Cb = cv2.resize(Cb,
                    (int(col / 2), int(row / 2)),
                    interpolation=cv2.INTER_CUBIC)
    Cr = cv2.resize(Cr,
                    (int(col / 2), int(row / 2)),
                    interpolation=cv2.INTER_CUBIC)

    # Y component
    dccofY, accofY = encryption_each_component(Y, keyY, type='Y', row=row, col=col, N=N, QF=QF)
    ## Cb and Cr component
    dccofCb, accofCb = encryption_each_component(Cb, keyCb, type='Cb', row=int(row / 2), col=int(col / 2), N=N, QF=QF)
    dccofCr, accofCr = encryption_each_component(Cr, keyCr, type='Cr', row=int(row / 2), col=int(col / 2), N=N, QF=QF)

    accofY = accofY.astype(np.int8)
    dccofY = dccofY.astype(np.int8)
    accofCb = accofCb.astype(np.int8)
    dccofCb = dccofCb.astype(np.int8)
    accofCr = accofCr.astype(np.int8)
    dccofCr = dccofCr.astype(np.int8)
    return accofY, dccofY, accofCb, dccofCb, accofCr, dccofCr


# read plain-images
def read_plain_images():
    plainimage_all = loadImageSet('../data/plainimages/*.jpg')
    # save size information
    img_size = []
    for temp in plainimage_all:
        row, col, _ = temp.shape
        img_size.append((row, col))
    np.save("../data/plainimages.npy", plainimage_all)
    np.save("../data/img_size.npy", img_size)
    return plainimage_all


# generate encryption key and embedding key
# keys are independent from plainimage
# encryption key generation - RC4
hash = hashlib.sha256()


def generate_hash(inp):
    hash.update(bytes(str(inp), encoding='utf-8'))
    res = hash.hexdigest()
    hash_list = []
    for i in range(0, len(res), 2):
        hash_list.append(int(res[i:i + 2], 16))
    return hash_list


def generate_keys(img, control_length=256 * 284):
    # secret keys
    data_lenY = np.ones([1, int(control_length)])

    keyY = generate_hash(img[:, :, 0])
    keyCb = generate_hash(img[:, :, 0])
    keyCr = generate_hash(img[:, :, 0])

    # keys stream
    s = ksa(keyY)
    r = prga(s, data_lenY)
    encryption_keyY = ''
    for i in range(0, len(r)):
        temp1 = str(r[i])
        temp2 = bin(int(temp1, 10))
        temp2 = temp2[2:]
        for j in range(0, 8 - len(temp2)):
            temp2 = '0' + temp2
        encryption_keyY = encryption_keyY + temp2

    data_lenC = np.ones([1, int(control_length // 4)])
    s1 = ksa(keyCb)
    r1 = prga(s1, data_lenC)
    encryption_keyCb = ''
    for i in range(0, len(r1)):
        temp1 = str(r1[i])
        temp2 = bin(int(temp1, 10))
        temp2 = temp2[2:]
        for j in range(0, 8 - len(temp2)):
            temp2 = '0' + temp2
        encryption_keyCb = encryption_keyCb + temp2

    s2 = ksa(keyCr)
    r2 = prga(s2, data_lenC)
    encryption_keyCr = ''
    for i in range(0, len(r2)):
        temp1 = str(r2[i])
        temp2 = bin(int(temp1, 10))
        temp2 = temp2[2:]
        for j in range(0, 8 - len(temp2)):
            temp2 = '0' + temp2
        encryption_keyCr = encryption_keyCr + temp2

    return encryption_keyY, encryption_keyCb, encryption_keyCr


if __name__ == '__main__':
    # image encryption
    QF = 90
    plainimage_all = read_plain_images()
    num = len(plainimage_all)  # test images num
    del plainimage_all

    imageFiles = loadImageFiles('../data/plainimages/*.jpg')[:num]

    for k in tqdm.tqdm([i for i in range(len(imageFiles))]):
        # read plain-image
        img = cv2.imread(imageFiles[k])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encryption_keyY, encryption_keyCb, encryption_keyCr = generate_keys(img)
        accofY, dccofY, accofCb, dccofCb, accofCr, dccofCr = encryption(img, encryption_keyY, encryption_keyCb,
                                                                        encryption_keyCr, QF, N=16)

        np.save(f'../data/JPEGBitStream/YAC/acallY_{k}.npy', accofY)
        np.save(f'../data/JPEGBitStream/YDC/dcallY_{k}.npy', dccofY)
        np.save(f'../data/JPEGBitStream/CbAC/acallCb_{k}.npy', accofCb)
        np.save(f'../data/JPEGBitStream/CbDC/dcallCb_{k}.npy', dccofCb)
        np.save(f'../data/JPEGBitStream/CrAC/acallCr_{k}.npy', accofCr)
        np.save(f'../data/JPEGBitStream/CrDC/dcallCr_{k}.npy', dccofCr)

    # generate cipher-images
    Gen_cipher_images(QF=QF, Image_num=len(imageFiles))
