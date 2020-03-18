import cv2
import numpy as np
from matplotlib import pyplot as plt


def equalize_hist(img):
    """
    直方图均衡
    :param img: 图像
    :return: 均衡后图像
    """
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = cdf_m * 255/(cdf_m.max() - cdf_m.min())
    cdf_m = cdf_m.astype(np.uint8)
    return cdf_m[img]


if __name__ == '__main__':
    img = cv2.imread('fig1.jpg')
    img_equ = equalize_hist(img)

    plt.subplot(221), plt.imshow(img, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.title('Original image')

    plt.subplot(222), plt.imshow(img_equ, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.title('Hist equalize')

    plt.subplot(223), plt.hist(img.ravel(), 256, [0, 256])
    plt.xlim()
    plt.title('Hist before')

    plt.subplot(224), plt.hist(img_equ.ravel(), 256, [0, 256])
    plt.title('Hist after')

    plt.show()
