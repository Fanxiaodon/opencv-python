import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_thre(hist):
    """
    大津算法计算分割阈值
    :param hist: 图像直方图
    :return: 图像分割阈值
    """
    delta_max = 0
    threshold = -1
    for i in range(256):
        w0 = sum(hist[:i, 0]) / sum(hist[:, 0])
        if w0 == 0:
            continue
        if w0 == 1:
            break
        w1 = 1 - w0
        hist_pixel = [x * hist[x, 0] for x in range(256)]
        u0 = sum(hist_pixel[:i]) / sum(hist[:i, 0])
        u1 = sum(hist_pixel[i:]) / sum(hist[i:, 0])
        delta = w0 * w1 * (u1 - u0) * (u1 - u0)
        if delta > delta_max:
            delta_max = delta
            threshold = i
    return threshold


def cut_img(img, threshold):
    """
    根据阈值分割图像，大于阈值置255，小于阈值置0
    :param img: 代分割图像
    :param threshold: 分割阈值
    :return:
    """
    img_thre = np.zeros(img.shape, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > threshold:
                img_thre[i, j] = 255
            else:
                img_thre[i, j] = 0
    return img_thre


if __name__ == "__main__":
    img = cv2.imread('cells.bmp', cv2.IMREAD_GRAYSCALE)
    # cv2.calcHist()计算图像直方图，得到256*1的数组
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    threshold = get_thre(hist)
    img_thre = cut_img(img, threshold)

    plt.subplot(221)
    plt.imshow(img, cmap='gray')
    plt.xticks([]), plt.yticks([])

    plt.subplot(222)
    plt.plot(hist, color='b')
    plt.axvline(threshold, ymax=4000, color='r')

    plt.subplot(223)
    plt.imshow(img_thre, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()