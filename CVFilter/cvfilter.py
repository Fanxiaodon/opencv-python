import cv2
import numpy as np
import random
from matplotlib import pyplot as plt


def add_salt_pepper(img, prob=0.02):
    """
    给图像加入椒盐噪声
    :param img: 图像
    :param prob: 噪声比例
    :return: 噪声图像
    """
    output = np.zeros(img.shape, np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if random.random() > 1 - prob:
                output[i, j] = 255
            elif random.random() < prob:
                output[i, j] = 0
            else:
                output[i, j] = img[i, j]
    return output


def add_impulse(img, prob=0.02):
    """
    给图像加入脉冲噪声
    :param img: 图像
    :param prob: 噪声比例
    :return: 噪声图像
    """
    output = np.zeros(img.shape, np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if random.random() > 1 - prob:
                output[i, j] = 255
            else:
                output[i, j] = img[i, j]
    return output


def add_gauss(img, mean=0, var=0.001):
    """
    给图像加入高斯噪声
    :param img: 图像
    :param mean: 噪声均值
    :param var: 噪声方差
    :return: 噪声图像
    """
    image = np.array(img / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    output = image + noise
    if output.min() < 0:
        low_clip = -1.0
    else:
        low_clip = 0.0
    output = np.clip(output, low_clip, 1.0)
    output = np.uint8(output * 255)
    return output


if __name__ == "__main__":
    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
    img_salt = add_salt_pepper(img, 0.02)
    img_pulse = add_impulse(img, 0.02)
    img_gauss = add_gauss(img, var=0.001)
    img_aver = cv2.blur(img_salt, (5, 5))
    img_media = cv2.medianBlur(img_salt, 3)

    plt.subplot(231)
    plt.imshow(img, cmap='gray'), plt.title('Origin')
    plt.xticks([]), plt.yticks([])

    plt.subplot(232)
    plt.imshow(img_salt, cmap='gray'), plt.title('Salt & Pepper')
    plt.xticks([]), plt.yticks([])

    plt.subplot(233)
    plt.imshow(img_pulse, cmap='gray'), plt.title('Impulse')
    plt.xticks([]), plt.yticks([])

    plt.subplot(234)
    plt.imshow(img_gauss, cmap='gray'), plt.title('Gauss noise')
    plt.xticks([]), plt.yticks([])

    plt.subplot(235)
    plt.imshow(img_aver, cmap='gray'), plt.title('average')
    plt.xticks([]), plt.yticks([])

    plt.subplot(236)
    plt.imshow(img_media, cmap='gray'), plt.title('median')
    plt.xticks([]), plt.yticks([])

    plt.show()
