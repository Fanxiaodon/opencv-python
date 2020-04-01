import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def conv(img, filter):
    img_out = np.zeros(img.shape, np.float)
    height, width = img.shape
    h, w = filter.shape
    # 周围补零操作
    img_padding = np.pad(img, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
    for i in range(height):
        for j in range(width):
            img_out[i][j] = np.sum(filter * (img_padding[i:i+h, j:j+w]))
    return img_out.clip(0, 255)


def merge_xy(img_x, img_y, threshold=100):
    img_grad = np.zeros(img_x.shape, np.float)
    img_out = np.zeros(img_x.shape, np.float)
    for i in range(img_x.shape[0]):
        for j in range(img_x.shape[1]):
            img_grad[i][j] = math.sqrt((img_x[i][j]) ** 2 + (img_y[i][j]) ** 2)
            if img_grad[i][j] > threshold:
                img_out[i][j] = 0
            else:
                img_out[i][j] = 255
    return img_out, img_grad


if __name__ == '__main__':
    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.transpose(sobel_x[:, ::-1])

    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.transpose(prewitt_x[:, ::-1])

    laplace = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    img_sobel_x = conv(img, sobel_x)
    img_sobel_y = conv(img, sobel_y)
    img_sobel, img_sobel_grad = merge_xy(img_sobel_x, img_sobel_y, 80)

    img_prewitt_x = conv(img, prewitt_x)
    img_prewitt_y = conv(img, prewitt_y)
    img_prewitt, img_prewitt_grad = merge_xy(img_prewitt_x, img_prewitt_y, 80)

    img_laplace = conv(img, laplace)
    img_laplace, img_laplace_grad = merge_xy(img_laplace, np.zeros(img_laplace.shape), 20)

    plt.subplot(231)
    plt.imshow(img_sobel_grad, cmap='gray')
    plt.title("Sobel gradient")

    plt.subplot(232)
    plt.imshow(img_prewitt_grad, cmap='gray')
    plt.title("Prewitt gradient")

    plt.subplot(233)
    plt.imshow(img_laplace_grad, cmap='gray')
    plt.title("Laplace gradient")

    plt.subplot(234)
    plt.imshow(img_sobel, cmap='gray')
    plt.title("Sobel result after choose a threshold")

    plt.subplot(235)
    plt.imshow(img_prewitt, cmap='gray')
    plt.title("Prewitt result after choose a threshold")

    plt.subplot(236)
    plt.imshow(img_laplace, cmap='gray')
    plt.title("Laplace result after choose a threshold")

    plt.show()
