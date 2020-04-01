import cv2
import numpy as np
import matplotlib.pyplot as plt


def conv(img, filter):
    """
    卷积运算
    :param img: 待卷积图像
    :param filter: 卷积核
    :return: 卷积后图像
    """
    img_out = np.zeros(img.shape, np.float)
    height, width = img.shape
    h, w = filter.shape
    # 周围补零操作
    img_padding = np.pad(img, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
    for i in range(height):
        for j in range(width):
            img_out[i][j] = np.sum(filter * (img_padding[i:i+h, j:j+w]))
    return img_out


def img_nms(img, grad_x, grad_y):
    """
    非极大值抑制，通过线性插值的方法求梯度方向的两个相邻梯度
    :param img: 输入图像
    :param grad_x: sobel算子得到的x方向梯度
    :param grad_y: sobel算子得到的y方向梯度
    :return:
    """
    h, w = img.shape
    grad = np.sqrt(np.square(grad_x) + np.square(grad_y))
    # print(grad)
    # 扩边，方便非极大值抑制
    grad_pad = np.pad(grad, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
    # 防止除0错误
    for i in range(grad_x.shape[0]):
        for j in range(grad_x.shape[1]):
            if grad_x[i][j] == 0:
                grad_x[i][j] = 1
    # 非极大值抑制，线性插值
    for i in range(h):
        for j in range(w):
            if abs(grad_x[i][j]) > abs(grad_y[i][j]):
                weight = abs(grad_y[i][j]) / abs(grad_x[i][j])
                if grad_y[i][j] * grad_x[i][j] >= 0:
                    grad_p1 = grad_pad[i+1][j+2]
                    grad_p2 = grad_pad[i][j+2]
                    grad_p3 = grad_pad[i+1][j]
                    grad_p4 = grad_pad[i+2][j]
                else:
                    grad_p1 = grad_pad[i+1][j+2]
                    grad_p2 = grad_pad[i+2][j+2]
                    grad_p3 = grad_pad[i+1][j]
                    grad_p4 = grad_pad[i][j]
            else:
                weight = abs(grad_x[i][j]) / abs(grad_y[i][j])
                if grad_y[i][j] * grad_x[i][j] >= 0:
                    grad_p1 = grad_pad[i][j+1]
                    grad_p2 = grad_pad[i][j+2]
                    grad_p3 = grad_pad[i+2][j+1]
                    grad_p4 = grad_pad[i+2][j]
                else:
                    grad_p1 = grad_pad[i][j+1]
                    grad_p2 = grad_pad[i][j]
                    grad_p3 = grad_pad[i+2][j+1]
                    grad_p4 = grad_pad[i+2][j+2]
            p1 = weight * grad_p1 + (1 - weight) * grad_p2
            p2 = weight * grad_p3 + (1 - weight) * grad_p4
            # 如果不是极大值，赋0
            if grad[i][j] < p1 or grad[i][j] < p2:
                grad[i][j] = 0

    # print(grad)
    return grad


def img_canny(img, threshold_l, threshold_h):
    # sobel算子计算梯度
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.transpose(sobel_x[:, ::-1])
    # 首先高斯平滑
    img_gass = cv2.GaussianBlur(img, (5, 5), 0)
    # plt.imshow(img_gass, cmap='gray')
    img_sobel_x = conv(img_gass, sobel_x)
    img_sobel_y = conv(img_gass, sobel_y)
    grad = img_nms(img, img_sobel_x, img_sobel_y)

    h, w = img.shape
    img_out = np.zeros(img.shape, np.uint8)
    threshold_l = threshold_l * np.max(grad)
    threshold_h = threshold_h * np.max(grad)
    for i in range(h):
        for j in range(w):
            if grad[i][j] > threshold_h:
                img_out[i][j] = 0
            if grad[i][j] < threshold_l:
                img_out[i][j] = 255
            else:
                pass
    return img_out


if __name__ == '__main__':
    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
    img_out = img_canny(img, 0.1, 0.2)
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title("Origin")

    plt.subplot(122)
    plt.imshow(img_out, cmap='gray')
    plt.title("Canny edge detect")
    plt.show()