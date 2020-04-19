import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft


def ideal_lp_hp(img):
    """
    理想高通低通滤波器
    :param img:
    """
    height, width = img.shape
    h = np.zeros(img.shape)
    h[height // 2 - 60: height // 2 + 60, width // 2 - 60: width // 2 + 60] = 1
    # h = 1 - h
    img_fft = fft.fft2(img)
    img_shift = fft.fftshift(img_fft)
    # 低通滤波乘子h
    img_shift_lp = h * img_shift
    # 避免log计算出无意义值
    img_shift_lp[img_shift_lp == 0] = 1
    # 频谱进行处理，使得对比度更高
    magnitude_spectrum_lp = 20 * np.log(np.abs(img_shift_lp))
    img_ift_lp = fft.ifft2(fft.ifftshift(img_shift_lp))

    # 高通滤波乘子
    h = np.zeros(img.shape)
    h[height // 2 - 5: height // 2 + 5, width // 2 - 5: width // 2 + 5] = 1
    img_shift_hp = (1 - h) * img_shift
    img_shift_hp[img_shift_hp == 0] = 1
    magnitude_spectrum_hp = 20 * np.log(np.abs(img_shift_hp))
    img_ift_hp = fft.ifft2(fft.ifftshift(img_shift_hp))

    plt.subplot(221)
    plt.title("LP Filter")
    plt.imshow(np.abs(img_ift_lp), 'gray')
    plt.subplot(222)
    plt.title("magnitude spectrum")
    plt.imshow(magnitude_spectrum_lp, cmap='gray')
    plt.subplot(223)
    plt.title("HP Filter")
    plt.imshow(np.abs(img_ift_hp), 'gray')
    plt.subplot(224)
    plt.title("magnitude spectrum")
    plt.imshow(magnitude_spectrum_hp, cmap='gray')
    plt.show()


def ba_lp(img, d0=30, n=2):
    """
    巴特沃斯低通滤波
    :param img:
    :param d0: 截止频率（像素距离）
    :param n: 滤波器阶数
    :return:
    """
    img_fft = fft.fft2(img)
    img_shift = fft.fftshift(img_fft)
    # 巴特沃斯滤波乘子
    h = np.zeros(img.shape)
    height, width = h.shape
    for i in range(height):
        for j in range(width):
            h[i][j] = np.sqrt((i - height/2)**2 + (j - width/2)**2)
    h = 1 / (1 + (h / d0)**(2*n))
    img_shift = img_shift * h
    img_ift = fft.ifft2(fft.ifftshift(img_shift))
    return np.abs(img_ift)


def gass_lp(img, d0=30):
    """
    高斯低通滤波器
    :param img:
    :param d0: 滤波器截止频率（像素距离）
    :return:
    """
    img_fft = fft.fft2(img)
    img_shift = fft.fftshift(img_fft)
    # 高斯滤波乘子
    h = np.zeros(img.shape)
    height, width = h.shape
    for i in range(height):
        for j in range(width):
            h[i][j] = np.sqrt((i - height / 2) ** 2 + (j - width / 2) ** 2)
    h = np.exp(-h*h / (2 * d0 * d0))
    img_shift = img_shift * h
    img_ift = fft.ifft2(fft.ifftshift(img_shift))
    return np.abs(img_ift)


def homo_filter(img, d0=10, c=2, rH=2.2, rL=0.25):
    print(img.shape)
    # 首先对图像取对数
    img[img == 0] = 1
    img = np.log(img)
    # 然后Fourier变换
    img_fft = fft.fft2(img)
    img_shift = fft.fftshift(img_fft)
    # 高斯滤波乘子
    h = np.zeros(img.shape)
    height, width = h.shape
    for i in range(height):
        for j in range(width):
            h[i][j] = np.sqrt((i - height / 2) ** 2 + (j - width / 2) ** 2)
    h = np.exp(-h * h / (c * d0 * d0))
    h = (1 - h) * (rH - rL) + rL
    img_shift = img_shift * h
    img_ift = fft.ifft2(fft.ifftshift(img_shift))
    img_ift = np.exp(np.real(img_ift))
    return img_ift


if __name__ == '__main__':
    # img = cv2.imread('grid.bmp', 0)
    # ideal_lp_hp(img)
    img1 = cv2.imread('lena.bmp', 0)
    img2 = cv2.imread('cave.jpg', 0)
    img_ba_lp = ba_lp(img1, d0=30, n=2)
    img_ga_lp = gass_lp(img1, d0=30)
    img_homo = homo_filter(img2, d0=1000)
    plt.subplot(231)
    plt.title("Origin")
    plt.imshow(img1, cmap='gray')
    plt.subplot(232)
    plt.title("Gass Filter")
    plt.imshow(img_ga_lp, cmap='gray')
    plt.subplot(233)
    plt.title("Bat Filter")
    plt.imshow(img_ba_lp, cmap='gray')
    plt.subplot(234)
    plt.title("Origin")
    plt.imshow(img2, cmap='gray')
    plt.subplot(235)
    plt.title("Homo Filter")
    plt.imshow(img_homo, cmap='gray')
    plt.show()