import cv2
import numpy.fft as fft
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = cv2.imread('rect.bmp', 0)
    img_45 = cv2.imread('rect-45.bmp', 0)
    img_move = cv2.imread('rect-move.bmp', 0)

    img_fft = fft.fft2(img)
    img_fft_shift = np.abs(fft.fftshift(img_fft))
    # 数组中的0元素处理，便于后面的log运算
    img_fft_shift[img_fft_shift == 0] = 0.0001
    magnitude_spectrum = 20 * np.log(img_fft_shift)

    img_45_fft = fft.fft2(img_45)
    img_45_fft_shift = np.abs(fft.fftshift(img_45_fft))
    img_45_fft_shift[img_45_fft_shift == 0] = 0.0001
    magnitude_spectrum_45 = 20 * np.log(np.abs(img_45_fft_shift))

    img_fft_move = fft.fft2(img_move)
    img_move_fft_shift = np.abs(fft.fftshift(img_fft_move))
    # 数组中的0元素处理，便于后面的log运算
    img_move_fft_shift[img_move_fft_shift == 0] = 0.0001
    magnitude_spectrum_move = 20 * np.log(img_move_fft_shift)

    plt.subplot(231)
    plt.title('Origin')
    plt.imshow(img, cmap='gray')
    plt.subplot(234)
    plt.title('magnitude spectrum')
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.subplot(232)
    plt.title('Origin rotate')
    plt.imshow(img_45, cmap='gray')
    plt.subplot(235)
    plt.title('Rotate magnitude spectrum')
    plt.imshow(magnitude_spectrum_45, cmap='gray')
    plt.subplot(233)
    plt.title('Origin move')
    plt.imshow(img_move, cmap='gray')
    plt.subplot(236)
    plt.title('Move magnitude spectrum')
    plt.imshow(magnitude_spectrum_move, cmap='gray')

    plt.show()
