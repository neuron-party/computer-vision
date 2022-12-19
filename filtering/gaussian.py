import numpy as np
import scipy


def getGaussianKernel(kernel_size: int, sigma: float):
    '''https://dsp.stackexchange.com/questions/69891/having-trouble-calculating-the-correct-gaussian-kernel-values-from-the-gaussian'''
    k = (kernel_size - 1) // 2 
    kernel = np.zeros((kernel_size, kernel_size))
    
    for i in range(1, kernel_size + 1):
        for j in range(1, kernel_size + 1):
            x = (i - (k + 1)) ** 2 + (j - (k + 1)) ** 2
            x = x / (2 * sigma ** 2)
            x = np.exp(-x)
            x = x / (2 * np.pi * sigma ** 2)
            kernel[i - 1, j - 1] = x
            
    return kernel


def GaussianBlur(img, kernel_size, sigma=1):
    '''
    Input: numpy array of shape [h, w, c] where c == 3 (RGB) or c == 1 (Grayscale)
    '''
    gaussian_kernel = getGaussianKernel(kernel_size, sigma)
    if img.shape[-1] == 1:
        transformed_img = scipy.ndimage.convolve(img, gaussian_kernel)
    elif img.shape[-1] == 3:
        transformed_img = np.dstack([
            scipy.ndimage.convolve(img[:, :, 0], gaussian_kernel),
            scipy.ndimage.convolve(img[:, :, 1], gaussian_kernel),
            scipy.ndimage.convolve(img[:, :, 2], gaussian_kernel)
        ])
    else:
        raise ValueError('invalid input image')
    return transformed_img