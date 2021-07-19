import numpy as np
import math
from sympy import *
import tensorflow as tf


def DoG_OOCS(x, y, center, gamma, radius):
    """compute weight at location (x, y) in the OOCS kernel with given parameters
        Parameters:
            x , y : position of the current weight
            center : position of the kernel center
            gamma : center to surround ratio
            radius : center radius

        Returns:
            excite and inhibit : calculated from Equation2 in the paper, without the coefficients A-c and A-s

    """
    # compute sigma from radius of the center and gamma(center to surround ratio)
    sigma = (radius / (2 * gamma)) * (math.sqrt((1 - gamma ** 2) / (-math.log(gamma))))
    excite = (1 / (gamma ** 2)) * math.exp(-1 * ((x - center) ** 2 + (y - center) ** 2) / (2 * ((gamma * sigma) ** 2)))
    inhibit = math.exp(-1 * ((x - center) ** 2 + (y - center) ** 2) / (2 * (sigma ** 2)))

    return excite , inhibit


def On_Off_Center_filters(radius, gamma, in_channels, out_channels, off=False):
    """compute the kernel filters with given shape and parameters
        Parameters:
            gamma : center to surround ratio
            radius : center radius
            in_channels and out_channels: filter dimensions
            off(boolean) : if false, calculates on center kernel, and if true, off center

        Returns:
            kernel : On or Off center conv filters with requested shape

    """

    # size of the kernel
    kernel_size = int((radius/gamma)*2-1)
    # center node index
    centerX = int((kernel_size+1)/2)

    posExcite = 0
    posInhibit = 0
    negExcite = 0
    negInhibit = 0

    for i in range(kernel_size):
        for j in range(kernel_size):
            excite, inhibit = DoG_OOCS(i+1,j+1, centerX, gamma, radius)
            if excite > inhibit:
                posExcite += excite
                posInhibit += inhibit
            else:
                negExcite += excite
                negInhibit += inhibit

    # Calculating A-c and A-s, with requiring the positive vlaues sum up to 1 and negative vlaues to -1
    x, y = symbols('x y')
    solution = solve((x * posExcite + y * posInhibit - 1, negExcite * x + negInhibit * y + 1), x, y)
    A_c, A_s = float(solution[x].evalf()), float(solution[y].evalf())

    # making the On-center and Off-center conv filters
    kernel = np.zeros([kernel_size, kernel_size, in_channels, out_channels])

    for i in range(kernel_size):
        for j in range(kernel_size):
            excite, inhibit = DoG_OOCS(i+1,j+1, centerX, gamma, radius)
            weight = excite*A_c + inhibit*A_s
            if off:
                weight *= -1.
            kernel[i][j] = tf.fill([in_channels, out_channels], weight)

    return kernel.astype(np.float32)


def DoG_SM(x, y, center, sigma_e, sigma_i):
    """compute weight at location (x, y) in the SM kernel with given parameters
        Parameters:
            x , y : position of the current weight
            sigma_e : excitation variance
            sigma_i : inhibition variance
            center : position of the kernel center


        Returns:
            excite and inhibit : calculated from DoG equation proposed in the paper: "surround modulation: a bio-inspired connectivity structure for convolutional neural networks"

    """

    excite = (1 / (2 * np.pi * (sigma_e ** 2))) * math.exp(-1 * ((x - center) ** 2 + (y - center) ** 2) / (2 * (sigma_e ** 2)))
    inhibit = (1 / (2 * np.pi * (sigma_i ** 2))) * math.exp(-1 * ((x - center) ** 2 + (y - center) ** 2) / (2 * (sigma_i ** 2)))

    return excite,  inhibit


def SM_kernel_DoG(kernel_size, sigma_e, sigma_i, in_channels, out_channels):
    """compute the kernel filters with given shape and parameters
        Parameters:
            kernel_size : size of the kernel
            sigma_e, sigma_i : variance parameters, needed to calculate DoG function
            in_channels and out_channels: filter dimensions

        Returns:
            kernel : SM conv filters with requested shape

    """

    centerX = int((kernel_size + 1) / 2)

    kernel = np.zeros([kernel_size, kernel_size, in_channels, out_channels])
    for i in range(kernel_size):
        for j in range(kernel_size):
            excite, inhibit = DoG_SM(i+1,j+1, centerX, sigma_e, sigma_i)
            weight = excite - inhibit
            kernel[i][j] = tf.fill([in_channels, out_channels], weight)

    return kernel.astype(np.float32)


def SM_Kernel(in_channels, out_channels):
    """The kernel filters taken from Figure1.b of the paper: "surround modulation: a bio-inspired connectivity structure for convolutional neural networks"
        Parameters:
            in_channels and out_channels: filter dimensions

        Returns:
            kernel : SM conv filters used in experiments of the SM paper with requested shape

    """
    full1 = tf.fill([in_channels, out_channels], 0.17)
    full2 = tf.fill([in_channels, out_channels], 0.49)
    full3 = tf.fill([in_channels, out_channels], 1.0)
    full4 = tf.fill([in_channels, out_channels], -0.27)
    full5 = tf.fill([in_channels, out_channels], -0.23)
    full6 = tf.fill([in_channels, out_channels], -0.18)
    filters = [[full4, full5, full6, full5, full4],
              [full5, full1, full2, full1, full5],
              [full6, full2, full3, full2, full6],
              [full5, full1, full2, full1, full5],
              [full4, full5, full6, full5, full4]]

    return filters


def Averaged_Kernel(radius, gamma, in_channels, out_channels, off=false):
    """compute the kernel filters with given shape and parameters with averaging
        Parameters:
            gamma : center to surround ratio
            radius : center radius
            in_channels and out_channels: filter dimensions
            off(boolean) : if false, calculates on center kernel, and if true, off center

        Returns:
            kernel : On or Off center conv filters with requested shape,
                     with 1/n_s and 1/n_c as the absolute weight values of surround and center

    """

    # size of the kernel
    kernel_size = int((radius / gamma) * 2 - 1)
    # number of elements in center
    n_c = (radius + 1) ** 2
    # number of elements in surround
    n_s = (kernel_size ** 2) - n_c

    center_weight, surround_weight = 1.0 / n_c , -1.0 / n_s

    if off:
        center_weight, surround_weight = -1.0 / n_c, 1.0 / n_s

    full1 = tf.fill([in_channels, out_channels], surround_weight)
    full2 = tf.fill([in_channels, out_channels], center_weight)
    filters = [[full1, full1, full1, full1, full1],
              [full1, full2, full2, full2, full1],
              [full1, full2, full2, full2, full1],
              [full1, full2, full2, full2, full1],
              [full1, full1, full1, full1, full1]]

    return filters

