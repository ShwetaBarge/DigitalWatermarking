# -*- coding: utf-8 -*-
# @Time    : 14/04/2017 14:46
# @Author  : Shweta Barge
import cv2
import pywt
import numpy as np
import skimage.metrics
from matplotlib import pyplot as plt
import random

class Components():
    Coefficients = []
    U = None
    S = None
    V = None


class watermarking():
    def __init__(self, watermark_path="watermark1.jpg", ratio=0.1, wavelet="haar",
                 level=2, x = [0.1]):
        self.level = level
        self.wavelet = wavelet
        self.ratio = x[0]
        self.shape_watermark = cv2.imread(watermark_path, 0).shape
        #print("Shape", self.shape_watermark)
        self.W_components = Components()
        self.img_components = Components()
        self.W_components.Coefficients, self.W_components.U, \
        self.W_components.S, self.W_components.V = self.calculate(watermark_path)
        self.x = x
        #self.W_ndarr = cv2.imread(watermark_path,0)
        #print("watermark", self.W_ndarr.shape)

    def calculate(self, img):
        '''
        img is either numpy array or path of Image
        This function returns the SVD components of the image
        '''
        if isinstance(img, str):
            img = cv2.imread(img, 0)
        Coefficients = pywt.wavedec2(img, wavelet=self.wavelet, level=self.level)
        self.shape_LL = Coefficients[0].shape
        U, S, V = np.linalg.svd(Coefficients[0])
        return Coefficients, U, S, V

    def diag(self, s):
        '''
        To recover the singular values to be a matrix.
        s:  1D array
        '''
        S = np.zeros(self.shape_LL)
        row = min(S.shape)
        S[:row, :row] = np.diag(s)
        return S

    def recover(self, name):
        '''
        To recover the image from the svd components and DWT
        :param name:
        '''
        components = eval("self.{}_components".format(name))
        s = eval("self.S_{}".format(name))
        components.Coefficients[0] = components.U.dot(self.diag(s)).dot(components.V)
        return pywt.waverec2(components.Coefficients, wavelet=self.wavelet)

    def watermark(self, img="lena.jpg", path_save=None):
        '''
        This is the main function for image watermarking.
        :param img: image path or numpy array of the image.
        '''
        if not path_save:
            path_save = "watermarked_" + img
        self.path_save = path_save
        self.img_components.Coefficients, self.img_components.U, \
        self.img_components.S, self.img_components.V = self.calculate(img)
        self.embed()
        img_rec = self.recover("img")   #watermarked image
        cv2.imwrite(path_save, img_rec)

    def extracted(self, image_path="watermarked_lena.jpg", ratio=None, extracted_watermark_path = "watermark_extracted.jpg"):
        '''
        Extracted the watermark from the given image.
        '''
        if not extracted_watermark_path:
            extracted_watermark_path = "watermark_extracted.jpg"
        if not image_path:
            image_path = self.path_save
        img = cv2.imread(image_path,0)
        img = cv2.resize(img, self.shape_watermark)
        img_components = Components() #watermarked image
        img_components.Coefficients, img_components.U, img_components.S, img_components.V = self.calculate(img)
        ratio_ = self.ratio if not self.x[0] else ratio
        self.S_W = (img_components.S - self.img_components.S) / self.x[0]
        watermark_extracted = self.recover("W")
        cv2.imwrite(extracted_watermark_path, watermark_extracted)

    def embed(self):
        #print("Level: ", self.level)
        self.S_img = self.img_components.S + self.x[0] * self.W_components.S * \
                                             (self.img_components.S.max() / self.W_components.S.max())

    def psnr_cal(self, img1="lena.jpg" , img2="watermarked_lena.jpg"):
        im1 = cv2.imread(img1,0)
        im2 = cv2.imread(img2,0)
        # Compute PSNR over tf.uint8 Tensors.
        psnr = skimage.metrics.peak_signal_noise_ratio(im1, im2)
        numerator = im1.dot(im2)
        numerator = numerator.sum()
        fac = 100000
        deno1 = pow(im1.dot(im1), 0.5)
        deno1 = deno1.sum()
        deno2 = pow(im2.dot(im2), 0.5)
        deno2 = deno2.sum()
        deno = deno1 * deno2
        if deno == 0:
            answer = 0
        else:
            answer = (numerator/ deno)*fac
        return psnr, answer
    
 



    



