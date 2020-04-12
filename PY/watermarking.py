import cv2
import pywt
import numpy as np


class Components:
    Coefficients = []
    LL = None
    LH = None
    HL = None
    HH = None


class watermarking:
    def __init__(self, watermark_path="watermark1.jpg", ratio=0.1, wavelet="haar", level=3, x=[0.1],
                 cover_image="lena.jpg"):
        self.level = level
        self.wavelet = wavelet
        self.ratio = x[0]
        self.shape_watermark = cv2.imread(watermark_path, 0).shape
        self.x = x
        self.wmk_img_components = Components()
        self.cover_img_components = Components()

        self.cover_image_data = cv2.imread(cover_image, 0)
        if self.cover_image_data.shape != (512, 512):
            self.cover_image_data.resize(512, 512)
            cv2.imwrite(cover_image, self.cover_image_data)

    def calculate_dwt(self, img, lvl):

        if isinstance(img, str):
            img = cv2.imread(img, 0)
        coefficients = pywt.wavedec2(img, wavelet=self.wavelet, level=lvl)

        coeff_arr, slices = pywt.coeffs_to_array(coefficients)

        LL = coeff_arr[slices[0]]
        HL = coeff_arr[slices[1]['da']]
        LH = coeff_arr[slices[1]['ad']]
        HH = coeff_arr[slices[1]['dd']]

        self.shape_LL = coefficients[0].shape

        sub_bands = LL, HL, LH, HH
        return LL, sub_bands, coeff_arr, slices

    def calculate_svd(self, LL):
        U, S, V = np.linalg.svd(LL)

        return U, S, V

    def diag(self, s):
        '''
        To recover the singular values to be a matrix.
        s:  1D array
        '''
        S = np.zeros(self.shape_LL)
        row = min(S.shape)
        S[:row, :row] = np.diag(s)
        # print("S shape: ", S.shape)
        return S

    def recover(self, name):
        '''
        To recover the image from the svd components and DWT
        :param name:
        '''
        components = eval("self.{}_components".format(name))
        sll = eval("self.SLL_{}".format(name))

        components.LL = components.ULL.dot(self.diag(sll)).dot(components.VLL)

        coeffs_from_arr = pywt.array_to_coeffs(components.coeff_arr, components.slices, output_format='wavedec2')
        return pywt.waverec2(coeffs_from_arr, wavelet=self.wavelet)

    def watermark(self, img="lena.jpg", watermark_path="watermark1.jpg", path_save=None):
        '''
        This is the main function for image watermarking.
        :param img: image path or numpy array of the image.
        '''
        if not path_save:
            path_save = "watermarked_" + img
        self.path_save = path_save
        # Cover Image
        self.cover_img_components.LL, self.cover_img_components.sub_bands, \
        self.cover_img_components.coeff_arr, self.cover_img_components.slices = self.calculate_dwt(img, 3)
        self.cover_img_components.ULL, self.cover_img_components.SLL, self.cover_img_components.VLL = self.calculate_svd(
            self.cover_img_components.LL)
        # Watermark Image
        self.wmk_img_components.LL, self.wmk_img_components.sub_bands, \
        self.wmk_img_components.coeff_arr, self.wmk_img_components.slices = self.calculate_dwt(watermark_path, 3)
        self.wmk_img_components.ULL, self.wmk_img_components.SLL, self.wmk_img_components.VLL = self.calculate_svd(
            self.wmk_img_components.LL)
        # Embed Watermark
        self.embed()

        recovered_image = self.recover("cover_img")  # watermarked image
        cv2.imwrite(path_save, recovered_image)

    def extracted(self, image_path="watermarked_lena.jpg", ratio=None,
                  extracted_watermark_path="watermark_extracted.jpg"):
        '''
        Extracted the watermark from the given image.
        '''
        if not extracted_watermark_path:
            extracted_watermark_path = "watermark_extracted.jpg"
        if not image_path:
            image_path = self.path_save
        img = cv2.imread(image_path, 0)
        img = cv2.resize(img, self.shape_watermark)

        # Watermarked Image
        wmkd_img_components = Components()

        wmkd_img_components.LL, wmkd_img_components.sub_bands, \
        wmkd_img_components.coeff_arr, wmkd_img_components.slices = self.calculate_dwt(img, 3)

        wmkd_img_components.ULL, wmkd_img_components.SLL, wmkd_img_components.VLL = self.calculate_svd(
            wmkd_img_components.LL)

        ratio_ = self.ratio if not self.x[0] else ratio

        self.SLL_wmk_img = (wmkd_img_components.SLL - self.cover_img_components.SLL) / self.x[0]

        watermark_extracted = self.recover("wmk_img")

        cv2.imwrite(extracted_watermark_path, watermark_extracted)

    def embed(self):
        self.SLL_cover_img = self.cover_img_components.SLL + self.x[0] * self.wmk_img_components.LL
