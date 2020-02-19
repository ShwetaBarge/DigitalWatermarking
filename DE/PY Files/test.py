# -*- coding: utf-8 -*-
# @Time    : 2018/5/30
# @Author  : Luke

from scipy.optimize import differential_evolution
from watermarking import watermarking
import numpy as np

watermarking = watermarking()
watermarking.watermark()
watermarking.extracted()
watermarking.psnr_cal()
#bounds = np.linspace(0, 1, num = 30, endpoint = False)
#result = differential_evolution(watermarking.func, [(0,1),(0,1),(0,1)])
#result = differential_evolution(watermarking.func, bounds)
#print(result)