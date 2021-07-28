# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 22:39:04 2019

@author: USER
"""

import os, sys
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.functions import softmax, cross_entropy_error

class SoftmaxWithLoss:
   def __init__(self):
       self.loss = None  # 손실
       self.y = None  # softmax의 출력
       self.t = None  # 정답 레이블(one-hot)
       
   def forward(self, x, t):
       self.t = t
       self.y = softmax(x)
       self.loss = cross_entropy_error(self.y, self.t)
       return self.loss
   
   def backward(self, dout=1):
       batch_size = self.t.shape[0]
       dx = (self.y - self.t) / batch_size
       
       return dx
 



