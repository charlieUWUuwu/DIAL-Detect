# 打包
# DialReader.py

import cv2
# import os
import numpy as np
# import matplotlib.pyplot as plt
# from google.colab.patches import cv2_imshow

# from IPython.core.inputtransformer2 import ESC_SHELL
from scipy.linalg import sqrtm
# import time

import math

class DIAL(object):
  def __init__(self,img_full):
    self.img_full = img_full
    # self.ellipses = [] 

  def __del__(self): # 釋放圖片占用記憶體
    del self.img_full

  def get_ellipses(self):
    ellipses = [] # (錶盤排序為小大小大)
    ellipses.append(((61.78, 103.16), (89.28, 124.89), 178.89))# 在imgs[0]
    ellipses.append(((202.32, 106.90), (136.77, 171.20), 179.77))# 在imgs[0]
    ellipses.append(((232.04, 159.34), (85.22, 103.44), 172.22)) # 在imgs[1]
    ellipses.append(((103.28, 106.36), (148.21, 157.25), 168.87)) # 在imgs[1]
    return ellipses

  def get_preprocessed_img(self, img):
    # 裁切圖片
    imgs = []
    imgs.append(cv2.resize(img[670:1025, 155:745], (300, 224), interpolation=cv2.INTER_AREA)) #y, x
    imgs.append(cv2.resize(img[155:550, 1130:1685], (300, 224), interpolation=cv2.INTER_AREA))
    return imgs

  def _convert_params(self, a, b, angle):
    """
    将椭圆的长短轴、旋转角度参数转换为 A,B,C
    A*x^2 + B*x*y + C*y^2 = 1
    (x,y) ((A,B/2) (x,y)^T =  1
          (B/2,C))
    """
    radian = math.radians(angle)
    A = (math.cos(radian))**2 / a**2 + (math.sin(radian))**2 / b**2
    B = 2 * math.cos(radian) * math.sin(radian) * (1/a**2 - 1/b**2)
    C = (math.sin(radian))**2 / a**2 + (math.cos(radian))**2 / b**2
    transfer_matrix = [[A, B/2], [B/2, C]]
    transfer_matrix = np.array(transfer_matrix)
    return transfer_matrix

  def get_transfer_matrics(self, ellipses):
    transfer_matrics = []
    for ellipse in ellipses:
      e_x, e_y = ellipse[0]
      a, b = ellipse[1]
      e_a, e_b = a*0.5, b*0.5  ##e_a-短轴半径 e_b-长轴半径  e_b > e_a
      angle = ellipse[2]
      transfer_matrics.append(self._convert_params(e_a, e_b, angle))
    return transfer_matrics




  def _pixelInImage(self, pt, W, H):
    # boolIn = pt[0]>=0 and pt[0]< W and pt[1]>=0 and pt[1] < H
    return pt[0]>=0 and pt[0]< W and pt[1]>=0 and pt[1] < H
 
  ## 像素插值函数
  def _interpolation(self, image, pt, type=0): 
    H = image.shape[0]
    W = image.shape[1]
    pixelVal = np.zeros((1, image.shape[2]))
    
    if type==0:
      pt = np.round(pt)
      if self._pixelInImage(pt, W, H):
        pixelVal = image[int(pt[1]), int(pt[0]), :]
      else:
        pixelVal = 0

    # 双线性插值 Bilinear interpolation
    if type==1:     
      ptTL = np.floor(pt).astype(np.int64)
      ptBR = np.ceil(pt).astype(np.int64)
      ptTR = np.array([int(np.ceil(pt[0])), int(np.floor(pt[1]))])
      ptBL = np.array([int(np.floor(pt[0])), int(np.ceil(pt[1]))])

      if self._pixelInImage(ptTL, W, H) and self._pixelInImage(ptBR, W, H) and self._pixelInImage(ptTR, W, H) and self._pixelInImage(ptBL, W, H):
        diff = pt - ptTL   
        Coeffs1 = np.array([1-diff[0], diff[0]]) #1*2
        Coeffs2 = np.array([[1-diff[1]], [diff[1]]]) #2*1
        for i in range(image.shape[2]):
          Pixels = [[image[ptTL[1], ptTL[0], i], image[ptTR[1], ptTR[0], i]], [image[ptBL[1], ptBL[0], i], image[ptBR[1], ptBR[0], i]]]
          pixelVal[:, i] = (Coeffs1 @ np.double(Pixels) @ Coeffs2).astype(np.uint8)
      else:
        pixelVal = 0
    return pixelVal


  def _ellipse2circle(self, image, A, center, R):
    """
    args:
        image: original image
        A: The coefficients of the ellipse equation centered at the origin x'Ax = 1
        center: the center of the ellipse
        R: The radius after the ellipse is normalized
    returns:
      image_ellipse: The genderated ellipse image
      image_circle: The generated circular image
    """
    Transform = np.linalg.inv(sqrtm(A)) / R
    #image h*w*c
    pts = []
    H = image.shape[0]
    W = image.shape[1]
    for y in np.arange(H):
      for x in np.arange(W):
        pt = np.array([x, y])
        ptRel = pt - center
        val = (ptRel.T) @ A @ ptRel
        if val <= 1:
          pts.append(pt)
    pts = np.array(pts)  

    # 估算椭圆图像的大小
    xmin = np.min(pts[:, 0])
    ymin = np.min(pts[:, 1])
    xmax = np.max(pts[:, 0])
    ymax = np.max(pts[:, 1])

    w = xmax - xmin + 1
    h = ymax - ymin + 1 
    # 为椭圆图像的像素赋值
    imageEllipse = np.zeros((h, w, image.shape[2])).astype(np.uint8)
    for i in np.arange(len(pts)): #len(pts)
      imageEllipse[pts[i,1]-ymin, pts[i,0]-xmin, :] = image[pts[i,1], pts[i,0], :]
    
    # 生成圆形图像
    normSize = 2 * R
    imageCircle = np.zeros((normSize, normSize, image.shape[2])).astype(np.uint8)
    for x in np.arange(normSize):
      for y in np.arange(normSize):
        # 计算出变换后的点在原始图像上的位置
        ptRet = Transform.T @ ((np.array([x, y]) - np.array([normSize/2, normSize/2])))
        val = (ptRet.T) @ A @ ptRet
        if val <= 1:
          pt = ptRet + center  
          imageCircle[y, x, :] = self._interpolation(image, pt, 1)
      
    return imageEllipse, imageCircle




  def get_circles(self, imgs, ellipses, transfer_matrics):
    outs = [] # 蒐集轉換出來的圓形
    # transfer_matrics = get_transfer_matrics()
    for i in range(4):
      A = transfer_matrics[i]
      A = np.array(A)
      EllipseCenter = ellipses[i][0]
      R = 50 # 出來的圓盤半徑長
      if i < 2:
        out = self._ellipse2circle(imgs[0], A, EllipseCenter, R)
      else:
        out = self._ellipse2circle(imgs[1], A, EllipseCenter, R)
      # print('橢圓形')
      # cv2_imshow(out[0])
      # print('圓形')
      # cv2_imshow(out[1])
      # print(' ')
      outs.append(out[1])
    return outs


  # 亮度增強
  # 自適應調整亮度 : https://www.twblogs.net/a/5db2ff46bd9eee310ee65cb4
  def _compute(self, img, min_percentile, max_percentile):
    """計算分位點，目的是去掉圖1的直方圖兩頭的異常情況"""
    max_percentile_pixel = np.percentile(img, max_percentile)
    min_percentile_pixel = np.percentile(img, min_percentile)
    return max_percentile_pixel, min_percentile_pixel

  def _get_lightness(self, src):
    # 計算亮度
    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lightness = hsv_image[:,:,2].mean()
    return  lightness

  def _reduce_highlights(self, img):
    # https://ithelp.ithome.com.tw/articles/10240334
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 先轉成灰階處理
    ret, thresh = cv2.threshold(img_gray, 200, 255, 0)  # 利用 threshold 過濾出高光的部分，目前設定高於 200 即為高光
    contours, hierarchy  = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_zero = np.zeros(img.shape, dtype=np.uint8) 
    
    # print(len(contours))
    mask = None
    for contour in contours:
      x, y, w, h = cv2.boundingRect(contour) 
      img_zero[y:y+h, x:x+w] = 255 
      mask = img_zero 

    # print("Highlight part: ")
    # show_img(mask)
    
    # alpha，beta 共同決定高光消除後的模糊程度
    # alpha: 亮度的缩放因子，默認是 0.2， 範圍[0, 2], 值越大，亮度越低
    # beta:  亮度缩放後加上的参数，默認是 0.4， 範圍[0, 2]，值越大，亮度越低
    result = cv2.illuminationChange(img, mask, alpha=0.2, beta=0.4) 
    # show_img(result)
        
    return result

  def aug(self, src):
    """圖像亮度增強"""
    if self._get_lightness(src)>130:
      #print("圖片亮度足夠，不做增強")
      # src = reduce_highlights(src) # 去除高光
      # src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)  # 先轉成灰階處理
      # src = cv2.equalizeHist(src)
      return src
    # 先計算分位點，去掉像素值中少數異常值，這個分位點可以自己配置。
    # 比如1中直方圖的紅色在0到255上都有值，但是實際上像素值主要在0到20內。
    else:
      max_percentile_pixel, min_percentile_pixel = self._compute(src, 1, 99)
      
      # 去掉分位值區間之外的值
      src[src>=max_percentile_pixel] = max_percentile_pixel
      src[src<=min_percentile_pixel] = min_percentile_pixel

    # 將分位值區間拉伸到0到255，這裏取了255*0.1與255*0.9是因爲可能會出現像素值溢出的情況，所以最好不要設置爲0到255。
      out = np.zeros(src.shape, src.dtype)
      cv2.normalize(src, out, 255*0.1, 255*0.9,cv2.NORM_MINMAX)
      # out = reduce_highlights(out)

      # 對圖像進行直方圖均衡化
      # out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)  # 先轉成灰階處理
      # out = cv2.equalizeHist(out)
      return out

  # 彩色
  def _pointer_value(self, x1, y1, x2, y2, mode=0):
    # 根據角度算出數值
    if abs(x1-46)+abs(y1-54) > abs(x2-46)+abs(y2-54): # 使x1, y1為中心點 (離中心點較近的點做為中心點)
      # x1, y1, x2, y2 = x2, y2, x1, y1
      x1, y1, x2, y2 = 46, 54, x1, y1
    else:
      x1, y1 = 46, 54

    vec_x, vec_y = x2 - x1, y2 - y1
    clockwise_angle = math.degrees(math.atan2(vec_y, vec_x)) #[-180, 180]
    # print('初始數值:', clockwise_angle)

    # 由于需要输出顺时针方向的角度，因此需要进行一些调整
    if clockwise_angle < 0: # 表示在上半圓，將九點鐘方向轉為180度、三點鐘轉為360度
      clockwise_angle += 360
    clockwise_angle -= 90+45 # 從六點鐘多一些作為角度起始點

    # 0~11 不知啥單位  mode==1
    if mode==0:
      # print('mode:', mode)
      # print('clockwise_angle:', clockwise_angle)
      result = 0 + 11*(clockwise_angle/270)
      if result < 0: result = 0
      elif result > 11: result=11
    
    # -5~55  mode==2(other)
    else:
      # print('mode:', mode)
      # print('clockwise_angle:', clockwise_angle)
      result = -5 + 60*(clockwise_angle/270)
      if result < -5: result = -5
      elif result > 55: result=55
    return result, (x1, y1), (x2, y2)


  def _detect_pointer(self, cir, mode):
    cir = self.aug(cir) #此處cir須為RGB格式
    result = cir.copy()

    img = cv2.GaussianBlur(cir, (3, 3), 0)
    # edges = cv2.Canny(img, 45, 140, apertureSize=3) 80 100 3
    edges = cv2.Canny(img, 45, 150, apertureSize=3)
    # print('edges')
    # cv2_imshow(edges)

    # lines = cv2.HoughLines(edges, 1, np.pi / 180, 45)  # 这里对最后一个参数使用了经验型的值
    lines = cv2.HoughLinesP(edges, 1.0, np.pi/180, 42)

    # 排序
    lines = sorted(lines, key=lambda x: np.sqrt((x[0][2]-x[0][0])**2 + (x[0][3]-x[0][1])**2), reverse=True)

    # 選擇最長的斜直線
    x1, y1, x2, y2 = lines[0][0]
    # cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)

    # 計算數值
    value, p1, p2 = self._pointer_value(x1, y1, x2, y2, mode)
    cv2.line(result, p1, p2, (0, 0, 255), 2, cv2.LINE_AA) # BGR
    # cv2.line(result, p1, p2, (255), 2, cv2.LINE_AA) # gray

    return result, round(value, 2)


  def get_pointer_value(self):
    obj = DIAL(self.img_full)
    # cv2.imshow('img_full', obj.img_full)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    imgs = obj.get_preprocessed_img(obj.img_full) # 一個鏡頭拍的畫面切為兩部分
    # cv2.imshow('imgs', imgs[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    ellipses = obj.get_ellipses() # 橢圓形已寫死
    transfer_matrics = obj.get_transfer_matrics(ellipses)
    outs = obj.get_circles(imgs, ellipses, transfer_matrics) # 獲得圓形表盤
    # cv2.imshow('outs', outs[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    results = [] # 圖片
    values = [] # 數值

    count = 0
    for out in outs:
      result, value = obj._detect_pointer(out, count%2)
      values.append(value)
      results.append(result)
      count += 1

    return results, values