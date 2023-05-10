# DialReader.py

import cv2, math
import numpy as np
from scipy.linalg import sqrtm

class DIAL(object):
  def __init__(self,img_full):
    self.img_full = img_full

  def __del__(self): # 釋放圖片占用記憶體
    del self.img_full

  def get_ellipses(self):
    # (錶盤排序為小大小大)
    ellipses = [((59.715301513671875, 101.79426574707031), (79.736328125, 102.50804138183594), 168.863037109375),  # 在imgs[0]
                ((212.0397491455078, 102.50831604003906), (126.36136627197266, 141.54234313964844), 176.99102783203125),  # 在imgs[0]
                ((193.982177734375, 133.10011291503906), (88.37353515625, 98.0330581665039), 161.79904174804688), # 在imgs[1]
                ((90.61474609375, 91.00980377197266), (124.4732666015625, 130.12530517578125), 137.1392364501953)]  # 在imgs[1]
    return ellipses

  def get_preprocessed_img(self, img):
    # 裁切圖片
    imgs = []
    imgs.append(cv2.resize(img[650:1025, 135:765], (300, 204), interpolation=cv2.INTER_AREA)) #y, x
    imgs.append(cv2.resize(img[205:600, 1130:1685], (300, 204), interpolation=cv2.INTER_AREA))
    return imgs

  def _convert_params(self, a, b, angle):
    """
    將橢圓的長短軸、旋轉角度參數轉換為 A,B,C
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

  def get_transfer_matrics(self, ellipses): # get the coefficients of the ellipse equation centered at the origin x'Ax = 1, transfer_matrics==[A1, A2, A3, A4] 一個橢圓一個A
    transfer_matrics = []
    for ellipse in ellipses:
      e_x, e_y = ellipse[0]
      a, b = ellipse[1]
      e_a, e_b = a*0.5, b*0.5  ##e_a-短軸半徑 e_b-長軸半徑  e_b > e_a
      angle = ellipse[2]
      transfer_matrics.append(self._convert_params(e_a, e_b, angle))
    return transfer_matrics

  def _pixelInImage(self, pt, W, H):
    return pt[0]>=0 and pt[0]< W and pt[1]>=0 and pt[1] < H
 
  ## 像素插值函數
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

    # 雙線性插值 Bilinear interpolation
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

  # ref : https://zhuanlan.zhihu.com/p/546951395
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

    # 估算橢圓圖像的大小
    xmin = np.min(pts[:, 0])
    ymin = np.min(pts[:, 1])
    xmax = np.max(pts[:, 0])
    ymax = np.max(pts[:, 1])

    w = xmax - xmin + 1
    h = ymax - ymin + 1 
    # 為橢圓圖像的像素賦值
    imageEllipse = np.zeros((h, w, image.shape[2])).astype(np.uint8)
    for i in np.arange(len(pts)): #len(pts)
      imageEllipse[pts[i,1]-ymin, pts[i,0]-xmin, :] = image[pts[i,1], pts[i,0], :]
    
    # 生成圓形圖像
    normSize = 2 * R
    imageCircle = np.zeros((normSize, normSize, image.shape[2])).astype(np.uint8)
    for x in np.arange(normSize):
      for y in np.arange(normSize):
        # 計算出變換後的點在原始圖像上的位置
        ptRet = Transform.T @ ((np.array([x, y]) - np.array([normSize/2, normSize/2])))
        val = (ptRet.T) @ A @ ptRet
        if val <= 1:
          pt = ptRet + center  
          imageCircle[y, x, :] = self._interpolation(image, pt, 1)
      
    return imageEllipse, imageCircle

  def get_circles(self, imgs, ellipses, transfer_matrics):
    outs = [] # 蒐集轉換出來的圓形
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

  # ref : https://ithelp.ithome.com.tw/articles/10240334
  def _reduce_highlights(self, img):
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 先轉成灰階處理
    ret, thresh = cv2.threshold(img_gray, 200, 255, 0)  # 利用 threshold 過濾出高光的部分，目前設定高於 200 即為高光
    contours, hierarchy  = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_zero = np.zeros(img.shape, dtype=np.uint8) 
    
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
      return out

  def _pointer_value(self, x1, y1, x2, y2, mode=0):
    # 根據角度算出數值
    if (x1-50)**2+(y1-50)**2 > (x2-50)**2+(y2-50)**2: # 使x1, y1為中心點 (離中心點較近的點做為中心點)
      x1, y1, x2, y2 = 50, 50, x1, y1
    else:
      x1, y1 = 50, 50

    vec_x, vec_y = x2 - x1, y2 - y1
    clockwise_angle = math.degrees(math.atan2(vec_y, vec_x)) #[-180, 180]
    # print('初始數值:', clockwise_angle)

    # 順時針旋轉為正，且以錶盤數值起始點作為0度
    if clockwise_angle < 0: # 表示在上半圓，將三點鐘方向轉為0度、九點鐘轉為180度
      clockwise_angle += 360

    if mode==0: # 0~11 不知啥單位  mode==1
        clockwise_angle -= 90+48
        # print("角度:", clockwise_angle)
        # print('mode:', mode)
        # print('clockwise_angle:', clockwise_angle)
        # print("順時針角度:", clockwise_angle)
        if clockwise_angle < -5.5: # 表示偵測到指針的另一端
          clockwise_angle += 180
          x2 = 100 - x2
          y2 = 100 - y2
        elif clockwise_angle > 267:
          clockwise_angle -= 180
          x2 = 100 - x2
          y2 = 100 - y2
        result = 0 + 11*(clockwise_angle/264)
        if result < 0: 
          result = 0
        elif result > 11: 
          result=11
    else: # -5~55  mode==2(other)
      clockwise_angle -= 90+50
      # print("角度:", clockwise_angle)
      # print('mode:', mode)
      # print('clockwise_angle:', clockwise_angle)
      # print("順時針角度:", clockwise_angle)
      if clockwise_angle < -2: # 表示偵測到指針的另一端，5為緩衝，正常要是0
        clockwise_angle += 180
        x2 = 100 - x2
        y2 = 100 - y2
      elif clockwise_angle >= 263: 
        clockwise_angle -= 180
        x2 = 100 - x2
        y2 = 100 - y2
      result = -5 + 60*(clockwise_angle/260)
      if result < -5: 
        result = -5
      elif result > 55: 
        result=55

    return result, (x1, y1), (x2, y2)

  def _detect_pointer(self, cir, mode):
    cir_copy = cir.copy()
    cir_copy = self.aug(cir_copy) #此處cir須為RGB格式
    cir_copy = cv2.medianBlur(cir_copy, 3)   # 模糊化，去除雜訊

    # 指針偵測
    lightness = self._get_lightness(cir_copy)
    # print("detect_pointer 亮度:", lightness)
    lines, edges = None, None

    if mode == 0: # 小錶盤
      if lightness < 130: # 偏暗(早上居多)
        edges = cv2.Canny(cir_copy, 15, 100, apertureSize=3) # 邊緣檢測 
        lines = cv2.HoughLinesP(edges, 1.0, np.pi/180, 40) #39
      else:
        edges = cv2.Canny(cir_copy, 45, 130, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1.0, np.pi/180, 32) #39
    else: # 大錶盤
      if lightness < 130: # 偏暗(早上居多)
        edges = cv2.Canny(cir_copy, 15, 130, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1.0, np.pi/180, 35) #39
      else: # 偏亮(晚上居多)
        edges = cv2.Canny(cir_copy, 45, 70, apertureSize=3) # 邊緣檢測 
        lines = cv2.HoughLinesP(edges, 1.0, np.pi/180, 36) #39
    # print('edges')
    # cv2_imshow(edges)

    # 若未偵測到直線
    if lines is None:
      return cir_copy, -100

    # 排序
    lines = sorted(lines, key=lambda x: (x[0][2]-x[0][0])**2 + (x[0][3]-x[0][1])**2, reverse=True)

    # 選擇最長的斜直線
    longest_line = lines[0]

    x1, y1, x2, y2 = longest_line[0]
    # cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)

    # print(len(lines)) # 輸出偵測到的線條總數量
    # for line in lines: 
    #   x1, y1, x2, y2 = line[0] 
    #   cv2.line(cir_copy, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)

    # 計算數值
    value, p1, p2 = self._pointer_value(x1, y1, x2, y2, mode)
    cv2.line(cir_copy, p1, p2, (0, 0, 255), 2, cv2.LINE_AA) # BGR
    value = round(value, 2)
    return cir_copy, value


  def get_pointer_value(self):
    obj = DIAL(self.img_full)
    imgs = obj.get_preprocessed_img(obj.img_full) # 一個鏡頭拍的畫面切為兩部分
    ellipses = obj.get_ellipses() # 橢圓形已寫死
    transfer_matrics = obj.get_transfer_matrics(ellipses)
    outs = obj.get_circles(imgs, ellipses, transfer_matrics) # 獲得圓形表盤

    results = [] # 圖片
    values = [] # 數值

    count = 0
    for out in outs:
      out_copy = out.copy()
      lightness = self._get_lightness(out_copy)
      # print("亮度:", lightness)

      if count % 2 == 0:
        result, value = self._detect_pointer(out_copy, count%2)
        # print('偵測數值:', value)
        # print('result')
        # cv2_imshow(result)
        # print(' ')
        values.append(value)
        results.append(result)
      else: # 大表盤
        # 額外影像處理 https://steam.oxxostudio.tw/category/python/ai/opencv-adjust.html   
        contrast, brightness = 50, 0 
        if lightness > 140: # 早上因沒有燈光對著照，所以反而較暗
          contrast, brightness = 100, -50
        out_copy = out_copy * (contrast/127 + 1) - contrast + brightness # 轉換公式
        out_copy = np.clip(out_copy, 0, 255)
        out_copy = np.uint8(out_copy)
        result, value = self._detect_pointer(out_copy, count%2)
        values.append(value)
        results.append(result)
      count += 1
    return results, values
