# Identify_img.py
import cv2
from DialReader import DIAL
import gc
import time

if __name__ == '__main__':
  
  img_full = cv2.imread('output.jpg')

  start = time.time()
  obj = DIAL(img_full)
  results, values = obj.get_pointer_value()

  for i in range(len(results)):

    print(values[i])
    # 顯示圖片
    # cv2.imshow('results', results[i])

    # 按下任意鍵則關閉所有視窗
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
  del img_full
  gc.collect()

  end = time.time()
  print("花費時間:"+str(end-start)+"秒")
