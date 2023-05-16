# 錶盤辨識

:bulb: 指針盤定位採用的是固定的橢圓位置，非即時偵測

😓 目前中間兩個大指針盤偵測還不是很穩定

[:cactus:點我到 github](https://github.com/charlieUWUuwu/DIAL-Detect.git)

 
## 數值輸出
由左到右的錶盤偵測數值，數值範圍:
- 小錶盤 : [0, 11]
- 大錶盤 : [-5, 55]
若偵測失敗，以 `-100` 當作偵測數值

## 偵測步驟
1. `get_preprocessed_img()` : 將左右錶盤兩兩為一組，切成兩張照片，並resize成`(300, 204)`後回傳。
2. `get_ellipses()` : 回傳錶盤的固定橢圓位置
3. `get_transfer_matrics(ellipses)` : 獲得每個橢圓以原點為中心的橢圓方程的係數 x'Ax = 1。
4. `get_circles(imgs, ellipses, transfer_matrics)` : 回傳橢圓轉換成的圓形。
5. `aug(src)` : 回傳調整亮度後的影像，若亮度超過 130 者才進行增強。
6. `_detect_pointer(cir, mode)` : 霍夫直線檢測，找出指針位子座標。mode為0表示處理小錶盤，為1者則處理大錶盤。
7. `_pointer_value(x1, y1, x2, y2, mode)` : 根據傳入指針座標計算對應的錶盤數值，並回傳。

## 當前困難點
- 橢圓辨識效果不佳，且尚未找到能將錯誤偵測的橢圓完全剔除的方法(故目前採用固定的橢圓定位)
- 數值辨識偶有不穩，仍可能出現錯誤辨識

## 使用方法
### **單張影像辨識**
```python
from DialReader import DIAL

# 讀取待辨識影像
img = cv2.imread(path)

# 初始化
obj = DIAL(img)

# 獲取影像中四個錶盤的偵測結果
# img:錶盤影像；values:指針數值
imgs, values = obj.get_pointer_value()
```

---

### **即時影像辨識**
🥺 請以`.py`檔案執行，不要使用`.ipynb`，否則daemon設定會有問題

:bulb: 鏡頭即時拍攝的圖片以 10 的倍數命名，存於指定的資料夾中

```python
# Identify_img.py
import cv2
from DialReader import DIAL
import time
import threading

def process_image(name, sec_count):
    img = cv2.imread('./FOLDER_PATH/' + name)

    obj = DIAL(img)
    imgs, values = obj.get_pointer_value()

    # 由左至右的錶盤
    print("sec_count: " + str(sec_count))
    print(values[0])
    print(values[1])
    print(values[3])
    print(values[2])

if __name__=='__main__':
    rtsp_url = "rtsp://justin:jufan2534879@210.61.41.223:554/stream1"

    # 創建VideoCapture對象
    cap = cv2.VideoCapture(rtsp_url)

    # 設置捕獲視頻的畫面大小
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # 設置每10秒為一個間隔捕獲一張圖像
    interval = 10

    # 定義保存圖像的文件夾路徑
    save_path = '../cold img output/'

    count = 0
    timeStart = int(time.time())
    while True:
        # 讀取RTSP影像
        ret, frame = cap.read()
        
        if ret:
            sec_count = int(time.time()-timeStart)
            print('sec:', sec_count , ' success')
            # 每interval秒捕獲一張圖像
            if sec_count % interval == 0:
            
                # 生成文件名
                filename = time.strftime("image") + str(sec_count) + ".jpg" 

                # 拼接文件路徑
                file_path = save_path + "/" + filename

                # 保存圖像
                lock.acquire()
                cv2.imwrite(file_path, frame)
                lock.release()

                # 同時處理圖片
                threading.Thread(target=process_image, args=(filename, sec_count,), daemon=True).start()  # 設定為daemon thread
                count += 1
        else:
            # 若讀取失敗，則重新連接
            print('error')
            cap.release()
            cap = cv2.VideoCapture(rtsp_url)

        # 暫停1秒
        time.sleep(1)
```
