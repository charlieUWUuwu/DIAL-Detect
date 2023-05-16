# Identify_img.py
import cv2
from DialReader import DIAL
import time
import threading

lock = threading.Lock() 

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
