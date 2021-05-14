import cv2

face = cv2.imread('niu_ece_2.jpg')  # 讀取照片
thug_life = cv2.imread('thug_life.png', cv2.IMREAD_UNCHANGED)  # 讀取圖片

thug_life_r = thug_life.shape[1] / thug_life.shape[0]
cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')  # 載入特徵分類器
grey = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # 色彩轉灰階
faceRects = cascade.detectMultiScale(
    grey, scaleFactor=1.35, minNeighbors=2)  # 開始辨識

for faceRect in faceRects:  # 依序框起來
    x, y, w, h = faceRect
    # cv2.rectangle(face, (x, y), (x + w, y + h), (255, 0, 0), 2)
    thug_life_resize = cv2.resize(thug_life, (w, int(w * thug_life_r)))  # 縮放
    for i in range(thug_life_resize.shape[0]):
        for j in range(thug_life_resize.shape[1]):
            if thug_life_resize[i, j, 3] != 0:
                for k in range(3):
                    try:
                        face[y + i + int(h / 10), x + j,
                             k] = thug_life_resize[i, j, k]
                    except:
                        pass

# cv2.imshow('thug_life', thug_life) #顯示圖片
cv2.imshow('face', face)  # 顯示圖片

cv2.waitKey(0)  # 等待按下任意鍵
cv2.destroyAllWindows()
