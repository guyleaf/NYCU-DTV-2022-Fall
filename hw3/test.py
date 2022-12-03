import cv2


def video_demo():
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 0為電腦內置攝像頭
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    capture.set(cv2.CAP_PROP_FPS, 24)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while True:
        # print("sss")
        (
            ret,
            frame,
        ) = capture.read()  # 攝像頭讀取,ret為是否成功打開攝像頭,true,false。 frame為視頻的每一幀圖像
        frame = cv2.flip(frame, 1)  # 攝像頭是和人對立的，將圖像左右調換回來正常顯示。
        cv2.imshow("video", frame)
        c = cv2.waitKey(50)
        if c == 27:
            break


video_demo()
cv2.destroyAllWindows()
