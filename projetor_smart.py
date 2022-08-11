import numpy as np
import cv2 as cv
import os

from modules.contour_generation import *
from modules.image_preprocessing import *

def read_webcam(webcam):
    # Alteração de env variable para resolver bug de lentidão na inicialização de camera logitech
    # https://github.com/opencv/opencv/issues/17687
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

    # Inicialização da câmera
    cap = cv.VideoCapture(webcam, cv.CAP_DSHOW)

    if not cap.isOpened():
        print("Erro ao abrir a webcam")
        exit()

    # Criar canvas simulando quadro de projeção
    status, frame = cap.read()
    blank = np.zeros(frame.shape, dtype='uint8')

    while (cap.isOpened()):
        # Execução a cada frame da webcam
        status, frame = cap.read()
        if status:
            thresh = image_preprocess(frame)
            contours_generated = identify_contour(thresh, blank)

            cv.namedWindow("projetor", cv.WINDOW_NORMAL)
            cv.setWindowProperty("projetor", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
            cv.imshow('projetor', contours_generated)

        else:
            print("Arquivo de vídeo terminou. Número total de frames: %d" % (cap.get(cv.CAP_PROP_FRAME_COUNT)))
            break

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    read_webcam(0)
    