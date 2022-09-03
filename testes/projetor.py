import numpy as np
import cv2 as cv
import os
import datetime

rafael = False #mudanças especificas para mim no uso do path e qual webcam usar

def show_image(path):
    cv.namedWindow("calibration", cv.WINDOW_NORMAL)
    cv.setWindowProperty("calibration", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    image = cv.imread(path)
    cv.imshow('calibration', image)

def calibrate(webcam):
    # Alteração de env variable para resolver bug de lentidão na inicialização de camera logitech
    # https://github.com/opencv/opencv/issues/17687
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

    # Inicialização da câmera
    cap = cv.VideoCapture(webcam, cv.CAP_DSHOW)

    if not cap.isOpened():
        print("Erro ao abrir a webcam")
        exit()

    # Exibir imagem de calibragem
    if not rafael:
        show_image('../calibration_images/camera_position.png')
    else:
        show_image('./trabalho\calibration_images\camera_position.png')

    while (cap.isOpened()):
        # Execução a cada frame da webcam
        status, frame = cap.read()
        if status:
            # Criar uma máscara que representa apenas um intervalo de vermelho
            lower_red = np.array([0, 0, 30])
            upper_red = np.array([70, 70, 255])
            mask = cv.inRange(frame, lower_red, upper_red)

            # Achar contornos
            contours, hier = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

            # Calcular maior contorno
            try: #para não crashar quando não acha contornos
                cnt = contours[0]
                for cont in contours:
                    if cv.contourArea(cont) > cv.contourArea(cnt):
                        cnt = cont

                # Aproximar maior contorno por um polígono
                epsilon = 0.01*cv.arcLength(cnt,True)
                approx = cv.approxPolyDP(cnt,epsilon,True)

                # Desenhar contorno do polígono na imagem
                mask_draw = mask.copy()
                mask_draw = cv.cvtColor(mask_draw, cv.COLOR_GRAY2BGR)
                contours_generated = cv.drawContours(mask_draw, [approx], -1, (0,0,255), 3)
                #print(f'Número de pontos do polígono: {len(approx)}')
            except:
                pass
            
            if len(approx)==4 and cv.contourArea(cnt)>5000:
                points = []
                for coord in approx:
                    points.append(coord[0])
                points = np.float32(points)
                old_points = [[0,0],[0,0],[0,0],[0,0]]
                try:
                    for point in points:
                        if np.sum(point)==min(map(np.sum,points)):
                            old_points[0] = point
                        elif np.sum(point)==max(map(np.sum,points)):
                            old_points[3] = point
                        elif point[1]/(point[0]+1)==max(map(lambda x:x[1]/(x[0]+1),points)):
                            old_points[2] = point
                        else:
                            old_points[1] = point
                    old_points = np.float32(old_points)
                except:
                    pass #o código se perde qunado se inclina a imagem 45 graus, por exemplo 
                if not rafael:
                    image = cv.imread('../calibration_images/camera_position.png')
                else:
                    image = cv.imread('./trabalho\calibration_images\camera_position.png')
                comprimento, altura = image.shape[1], image.shape[0]
                #a nova imagem tem as mesmas dimensoes que a imagem de calibraçao
                new_points = np.float32([[0,0],[comprimento,0],[0,altura],[comprimento,altura]])
                M = cv.getPerspectiveTransform(old_points,new_points)
                new_image = cv.warpPerspective(frame,M,(comprimento,altura))
                #diminuir tamanho para exibir
                resized_new = cv.resize(new_image, (comprimento//2,altura//2), interpolation=cv.INTER_AREA)
                cv.imshow("Imagem transformada", resized_new)
                old_points=old_points.reshape((-1,1,2))
                #cv.polylines(frame, old_points, True, (0,255,255),thickness=1)
                cv.imshow("Imagem normal",frame)

            # cv.namedWindow("projetor", cv.WINDOW_NORMAL)
            # cv.setWindowProperty("projetor", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
            cv.imshow('projetor', contours_generated)

        else:
            print("Arquivo de vídeo terminou. Número total de frames: %d" % (cap.get(cv.CAP_PROP_FRAME_COUNT)))
            break

        key = cv.waitKey(1)
        # Esperar por tecla "q" para sair
        if key == ord('q'):
            break
        # Esperar por tecla "l" para limpar a tela
        elif key == ord('l'):
            blank = np.zeros(frame.shape, dtype='uint8')
        # Esperar por tecla "s" para salvar imagem
        elif key == ord('s'):
            ct = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            cv.imwrite(f'../screenshots/projetor_{ct}.jpg', frame)
            print(f"Imagem salva em ../screenshots/projetor_{ct}.jpg")
        elif key == ord('c'):
            cv.destroyAllWindows()
            read_webcam(webcam, M)

    cap.release()
    cv.destroyAllWindows()

def read_webcam(webcam, matriz):
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

    if not rafael:
        show_image('../calibration_images/camera_position.png')
    else:
        show_image('./trabalho\calibration_images\camera_position.png')

    while (cap.isOpened()):
        # Execução a cada frame da webcam
        status, frame = cap.read()
        if status:
            # Transformar para escala de cinza e aplicar threshold
            # Analisar histograma de testes para validar melhor threshold
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            threshold, thresh = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)

            # Achar contornos
            contours, hier = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

            # Desenhar contornos em canvas escuro
            contours_generated = cv.drawContours(blank, contours, -1, (10,10,200), -1)

            if not rafael:
                image = cv.imread('../calibration_images/camera_position.png')
            else:
                image = cv.imread('./trabalho\calibration_images\camera_position.png')
            comprimento, altura = image.shape[1], image.shape[0]
            countours_transformed = cv.warpPerspective(contours_generated,matriz,(comprimento,altura))

            cv.namedWindow("projetor", cv.WINDOW_NORMAL)
            cv.setWindowProperty("projetor", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
            cv.imshow('projetor', countours_transformed)

        else:
            print("Arquivo de vídeo terminou. Número total de frames: %d" % (cap.get(cv.CAP_PROP_FRAME_COUNT)))
            break

        key = cv.waitKey(1)
        # Esperar por tecla "q" para sair
        if key == ord('q'):
            break
        # Esperar por tecla "l" para limpar a tela
        elif key == ord('l'):
            blank = np.zeros(frame.shape, dtype='uint8')
        # Esperar por tecla "s" para salvar imagem
        elif key == ord('s'):
            ct = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            cv.imwrite(f'./screenshots/projetor_{ct}.jpg', contours_generated)
            print(f"Imagem salva em ./screenshots/projetor_{ct}.jpg")

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    if not rafael:
        calibrate(0)
    else:
        calibrate(1)
    