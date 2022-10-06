import numpy as np
import cv2 as cv
import os
import datetime
import time

def show_image(path):
    cv.namedWindow("calibration", cv.WINDOW_NORMAL)
    cv.setWindowProperty("calibration", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    image = cv.imread(path)
    cv.imshow('calibration', image)
           

def find_rectangle(frame):
    approx = cnt = contours_generated = False
    # Criar uma máscara que representa apenas um intervalo de vermelho
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imwrite('./screenshots/calibrate_gray.png', frame)
    threshold, thresh = cv.threshold(frame, 50, 255, cv.THRESH_BINARY)  # THRESHOLD PARA DIFERENÇA DE IMAGES PB  
    cv.imwrite('./screenshots/calibrate_thresh.png', thresh)

    # Achar contornos
    contours, hier = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    # Calcular maior contorno
    try: 
        cnt = contours[0]
        for cont in contours:
            if cv.contourArea(cont) > cv.contourArea(cnt):
                cnt = cont

        # Aproximar maior contorno por um polígono
        epsilon = 0.01*cv.arcLength(cnt,True)
        approx = cv.approxPolyDP(cnt,epsilon,True)

        # Desenhar contorno do polígono na imagem
        mask_draw = thresh.copy()
        mask_draw = cv.cvtColor(mask_draw, cv.COLOR_GRAY2BGR)
        contours_generated = cv.drawContours(mask_draw, [approx], -1, (0,0,255), 3)
        print(f'Número de pontos do polígono: {len(approx)}')
    except:
        pass
    return approx, cnt, contours_generated

def find_coordinates(approx):
    points = []
    for coord in approx:
        points.append(coord[0])
    points = np.float32(points)
    old_points = [[0,0],[0,0],[0,0],[0,0]]

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
    return old_points

def use_digital_board(webcam, matriz, status, frame):
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
    clean = False

    while (cap.isOpened()):
        # Execução a cada frame da webcam
        status, frame = cap.read()
        if status and not clean:
            # Transformar para escala de cinza e aplicar threshold
            # Analisar histograma de testes para validar melhor threshold
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            threshold, thresh = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)  # THRESHOLD PARA ENCONTRAR LASER VERMELHO

            # Achar contornos
            contours, hier = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

            # Desenhar contornos em canvas escuro
            contours_generated = cv.drawContours(blank, contours, -1, (20,20,180), -1)

            image = cv.imread('./calibration_images/calibrate_black.png')
            
            comprimento, altura = image.shape[1], image.shape[0]
            countours_transformed = cv.warpPerspective(contours_generated,matriz,(comprimento,altura))

            # Borda da imagem
            cv.rectangle(countours_transformed,(0,0),(countours_transformed.shape[1],countours_transformed.shape[0]),(135,135,135),6)
            # Legendas
            font = cv.FONT_HERSHEY_COMPLEX 
            text = "Teclas de atalho: s = Salvar imagem, q = Sair, l = Limpar tela"
            cv.putText(countours_transformed,text,(10,countours_transformed.shape[0]-10), font, 0.5,(150,150,150),1,cv.LINE_AA)

            cv.namedWindow("projetor", cv.WINDOW_NORMAL)
            cv.setWindowProperty("projetor", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
            cv.imshow('projetor', countours_transformed)

        elif clean:
            # Limpar tela
            blank = np.zeros(frame.shape, dtype='uint8')
            clean = False
            contours_generated = []
            countours_transformed = cv.warpPerspective(blank,matriz,(comprimento,altura))
            # Borda da imagem
            cv.rectangle(countours_transformed,(0,0),(countours_transformed.shape[1],countours_transformed.shape[0]),(155,155,155),6)
            # Legendas
            font = cv.FONT_HERSHEY_COMPLEX 
            text = "Teclas de atalho: s = Salvar imagem, q = Sair, l = Limpar tela"
            cv.putText(countours_transformed,text,(10,countours_transformed.shape[0]-10), font, 0.5,(155,155,155),1,cv.LINE_AA)

            cv.namedWindow("projetor", cv.WINDOW_NORMAL)
            cv.setWindowProperty("projetor", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
            cv.imshow('projetor', countours_transformed)
            #time.sleep(0.5)
        else:
            print("Arquivo de vídeo terminou. Número total de frames: %d" % (cap.get(cv.CAP_PROP_FRAME_COUNT)))
            break

        # Tentar deletar essas keys para não ter código duplicado
        key = cv.waitKey(1)
        # Esperar por tecla "q" para sair
        if key == ord('q'):
            break
        # Esperar por tecla "l" para limpar a tela
        elif key == ord('l'):
            blank = np.zeros(frame.shape, dtype='uint8')
            clean = True
        # Esperar por tecla "s" para salvar imagem
        elif key == ord('s'):
            ct = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            cv.imwrite(f'./screenshots/projetor_{ct}.jpg', contours_generated)
            print(f"Imagem salva em ./screenshots/projetor_{ct}.jpg")

def calibrate(webcam):
    # Alteração de env variable para resolver bug de lentidão na inicialização de camera logitech
    # https://github.com/opencv/opencv/issues/17687
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

    # Inicialização da câmera
    cap = cv.VideoCapture(webcam, cv.CAP_DSHOW)
    status, frame = cap.read()

    if not cap.isOpened():
        print("Erro ao abrir a webcam")
        exit()

    # Estado inicial da calibração
    black_calibration = False
    white_calibration = False
    frame_count = 0
    frames_sum = []

    while (cap.isOpened()):
        # Execução a cada frame da webcam
        status, frame = cap.read()
        # Checagem de status True para camera ativa
        if status:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # Comando por teclas
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
                cv.imwrite(f'./screenshots/projetor_{ct}.jpg', frame)
                #print(f"Imagem salva em ../screenshots/projetor_{ct}.jpg")

            # Definição de qual função usar o frame
            if black_calibration == False:
                show_image('./calibration_images/calibrate_black.png')
                time.sleep(0.1)
                if 20 <= frame_count < 30:
                    # Somar frames
                    frames_sum.append(frame)
                    frame_count += 1
                    #print(frame_count)
                elif frame_count < 20:
                    frame_count += 1 
                else:
                    # Calculate blended image
                    black_mean = frames_sum[0]
                    for i in range(len(frames_sum)):
                        if i == 0:
                            pass
                        else:
                            alpha = 1.0/(i + 1)
                            beta = 1.0 - alpha
                            black_mean = cv.addWeighted(frames_sum[i], alpha, black_mean, beta, 0.0)
                            black_mean = cv.add(black_mean, -10)
                    # Salvar imagem
                    cv.imwrite('./screenshots/calibrate_black.png', black_mean)

                    black_calibration = True
                    frame_count = 0
                    frames_sum = []

            elif black_calibration == True and white_calibration == False: 
                show_image('./calibration_images/calibrate_white.png')
                time.sleep(0.1)
                if frame_count < 10:
                    # Somar frames
                    frames_sum.append(frame)
                    frame_count += 1
                    #print(frame_count)
                else:
                    # Calculate blended image
                    white_mean = frames_sum[0]
                    for i in range(len(frames_sum)):
                        if i == 0:
                            pass
                        else:
                            alpha = 1.0/(i + 1)
                            beta = 1.0 - alpha
                            white_mean = cv.addWeighted(frames_sum[i], alpha, white_mean, beta, 0.0)
                    # Salvar imagem
                    cv.imwrite('./screenshots/calibrate_white.png', white_mean)
                    white_calibration = True

            else:
                cv.destroyAllWindows()
                diff = cv.absdiff(black_mean, white_mean)
                cv.imwrite('./screenshots/calibrate_diff.png', diff)

                approx, cnt, contours_generated = find_rectangle(diff)
                try:
                    if len(approx)==4 and cv.contourArea(cnt)>5000:
                        old_points = find_coordinates(approx)
                        # Acho que vai bugar usar os pontos da imagem ao invés do frame
                        # porque se o projetor for de uma resolução menor (quadrada) a imagem
                        # vai ficar distorcida
                        image = cv.imread('./calibration_images/calibrate_black.png')
                    
                        comprimento, altura = image.shape[1], image.shape[0]
                        #a nova imagem tem as mesmas dimensoes que a imagem de calibraçao
                        new_points = np.float32([[0,0],[comprimento,0],[0,altura],[comprimento,altura]])
                        M = cv.getPerspectiveTransform(old_points,new_points)
                        new_image = cv.warpPerspective(frame,M,(comprimento,altura))
                        #diminuir tamanho para exibir
                        resized_new = cv.resize(new_image, (comprimento//2,altura//2), interpolation=cv.INTER_AREA)
                        #cv.imshow("Imagem transformada", resized_new)
                        old_points=old_points.reshape((-1,1,2))
                        use_digital_board(webcam, M, status, frame)
                        cap.release()
                        cv.destroyAllWindows()

                        # Rodar código para desenho
                        # if key == ord('c'):
                        #     cv.destroyAllWindows()
                        #     use_digital_board(webcam, M, status, frame)      
                    else:
                        pass
                except:
                    pass

            #cv.imshow("Imagem normal",frame)
            # cv.namedWindow("projetor", cv.WINDOW_NORMAL)
            # cv.setWindowProperty("projetor", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
            #cv.imshow('projetor', contours_generated)

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    calibrate(0)
    