
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def dimensions_largest_contour(contour, img_rgb):

    rect = cv2.minAreaRect(contour)
    angle = rect[-1]
    #print(angle)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    #Maior dimensão -- w
    if rect[1][0]>rect[1][1]:
        H = rect[1][1]
        W = rect[1][0]
    else:
        H = rect[1][0]
        W = rect[1][1]

    # Calcula a área do contorno
    area = cv2.contourArea(contour)
    #print(area)

    # Calcula o perímetro do contorno
    perimeter = cv2.arcLength(contour, True)

    # Calcula a circularidade
    circularity = (4 * np.pi * area) / (perimeter * perimeter)

    C= circularity

    #Menor dimensão -- h
    x,y,w,h = cv2.boundingRect(contour)
    POS_Y= y+(h/2)
    POS_X= x+(w/2)

    cv2.drawContours(img_rgb,[box],0,(0,0,255),2)

    return [H,W,C,POS_X,POS_Y,angle,area]

def cut_and_list_imgs(img_gray, contours_red):
    imgs=[]
    positions=[]

    for contour in contours_red:

        x,y,w_red,h_red = cv2.boundingRect(contour)
        if h_red >= 123 and h_red <= 137: 
            #cv2.rectangle(img_contour,(x,y),(x+w,y+h),(0,255,0),2)

            POS_X=x
            POS_Y=y+(h_red/2)
            positions.append([POS_X, POS_Y])
        
            img_pill= img_gray[y+15:y+h_red-10,x+28:x+w_red-47]
            imgs.append(img_pill)

    return [imgs, positions]

def find_all_green(contour, img_rgb):
    # Iterar sobre os contornos verdes e calcular as coordenadas dos retângulos delimitadores
    x, y, w_green, h_green = cv2.boundingRect(contour)
    # rotrect= cv2.minAreaRect(contour)
    # h_green= rotrect[1][0]
    # w_green=rotrect[1][1]
    if h_green >= 123 and h_green <= 137: 
        STATUS = "COR"
        H2 = h_green
        #info_capsula.append([y, STATUS, H2])  
        #info_capsula.append((y, STATUS, H2, 'Verde'))  # Armazenar informações do retângulo verde
        # Desenhar contorno verde sobre a imagem original
        cv2.drawContours(img_rgb, [contour], -1, (0, 255, 0), 2)
        texto = f"Verde: {y}: ({H2},{STATUS})"
        cv2.putText(img_rgb, texto , (x-130, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return [y, STATUS, H2]

def find_all_blue(contour, img_rgb):
    # Iterar sobre os contornos azuis e calcular as coordenadas dos retângulos delimitadores
    x, y, w_blue, h_blue = cv2.boundingRect(contour)
    if h_blue >= 123 and h_blue <= 137: 
        STATUS = "COR"
        H2 = h_blue
        #info_capsula.append([y, STATUS, H2])  
        # info_capsula.append((y, STATUS, H2, 'Azul'))  # Armazenar informações do retângulo azul
        # Desenhar contorno azul sobre a imagem original
        cv2.drawContours(img_rgb, [contour], -1, (0, 0, 255), 2)
        texto = f"Azul: {y}: ({H2},{STATUS})"
        cv2.putText(img_rgb, texto , (x-130, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return [y, STATUS, H2]


def find_all_red(contour, img_rgb):
    # Iterar sobre os contornos azuis e calcular as coordenadas dos retângulos delimitadores
    x, y, w_red, h_red = cv2.boundingRect(contour)

    STATUS=''
    H2=0
    if h_red >= 123:
        if h_red <= 137: 
            if w_red > 225: 
                STATUS = "QUEBRADA"
                H2 = h_red
            else:
                STATUS = ""
                H2 = h_red

        cv2.drawContours(img_rgb, [contour], -1, (255, 0, 0), 2)
        cv2.rectangle
        texto = f"Vermelho: {y}: ({H2},{STATUS})"
        cv2.putText(img_rgb, texto , (x-130, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    return [y, STATUS, H2]

def find_all_pills(contours_red,contours_green,contours_blue, img_rgb ):
    info_capsula = []

    for contour_green in contours_green:
         # Iterar sobre os contornos verdes e calcular as coordenadas dos retângulos delimitadores
        x, y, w_green, h_green = cv2.boundingRect(contour_green)
        STATUS=''
        H2=h_green

        if h_green >= 123 and h_green <= 137: 
            STATUS = "COR"
            POS_X=x
            POS_Y=y+(H2/2)
            info_capsula.append([STATUS,POS_X, POS_Y,H2])  
 
    for contour_blue in contours_blue:
        # Iterar sobre os contornos azuis e calcular as coordenadas dos retângulos delimitadores
        x, y, w_blue, h_blue = cv2.boundingRect(contour_blue)
        STATUS=''
        H2=h_blue
        if h_blue >= 123 and h_blue <= 137: 
            STATUS = "COR"
            H2 = h_blue
            POS_X=x
            POS_Y=y+(H2/2)
            info_capsula.append([STATUS,POS_X, POS_Y,H2])  
  
    for contour_red in contours_red:
        # Iterar sobre os contornos azuis e calcular as coordenadas dos retângulos delimitadores
        x, y, w_red, h_red = cv2.boundingRect(contour_red)
    
        STATUS=''
        H2=h_red
        if h_red >= 123:
            if h_red <= 137: 
                if w_red > 225: 
                    STATUS = "QUEBRADA"
                    H2 = h_red
                else:
                    STATUS = ""
                    H2 = h_red
            POS_X=x
            POS_Y=y+(H2/2)
            info_capsula.append([STATUS,POS_X, POS_Y,H2])  

    return info_capsula

def reorder(info_capsula,i_classificador):
    #Menor para o menor:
    info_capsula_ordenada = sorted(info_capsula, key=lambda x: x[i_classificador],reverse=False)

    return info_capsula_ordenada

def mashed_classifier(info_capsula_ordenada, img_rgb, contours_black):
    for lista_info in info_capsula_ordenada:
        #columns=['Indice','STATUS', 'X', 'Y','H2', 'H1', 'W', 'C']) 
        STATUS=lista_info[1]
        H1=float(lista_info[5])
        W=float(lista_info[6])
        C=float(lista_info[7])
        angle=abs(int(lista_info[10]))
        area= int(lista_info[11])
        
        if STATUS=='':
            #Checando pelas dimensões, circulariedade sem considerar imagens inclinadas
            if ((W>441 or W<407) or (H1<127 or H1>141) or (C>0.62 or C<0.573)) and ((angle>=-5 and angle<5) or (angle>=80 and angle<95) ):
                lista_info[1]= 'AMASSADA'  
            #Imagens que não se encaixam nessas condições para serem classificadas por circulariedade:
            #Classificação por área (pílula comida)
            elif area<50000:
                lista_info[1]= 'AMASSADA'  
   
    return info_capsula_ordenada

def cracked_classifier(img_bin, info_capsula_ordenada):
    #NOVO BLOB- Apenas com cor:
    # Set up the detector with default parameters.
    params = cv2.SimpleBlobDetector_Params()
    # Set blob color (0=black, 255=white)
    params.filterByColor = True
    params.blobColor = 0
    # Filter by Area
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 500
    # Filter by Circularity
    params.filterByCircularity = False
    # params.minCircularity = 0.8
    # #params.maxCircularity = 1.2
    # Filter by Convexity
    params.filterByConvexity = False
    #params.minConvexity = 0.87
    #params.maxConvexity = 1
    # Filter by Inertia
    params.filterByInertia = False
    #params.minInertiaRatio = 0.01
    #params.maxInertiaRatio = 1
    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs- tags
    KP = detector.detect(img_bin)
    STATUS=''
    for KPi in KP:
        Y= KPi.pt[1]

        for lista_info in info_capsula_ordenada:
            POS_Y=int(lista_info[3])
            STATUS=lista_info[1]
            H1=int(lista_info[5])
            if STATUS=='':
                if abs(POS_Y-Y)<H1:
                    lista_info[1]='QUEBRADA'

    return(info_capsula_ordenada)

# Reversing a list using slicing technique
def Reverse(lst):
   new_lst = lst[::-1]

   return new_lst

def sobel(img): 
    (h,w) = img.shape
    fig_out = np.zeros((h,w), dtype = "uint8")

    m = 3
    d = int((m-1)/2) # N° de pixels a esquerda ou a direita do centro  
        
    kernel_sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype = "int16")
    kernel_sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype = "int16")

    # Varredura de imagem
    for i in range(d,h-d):
        for j in range(d,w-d):
            secao_img = img[i-d:i+d+1,j-d:j+d+1] # Extrai subset da matriz da vizinhança
            prod_img_x = kernel_sobel_x*secao_img
            prod_img_y = kernel_sobel_y*secao_img
            somatorio_x = prod_img_x.sum()
            somatorio_y = prod_img_y.sum()
            fig_out[i,j] = abs(somatorio_x) + abs(somatorio_y)
    return(fig_out)
