import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Aula10_selectBlob import*



# Carregar imagens - DESENVOLVIMENTO
cap_verde = cv2.imread("Projeto_01_Livia_Quezia/Imagens_DESENVOLVIMENTO/color1.png",cv2.IMREAD_COLOR)
cap_azul = cv2.imread("Projeto_01_Livia_Quezia/Imagens_DESENVOLVIMENTO/color3.png",cv2.IMREAD_COLOR)

cap_boa_1 = cv2.imread("Projeto_01_Livia_Quezia/Imagens_DESENVOLVIMENTO/_boa_01.png",cv2.IMREAD_COLOR)
cap_boa_2 = cv2.imread("Projeto_01_Livia_Quezia/Imagens_DESENVOLVIMENTO/_boa_02.png",cv2.IMREAD_COLOR)
cap_boa_3 = cv2.imread("Projeto_01_Livia_Quezia/Imagens_DESENVOLVIMENTO/_boa_02.png",cv2.IMREAD_COLOR)
cap_boa_4 = cv2.imread("Projeto_01_Livia_Quezia/Imagens_DESENVOLVIMENTO/_boa_02.png",cv2.IMREAD_COLOR)
cap_boa_5 = cv2.imread("Projeto_01_Livia_Quezia/Imagens_DESENVOLVIMENTO/_boa_02.png",cv2.IMREAD_COLOR)

cap_amassada_1 = cv2.imread("Projeto_01_Livia_Quezia/Imagens_DESENVOLVIMENTO/amassada1.png",cv2.IMREAD_COLOR)
cap_amassada_2 = cv2.imread("Projeto_01_Livia_Quezia/Imagens_DESENVOLVIMENTO/amassada2.png",cv2.IMREAD_COLOR)
cap_amassada_3 = cv2.imread("Projeto_01_Livia_Quezia/Imagens_DESENVOLVIMENTO/amassada3.png",cv2.IMREAD_COLOR)

cap_quebrada_1 = cv2.imread("Projeto_01_Livia_Quezia/Imagens_DESENVOLVIMENTO/quebrada_01.png",cv2.IMREAD_COLOR)
cap_quebrada_2 = cv2.imread("Projeto_01_Livia_Quezia/Imagens_DESENVOLVIMENTO/quebrada_01.png",cv2.IMREAD_COLOR)
cap_quebrada_3 = cv2.imread("Projeto_01_Livia_Quezia/Imagens_DESENVOLVIMENTO/quebrada_01.png",cv2.IMREAD_COLOR)


cap_riscada_1 = cv2.imread("Projeto_01_Livia_Quezia/Imagens_DESENVOLVIMENTO/riscada1.png",cv2.IMREAD_COLOR)
cap_riscada_2 = cv2.imread("Projeto_01_Livia_Quezia/Imagens_DESENVOLVIMENTO/riscada1.png",cv2.IMREAD_COLOR)


# Carregar imagens - TESTE_VALIDACAO
teste_0 = cv2.imread("Projeto_01_Livia_Quezia/Imagens_TESTE_VALIDACAO/Imagem_TESTE_0.png",cv2.IMREAD_COLOR)
validacao_1 = cv2.imread("Projeto_01_Livia_Quezia/Imagens_TESTE_VALIDACAO/Imagem_VALIDACAO_1.png",cv2.IMREAD_COLOR)
validacao_2 = cv2.imread("Projeto_01_Livia_Quezia/Imagens_TESTE_VALIDACAO/Imagem_VALIDACAO_2.png",cv2.IMREAD_COLOR)
validacao_3 = cv2.imread("Projeto_01_Livia_Quezia/Imagens_TESTE_VALIDACAO/Imagem_VALIDACAO_3.png",cv2.IMREAD_COLOR)


# Altere o número da fig 
img= validacao_3
img_gray = cv2.cvtColor(validacao_3, cv2.COLOR_BGR2GRAY)
img_bin=np.where(img_gray>160, 0, 255).astype(np.uint8)

#Realizando o Contorno
contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#contours = max(contours, key=lambda x: cv2.contourArea(x))
img_contour= cv2.cvtColor(img_bin,cv2.COLOR_GRAY2BGR)

angles=[]
imgs=[]

def tranformacao(a,b, c, d, e, f):
    matriz_transf=np.array([[a, b, c],[d, e, f],[0, 0, 1]], dtype=np.float32) #Terá cossenos, senos, numeros facionados...
    return matriz_transf

# #Para não deixar espaços vazios na imagem:  
def InverseMapping(img_in, TH, fundo):
    
    TH_inv = np.linalg.inv(TH)

    #(h, w, c)= img_in.shape
    (h, w, c)= img_in.shape
    img_out = np.ones((500, 500, c), dtype=np.uint8)*fundo

    #varredura da imagem
    for u in range(w): #--> varrendo as colunas 
        for v in range(h): #--> varrendo as linhas 
            p1=np.array([u, v, 1]).transpose() 
            p0=np.matmul(TH_inv,p1)
            x=int(p0[0]/p0[2])
            y=int(p0[1]/p0[2])

            if (x>=0) and (x<w) and (y>=0) and (y<h):
                #img_out[v,u] = img_in[y,x]
                img_out[v,u] = img_in[y,x]
                # img_out[v,u, 2] = img_in[y,x, 2]
                # img_out[v,u, 3] = img_in[y,x, 3] 

    return img_out

# def rotate_image(image, angle):
#   image_center = tuple(np.array(image.shape[1::-1]) / 2)
#   rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
#   result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
#   return result


def rotate_image(image, angle_deg):
    # Get image dimensions
    h, w = image.shape[:2]

    # Calculate the rotation matrix
    center = (w // 2, h // 2)

    #rotated = cv2.imutils.rotate(image, angle_deg)

    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)


    # Perform the rotation
    rotated_image = cv2.warpAffine(image, M, (w, h))
    #rotated_image =

    return rotated_image


i=0
for contour in contours:
    # get rotated rectangle from outer contour
    rotrect = cv2.minAreaRect(contour)
    i+=1
    angle = rotrect[-1]
    print(rotrect[1][1])

    # from https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)
    
    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle
    
    angles.append(angle)

    x,y,w,h = cv2.boundingRect(contour)
    cv2.rectangle(img_contour,(x,y),(x+w,y+h),(0,255,0),2)

    img_pill= img[y:y+h,x:x+w]
    imgs.append(img_pill)

    cv2.putText(img_contour, str(i) , (x-130, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),2,cv2.LINE_AA)

i_angle=0
imgs_out=[]

for img in imgs:

    (h,w,c)= img.shape
    # img_out=np.ones((500,500), dtype="uint8")*255
    angle= angles[i_angle]

    if abs(angle) != 90.0 and abs(angle) != 0.0:

        # img_mask= np.zeros((500, 500, 3), dtype=np.uint8)
        # #img= img+img_mask
        # np.where(img== [255,255,255], img_mask, img)
        # img_out= rotate_image(img, 90-angle)

        matriz_transla= tranformacao(1, 0, 100, 0,1, 100)
        rad= np.deg2rad(90-angle)
        matriz_rotacao= tranformacao(np.cos(rad), -np.sin(rad), 0, np.sin(rad), np.cos(rad), 0)
        matriz_transla_volta= tranformacao(1, 0, w/2, 0,1, h/2)
        #TH= np.matmul(matriz_transla_volta, matriz_rotacao, matriz_transla)
        TH= matriz_transla
        fundo= 255
        img_out= InverseMapping(img, TH, fundo)

    else:
        img_out= img.copy()

    i_angle+=1
    imgs_out.append(img_out)

plt.figure("Projeto")
plt.imshow(img_contour, cmap='gray')
plt.show()   






        # # get rotated rectangle from outer contour
        # rotrect = cv2.minAreaRect(contour)

        # angle = rotrect[-1]
        # print(rotrect[1][1])

        # # from https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
        # # the cv2.minAreaRect function returns values in the
        # # range [-90, 0); as the rectangle rotates clockwise the
        # # returned angle trends to 0 -- in this special case we
        # # need to add 90 degrees to the angle
        # if angle < -45:
        #     angle = -(90 + angle)
        
        # # otherwise, just take the inverse of the angle to make
        # # it positive
        # else:
        #     angle = -angle
        
        # angles.append(angle)