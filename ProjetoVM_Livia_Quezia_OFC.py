import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Functions_OFC import*

# Carregar imagens - DESENVOLVIMENTO

# Cápsulas CORES ERRADAS
cap_verde = cv2.imread("C:/Users/liviabn\Documents/7SEM/Visao_de_Maquina/Exercicios/Projeto_01_Livia_Quezia/Imagens_DESENVOLVIMENTO/color1.png",cv2.IMREAD_COLOR)
cap_azul = cv2.imread("C:/Users/liviabn\Documents/7SEM/Visao_de_Maquina/Exercicios/Projeto_01_Livia_Quezia/Imagens_DESENVOLVIMENTO/color3.png",cv2.IMREAD_COLOR)

# Cápsulas BOAS
cap_boa_1 = cv2.imread("C:/Users/liviabn\Documents/7SEM/Visao_de_Maquina/Exercicios/Projeto_01_Livia_Quezia/Imagens_DESENVOLVIMENTO/_boa_01.png",cv2.IMREAD_COLOR)
cap_boa_2 = cv2.imread("C:/Users/liviabn\Documents/7SEM/Visao_de_Maquina/Exercicios/Projeto_01_Livia_Quezia/Imagens_DESENVOLVIMENTO/_boa_02.png",cv2.IMREAD_COLOR)
cap_boa_3 = cv2.imread("C:/Users/liviabn\Documents/7SEM/Visao_de_Maquina/Exercicios/Projeto_01_Livia_Quezia/Imagens_DESENVOLVIMENTO/_boa_02.png",cv2.IMREAD_COLOR)
cap_boa_4 = cv2.imread("C:/Users/liviabn\Documents/7SEM/Visao_de_Maquina/Exercicios/Projeto_01_Livia_Quezia/Imagens_DESENVOLVIMENTO/_boa_02.png",cv2.IMREAD_COLOR)
cap_boa_5 = cv2.imread("C:/Users/liviabn\Documents/7SEM/Visao_de_Maquina/Exercicios/Projeto_01_Livia_Quezia/Imagens_DESENVOLVIMENTO/_boa_02.png",cv2.IMREAD_COLOR)

# Cápsulas AMASSADAS
cap_amassada_1 = cv2.imread("C:/Users/liviabn\Documents/7SEM/Visao_de_Maquina/Exercicios/Projeto_01_Livia_Quezia/Imagens_DESENVOLVIMENTO/amassada1.png",cv2.IMREAD_COLOR)
cap_amassada_2 = cv2.imread("C:/Users/liviabn\Documents/7SEM/Visao_de_Maquina/Exercicios/Projeto_01_Livia_Quezia/Imagens_DESENVOLVIMENTO/amassada2.png",cv2.IMREAD_COLOR)
cap_amassada_3 = cv2.imread("C:/Users/liviabn\Documents/7SEM/Visao_de_Maquina/Exercicios/Projeto_01_Livia_Quezia/Imagens_DESENVOLVIMENTO/amassada3.png",cv2.IMREAD_COLOR)

# Cápsulas QUEBRADAS
cap_quebrada_1 = cv2.imread("C:/Users/liviabn\Documents/7SEM/Visao_de_Maquina/Exercicios/Projeto_01_Livia_Quezia/Imagens_DESENVOLVIMENTO/quebrada_01.png",cv2.IMREAD_COLOR)
cap_quebrada_2 = cv2.imread("C:/Users/liviabn\Documents/7SEM/Visao_de_Maquina/Exercicios/Projeto_01_Livia_Quezia/Imagens_DESENVOLVIMENTO/quebrada_01.png",cv2.IMREAD_COLOR)
cap_quebrada_3 = cv2.imread("C:/Users/liviabn\Documents/7SEM/Visao_de_Maquina/Exercicios/Projeto_01_Livia_Quezia/Imagens_DESENVOLVIMENTO/quebrada_01.png",cv2.IMREAD_COLOR)

# Cápsulas RISCADAS
cap_riscada_1 = cv2.imread("C:/Users/liviabn\Documents/7SEM/Visao_de_Maquina/Exercicios/Projeto_01_Livia_Quezia/Imagens_DESENVOLVIMENTO/riscada1.png",cv2.IMREAD_COLOR)
cap_riscada_2 = cv2.imread("C:/Users/liviabn\Documents/7SEM/Visao_de_Maquina/Exercicios/Projeto_01_Livia_Quezia/Imagens_DESENVOLVIMENTO/riscada1.png",cv2.IMREAD_COLOR)

# Carregar imagens - TESTE_VALIDACAO
teste_0 = cv2.imread("C:/Users/liviabn\Documents/7SEM/Visao_de_Maquina/Exercicios/Projeto_01_Livia_Quezia/Imagens_TESTE_VALIDACAO/Imagem_TESTE_0.png",cv2.IMREAD_COLOR)
validacao_1 = cv2.imread("C:/Users/liviabn\Documents/7SEM/Visao_de_Maquina/Exercicios/Projeto_01_Livia_Quezia/Imagens_TESTE_VALIDACAO/Imagem_VALIDACAO_1.png",cv2.IMREAD_COLOR)
validacao_2 = cv2.imread("C:/Users/liviabn\Documents/7SEM/Visao_de_Maquina/Exercicios/Projeto_01_Livia_Quezia/Imagens_TESTE_VALIDACAO/Imagem_VALIDACAO_2.png",cv2.IMREAD_COLOR)
validacao_3 = cv2.imread("C:/Users/liviabn\Documents/7SEM/Visao_de_Maquina/Exercicios/Projeto_01_Livia_Quezia/Imagens_TESTE_VALIDACAO/Imagem_VALIDACAO_3.png",cv2.IMREAD_COLOR)
validacoes= [validacao_1,validacao_2, validacao_3]

import tkinter as tk #fornece ferramentas para criar interfaces gráficas de usuário (GUI)
# Entra com imagem:
n= input("Escolha a imagem para ser analisada (1, 2 ou 3):")
img= validacoes[int(n)-1]

# Converter a imagem para o espaço de cores HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

(altura,largura,canais) = img.shape

# Definir intervalos de cor para o verde e azul em HSV
lower_green = np.array([20, 40, 40])
upper_green = np.array([80, 255, 255])

lower_blue = np.array([90, 90, 90])
upper_blue = np.array([130, 255, 255])

lower_red = np.array([0, 40, 40])
upper_red = np.array([10, 255, 255])

# Defina os limites mínimo e máximo para o tamanho do contorno
tamanho_minimo = 10  # Defina o tamanho mínimo desejado
tamanho_maximo = 1000  # Defina o tamanho máximo desejado

# Aplicar um limiar para segmentar as áreas verdes e azuis na imagem
mask_green = cv2.inRange(hsv, lower_green, upper_green)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
mask_red = cv2.inRange(hsv, lower_red, upper_red)

# Encontrar contornos nas áreas segmentadas
contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

info_capsula= find_all_pills(contours_red,contours_green,contours_blue, img_rgb)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_bin=np.where(img_gray>160, 0, 255).astype(np.uint8)

#Realizando o Contorno
contours_bin, _= cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

dimensions=[]
for contour in contours_bin:
    dimensions.append(dimensions_largest_contour(contour, img_rgb))

dimensions_ordenada1= Reverse(dimensions)
dimensions_ordenada= reorder(dimensions, 4)
info_capsula_ordenada= reorder(info_capsula, 2)

#ADICAO DO INDICE
i_order= 1
for lista_info in info_capsula_ordenada:
    lista_info.insert(0, i_order) #Adicionar na primeira posição
    i_order+=1

for i in range(0,len(dimensions_ordenada)):
    info_capsula_ordenada[i]+=dimensions_ordenada[i]

img_bin1=np.where(img_gray<40, 255, 0).astype(np.uint8)
#Realizando o Contorno
contours_black, _= cv2.findContours(img_bin1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

info_capsula_ordenada= mashed_classifier(info_capsula_ordenada, img_rgb,contours_black)
info_capsula_ordenada= cracked_classifier(img_bin, info_capsula_ordenada)

# Converter a imagem para o espaço de cores HSV
lower_red_limpa = np.array([0, 100, 40])
upper_red_limpa = np.array([10, 255, 255])
mask_red_limpa = cv2.inRange(hsv, lower_red_limpa, upper_red_limpa)

img_final_limpa = cv2.bitwise_and(img, img, mask = mask_red_limpa)
img_limpa_gray = cv2.cvtColor(img_final_limpa, cv2.COLOR_BGR2GRAY)

imgs_list, positions= cut_and_list_imgs(img_limpa_gray , contours_red)

imgs_out=[]
i_position=0
imgs_riscadas_pos=[]
riscada= False

for img_uma_pilula in imgs_list:

    (h,w) = img_uma_pilula.shape
    fig_out_sobel = np.zeros((h,w), dtype = "uint8")

    m = 3
    d = int((m-1)/2) # N° de pixels a esquerda ou a direita do centro  
        
    kernel_sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype = "int16")
    kernel_sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype = "int16")

    # Varredura de imagem
    for i in range(d,h-d):
        for j in range(d,w-d):
            secao_img = img_uma_pilula[i-d:i+d+1,j-d:j+d+1] # Extrai subset da matriz da vizinhança
            prod_img_x = kernel_sobel_x*secao_img
            prod_img_y = kernel_sobel_y*secao_img
            somatorio_x = prod_img_x.sum()
            somatorio_y = prod_img_y.sum()
            fig_out_sobel[i,j] = abs(somatorio_x) + abs(somatorio_y)

    for i in range(h):
        for j in range(w):
            # Verificar se o pixel é preto
            if img_uma_pilula[i,j] == 0:
                # Se sim, trocar para branco
                img_uma_pilula[i, j] = 255
            else:
                img_uma_pilula[i, j] = 0

    fig_bin_sobel = np.where((fig_out_sobel < 40) & ((fig_out_sobel >= 0)), 0, 255).astype(np.uint8)

    kernel_dilate_1 = np.ones((3,3), dtype = np.uint8)

    fig_subtrai_final = cv2.dilate(img_uma_pilula, kernel_dilate_1, iterations = 2) 

    for i in range(h):
        for j in range(w):
            # Verificar se o pixel é branco
            if fig_subtrai_final[i,j] == 255:
                # Se sim, trocar para preto no sobel
                fig_bin_sobel[i,j]=0
                
    bordas = np.count_nonzero(fig_bin_sobel)
    #print(bordas)

    imgs_out.append(fig_bin_sobel)

    if bordas > 150:
        #print(positions[i_position])
        X,Y=positions[i_position]
        imgs_riscadas_pos.append([X,Y])
    i_position+=1


for position in  imgs_riscadas_pos:
    for lista_info in info_capsula_ordenada:
        #columns=['Indice','STATUS', 'X', 'Y','H2', 'H1', 'W', 'C']) 
        STATUS=lista_info[1]
        POS_X=int(lista_info[2])
        POS_Y=int(lista_info[3])
        H1=int(lista_info[5])
        W=int(lista_info[6])
        X= position[0]
        Y= position[1]
        if STATUS=='':
            if abs(POS_Y-Y)<H1 and abs(POS_X-X)<W:
                lista_info[1]='RISCADA'

for lista_info in info_capsula_ordenada:
    STATUS=lista_info[1]
    if STATUS=='':
        lista_info[1]='OK'

relatorio_final=[]
for lista_info in info_capsula_ordenada:
    #columns=['Indice','STATUS', 'X', 'Y','H2', 'H1', 'W', 'C', 'POSX', 'POSY', 'angle']) 
    INDICE=int(lista_info[0])
    POS_Y=int(lista_info[3])
    POS_X=int(lista_info[2])
    STATUS=lista_info[1]
    H2=int(lista_info[4])
    H1=int(lista_info[5])
    W=int(lista_info[6])
    C=float(lista_info[7])
    X=int(lista_info[8])
    Y=int(lista_info[9])
    angle=int(lista_info[10])

    relatorio_list=[INDICE, STATUS, POS_X, POS_Y, W, H1, H2]
    relatorio_final.append(relatorio_list)
    #cv2.putText(img_rgb, texto , (x-130, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    texto = f"(Dados: ({STATUS},{POS_X},{POS_Y})"
    #texto = f"(Dados: {STATUS},{POS_X},{POS_Y}, {W}, {H1},{H2},{C})"
    #print(texto)
    cv2.putText(img_rgb, texto , (POS_X-250, POS_Y-H2-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    #cv2.putText(img_rgb, texto , (130,10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


columns_name=['Index', 'Status', 'Pos X', 'Pos Y', 'W', 'H1', 'H2']
#Instalar pip install openpyxl caso dê erro :)
df = pd.DataFrame(np.array(relatorio_final), columns= columns_name)
df.to_excel('Relatorio_Antibióticos'+ str(n)+'.xlsx', sheet_name='validacao_'+str(n))

plt.figure("Projeto")
plt.imshow(img_rgb, cmap='gray')
plt.show()

# print(df)

