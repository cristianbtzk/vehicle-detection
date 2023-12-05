import os
import re
import cv2
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt

import sys

args = sys.argv[1:]

plot = args[0] == 'plotImages'

frames = os.listdir('frames/')

# Ordenação dos frames
frames.sort(key=lambda f: int(re.sub('\D', '', f)))

imagens=[]

for i in frames:
    # Lê os frames
    img = cv2.imread('./frames/'+i)
    if np.shape(img) == ():
        print('./frames/'+i)
    # Adiciona na lista
    imagens.append(img)

i = 13

for frame in [i, i+1]:
    plt.imshow(cv2.cvtColor(imagens[frame], cv2.COLOR_BGR2RGB))
    plt.title("frame: "+str(frame))
    if plot:
      plt.show()

# Coverte os frames para escala de cinza

grayA = cv2.cvtColor(imagens[i], cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imagens[i+1], cv2.COLOR_BGR2GRAY)

# Imagem mostrando a diferença
plt.imshow(cv2.absdiff(grayB, grayA), cmap = 'gray')
if plot:
    plt.title("Diferença entre as imagens")
    plt.show()

imagemDiff = cv2.absdiff(grayB, grayA)

# Limiarização
ret, thresh = cv2.threshold(imagemDiff, 30, 255, cv2.THRESH_BINARY)

# Mostra a imagem após aplicar o limiar
plt.imshow(thresh, cmap = 'gray')
if plot:
    plt.title("Aplicação do limiar")
    plt.show()

# Dilatação
kernel = np.ones((3,3),np.uint8)
imagemDilatada = cv2.dilate(thresh,kernel,iterations = 1)

# Mostra a imagem dilatada
plt.imshow(imagemDilatada, cmap = 'gray')
if plot:
    plt.title("Imagem após dilatação")  
    plt.show()

# Exibe a zona de detecção
cv2.line(imagemDilatada, (0, 80),(256,80),(100, 0, 0))
plt.imshow(imagemDilatada)
if plot:
      plt.title("Zona de detecção")  
      plt.show()

# Identifica bordas
bordas, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

bordasValidas = []

for i,borda in enumerate(bordas):
    x,y,w,h = cv2.boundingRect(borda)
    # Verifica se a coordenada x é menor do que 200, y maior do que 80 e a área do contorno maior do que 25
    if (x <= 200) & (y >= 80) & (cv2.contourArea(borda) >= 25):
        bordasValidas.append(borda)

# Conta contornos válidos       
print(len(bordasValidas))

dmy = imagens[13].copy()

cv2.drawContours(dmy, bordasValidas, -1, (127,200,0), 2)
cv2.line(dmy, (0, 80),(256,80),(100, 255, 255))
plt.imshow(dmy)
if plot:
      plt.title("Exibe os contornos na imagem original")  
      plt.show()



# Fonte
font = cv2.FONT_HERSHEY_SIMPLEX

# Diretório para salvar os frames com bordas
path = "./frames_borda/"

if not os.path.isdir(path):
  os.mkdir(path)

for i in range(len(imagens)-1):
    
    grayA = cv2.cvtColor(imagens[i], cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imagens[i+1], cv2.COLOR_BGR2GRAY)
    imagemDiff = cv2.absdiff(grayB, grayA)
    
    ret, thresh = cv2.threshold(imagemDiff, 30, 255, cv2.THRESH_BINARY)
    
    imagemDilatada = cv2.dilate(thresh,kernel,iterations = 1)
    
    bordas, hierarchy = cv2.findContours(imagemDilatada.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    bordasValidas = []
    for borda in bordas:
        x,y,w,h = cv2.boundingRect(borda)
        if (x <= 200) & (y >= 80) & (cv2.contourArea(borda) >= 25):
            if (y >= 90) & (cv2.contourArea(borda) < 40):
                break
            bordasValidas.append(borda)
            
    dmy = imagens[i].copy()
    cv2.drawContours(dmy, bordasValidas, -1, (127,200,0), 2)
    
    cv2.putText(dmy, "Veículos detectados: " + str(len(bordasValidas)), (55, 15), font, 0.6, (0, 180, 0), 2)
    cv2.line(dmy, (0, 80),(256,80),(100, 255, 255))
    cv2.imwrite(path+str(i)+'.png',dmy)  


pathOut = 'deteccao-veiculos.mp4'

fps = 14.0

frames = []
arquivos = [f for f in os.listdir(path) if isfile(join(path, f))]

arquivos.sort(key=lambda f: int(re.sub('\D', '', f)))

for i in range(len(arquivos)):
    filename=path + arquivos[i]
    
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    
    frames.append(img)

# Monta um vídeo
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(len(frames)):
    out.write(frames[i])

out.release()