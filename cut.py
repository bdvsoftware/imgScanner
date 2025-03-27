import cv2
import numpy as np

# Cargar la imagen
imagen = cv2.imread("img.jpg")

# Definir las coordenadas del polígono
poligono = np.array([[1243, 572], [1592, 579], [1596, 72], [1237, 69]], np.int32)
poligono = poligono.reshape((-1, 1, 2))  # Ajustar la forma para OpenCV

# Crear una máscara negra
mascara = np.zeros_like(imagen, dtype=np.uint8)

# Dibujar el polígono en la máscara en blanco
cv2.fillPoly(mascara, [poligono], (255, 255, 255))

# Aplicar la máscara sobre la imagen
resultado = cv2.bitwise_and(imagen, mascara)

# Encontrar el bounding box del polígono para recortar solo esa región
x, y, w, h = cv2.boundingRect(poligono)
recorte = resultado[y:y+h, x:x+w]

# Guardar o mostrar el resultado
cv2.imwrite("recorte.png", recorte)
cv2.imshow("Recorte", recorte)
cv2.waitKey(0)
cv2.destroyAllWindows()
