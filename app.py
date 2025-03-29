import numpy as np 
import cv2
from matplotlib import pyplot as plt
import imutils
import easyocr
import os

# Configurar ruta de Tesseract (ajustar según el sistema)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows

def speed_zone(img):
    # Definir las coordenadas del polígono
    poly = np.array([[1454,57],[1458,570],[1595,578],[1599,57]], np.int32)
    poly = poly.reshape((-1, 1, 2))  # Ajustar la forma para OpenCV

    # Crear una máscara negra
    mask = np.zeros_like(img, dtype=np.uint8)

    # Dibujar el polígono en la máscara en blanco
    cv2.fillPoly(mask, [poly], (255, 255, 255))

    # Aplicar la máscara sobre la img
    res = cv2.bitwise_and(img, mask)

    # Encontrar el bounding box del polígono para recortar solo esa región
    x, y, w, h = cv2.boundingRect(poly)
    cut = res[y:y+h, x:x+w]

    # Guardar o mostrar el resultado
    cv2.imwrite("temp/speed_zone.jpg", cut)
    return cut

original = cv2.imread("frames/frame_16.jpg")
cv2.imwrite("temp/fr_16.png", original)
img = speed_zone(original)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("temp/gray.png", cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))

bfilter = cv2.bilateralFilter(gray, 11, 17, 17) # for noise
edged = cv2.Canny(bfilter, 0, 200) #Edge detection
cv2.imwrite("temp/edged.png", cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))

kernel = np.ones((6, 9), np.uint8) 
dilat = cv2.dilate(edged.copy(), kernel, iterations=2)
cv2.imwrite("temp/expanded.png", dilat)

keypoints = cv2.findContours(dilat.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse = True)[:10]
locations = []
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if(len(approx) > 4):
        locations.append(approx)

mask = np.zeros(gray.shape, np.uint8)
for location in locations:
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite("temp/new_image.png", new_image)

(x, y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]
cv2.imwrite("temp/cropped_image.png", cropped_image)
cv2.imwrite("temp/new_image_color.png", cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

reader = easyocr.Reader(['en'], gpu=True)
results = reader.readtext(cropped_image, allowlist='0123456789')
filtered_results = [res for res in results if res[2] > 0.8]

print(filtered_results)

