import cv2
import pytesseract
import numpy as np

# Configurar ruta de Tesseract (ajustar según el sistema)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows

def relevant_zone(img):
    # Definir las coordenadas del polígono
    poly = np.array([[1243, 572], [1592, 579], [1596, 72], [1237, 69]], np.int32)
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
    cv2.imwrite("temp/relevant_zone.jpg", cut)
    return cut

def rotate_image_no_crop(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Calcular la nueva dimensión de la imagen después de la rotación
    angle_rad = np.radians(angle)
    new_w = int(abs(h * np.sin(angle_rad)) + abs(w * np.cos(angle_rad)))
    new_h = int(abs(h * np.cos(angle_rad)) + abs(w * np.sin(angle_rad)))

    # Crear nueva matriz de rotación con traslación para centrar
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    # Aplicar la transformación con el nuevo tamaño
    rotated_image = cv2.warpAffine(image, M, (new_w, new_h))

    return rotated_image

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Cargar la imagen
image = cv2.imread("img.jpg")

relevant_img = relevant_zone(image)

relevant_img = cv2.GaussianBlur(relevant_img, (3,3), 0)
kernel = np.ones((2,2), np.uint8)
img_proc = cv2.morphologyEx(relevant_img, cv2.MORPH_OPEN, kernel)

rotated_img = rotate_image_no_crop(img_proc, -6)

gray_img = grayscale(rotated_img)
cv2.imwrite("temp/gray.jpg", gray_img)

inv_img = cv2.bitwise_not(gray_img)
cv2.imwrite("temp/inv_img.jpg", inv_img)

im_bw = cv2.adaptiveThreshold(inv_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
cv2.imwrite("temp/bw_image.jpg", im_bw)

# Opciones de configuración de Tesseract
custom_config = r'--oem 3 --psm 6 outputbase digits'  # Solo números

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
ocr_result = pytesseract.image_to_string(im_bw, config=custom_config)
print(ocr_result)
