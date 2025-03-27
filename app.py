import cv2
import pytesseract
from PIL import Image

# Configurar ruta de Tesseract (ajustar según el sistema)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Cargar la imagen
image = cv2.imread("img.jpg")

gray_img = grayscale(image)
cv2.imwrite("temp/gray.jpg", gray_img)

thresh, im_bw = cv2.threshold(gray_img, 200, 230, cv2.THRESH_BINARY)
cv2.imwrite("temp/bw_imagev2.jpg", im_bw)


pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
ocr_result = pytesseract.image_to_string(im_bw)
print(ocr_result)
