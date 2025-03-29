import numpy as np 
import cv2
from matplotlib import pyplot as plt
import imutils
import easyocr
import os
import csv

# Configurar ruta de Tesseract (ajustar según el sistema)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows

def speed_zone(img):
    # Coordinates for speed display in a 1280x720p img
    poly = np.array([[984,63],[1014,463],[1255,477],[1252,67]], np.int32)
    poly = poly.reshape((-1, 1, 2))  # Ajustar la forma para OpenCV

    mask = np.zeros_like(img, dtype=np.uint8)

    cv2.fillPoly(mask, [poly], (255, 255, 255))

    res = cv2.bitwise_and(img, mask)

    x, y, w, h = cv2.boundingRect(poly)
    cut = res[y:y+h, x:x+w]

    cv2.imwrite("temp/speed_zone.jpg", cut)
    return cut

debug = False

speed = []

with os.scandir("frames") as files:
    for file in files:
        if file.is_file():
            print(f"Processing file: {file.name}")

            original = cv2.imread(f"frames/{file.name}")
            if debug:
                cv2.imwrite("temp/fr_16.png", original)
            img = speed_zone(original)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if debug:
                cv2.imwrite("temp/gray.png", cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))

            bfilter = cv2.bilateralFilter(gray, 11, 17, 17) # for noise
            edged = cv2.Canny(bfilter, 0, 200) #Edge detection
            if debug:
                cv2.imwrite("temp/edged.png", cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))

            kernel = np.ones((6, 9), np.uint8) 
            dilat = cv2.dilate(edged.copy(), kernel, iterations=2)
            if debug:
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
                if debug:
                    cv2.imwrite("temp/new_image.png", new_image)

            (x, y) = np.where(mask==255)
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))
            cropped_image = gray[x1:x2+1, y1:y2+1]
            if debug:
                cv2.imwrite("temp/cropped_image.png", cropped_image)
                cv2.imwrite("temp/new_image_color.png", cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

            reader = easyocr.Reader(['en'], gpu=True)
            results = reader.readtext(cropped_image, allowlist='0123456789')
            filtered_results = [
                {"value": res[1], "confidence": res[2], "frame": file.name}
                for res in results if res[2] > 0.8
            ]

            speed.append(filtered_results)

headers = ["value", "confidence", "frame"]

# Abrimos el archivo en modo escritura
with open("speed.csv", mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=headers)
    
    writer.writeheader()

    for result_list in speed:
        for result in result_list:  
            writer.writerow(result)

print("FILE GENERATED")

