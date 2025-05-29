from skimage.metrics import structural_similarity as ssim
import cv2

img1 = cv2.imread("RTC UP INV.jpeg")
img2 = cv2.imread("RTC UP.jpeg")

# Redimensionar para reduzir ruído visual
img1 = cv2.resize(img1, (0, 0), fx=0.25, fy=0.25)
img2 = cv2.resize(img2, (0, 0), fx=0.25, fy=0.25)

# Converter para cinza
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Calcular SSIM e mapa de diferença
score, diff = ssim(gray1, gray2, full=True)
diff = (1 - diff) * 255
diff = diff.astype("uint8")

# Threshold da diferença
_, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
marked = img2.copy()

for cnt in contours:
    if cv2.contourArea(cnt) > 300:  # ignora ruído
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(marked, (x, y), (x+w, y+h), (0, 0, 255), 2)


# Mostrar resultados
cv2.imshow("Diferença", diff)
cv2.imshow("Threshold", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
