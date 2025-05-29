from ultralytics import YOLO
import cv2

class Component:
    def __init__(self, id, x1, y1, x2, y2):
        self.cls_id = id
        self.x1 = x1 
        self.y1 = y1 
        self.x2 = x2 
        self.y2 = y2
        self.status = "NÂO ENCONTRADO"

# BATERIA RTC
BH1 = Component(0, 618, 10, 727, 81)

#CAPACITOR_D
C41 = Component(2, 531, 74, 580, 126)

#CAPACITOR_L
C63 = Component(3, 684, 624, 742, 679)

#CAPACITOR_R
C42 = Component(4, 435, 74, 486, 122)
C30 = Component(4, 413, 139, 472, 197)
C32 = Component(4, 737, 435, 788, 483)

#CAPACITOR_U
C77 = Component(5, 819, 361, 870, 414)


components = [BH1, C41, C63, C42, C30, C32, C77]
classes = [
"Bateria RTC",
"Bateria RTC Invertida",
"Capacitor_D",
"Capacitor_L",
"Capacitor_R",
"Capacitor_U",
"Opto",
"Trimmer",
"Capacitor"
]

model = YOLO("my_model.pt")

path = "REGULAR.jpeg"
image = cv2.imread(path)
boxes = model(path)[0].boxes.numpy()

for box in boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    center_x = x1 + (x2-x1)/2
    center_y = y1 + (y2-y1)/2
    for component in components:
        if component.x1 < center_x < component.x2 and component.y1 < center_y < component.y2:
            component.status = "OK" if box.cls == component.cls_id else "Componente Invertido"
            break

for component in components:
    if component.status == "OK":
        continue
    x1, y1, x2, y2 = (component.x1, component.y1, component.x2, component.y2)
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    cv2.putText(image, classes[int(component.cls_id)], (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    

image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
cv2.imshow("Image", image)
cv2.waitKey(0)