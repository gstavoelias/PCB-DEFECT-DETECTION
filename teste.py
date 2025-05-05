import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class PCBInspectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PCB Inspector - Image Comparison")
        self.ref_image = None
        self.new_image = None
        self.toggle = True
        self.max_w, self.max_h = 800, 500  # Tamanho máximo da imagem exibida

        self.canvas = tk.Label(root)
        self.canvas.pack()

        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack()

        tk.Button(self.btn_frame, text="Carregar Referência", command=self.load_ref_image).grid(row=0, column=0)
        tk.Button(self.btn_frame, text="Carregar Imagem Nova", command=self.load_new_image).grid(row=0, column=1)
        tk.Button(self.btn_frame, text="Alternar", command=self.toggle_images).grid(row=0, column=2)
        tk.Button(self.btn_frame, text="Inspecionar", command=self.inspect_images).grid(row=0, column=3)

    def load_ref_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.ref_image = cv2.imread(path)
            self.display_image(self.ref_image)

    def load_new_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.new_image = cv2.imread(path)
            self.display_image(self.new_image)

    def toggle_images(self):
        if self.ref_image is not None and self.new_image is not None:
            self.toggle = not self.toggle
            img = self.ref_image if self.toggle else self.new_image
            self.display_image(img)

    def inspect_images(self):
        if self.ref_image is None or self.new_image is None:
            return

        gray_ref = cv2.cvtColor(self.ref_image, cv2.COLOR_BGR2GRAY)
        gray_new = cv2.cvtColor(self.new_image, cv2.COLOR_BGR2GRAY)

        # Reduz ruído
        gray_ref = cv2.GaussianBlur(gray_ref, (5, 5), 0)
        gray_new = cv2.GaussianBlur(gray_new, (5, 5), 0)

        aligned = self.align_images(gray_ref, gray_new)

        diff = cv2.absdiff(gray_ref, aligned)
        _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result_img = self.new_image.copy()
        for cnt in contours:
            if cv2.contourArea(cnt) > 50:
                x, y, w, h = cv2.boundingRect(cnt)
                center = (x + w // 2, y + h // 2)
                radius = max(w, h) // 2
                cv2.circle(result_img, center, radius, (0, 0, 255), 2)

        self.display_image(result_img)

    def align_images(self, ref, img):
        orb = cv2.ORB_create(500)
        kp1, des1 = orb.detectAndCompute(ref, None)
        kp2, des2 = orb.detectAndCompute(img, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 4:
            return img

        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = ref.shape
        aligned = cv2.warpPerspective(img, M, (w, h))
        return aligned

    def display_image(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_rgb)

        # Redimensiona proporcionalmente com limite de largura e altura
        w, h = im_pil.size
        scale = min(self.max_w / w, self.max_h / h, 1.0)
        new_size = (int(w * scale), int(h * scale))
        im_pil = im_pil.resize(new_size, Image.LANCZOS)

        imgtk = ImageTk.PhotoImage(image=im_pil)
        self.canvas.imgtk = imgtk
        self.canvas.configure(image=imgtk)

# Executa a GUI local
if __name__ == "__main__":
    root = tk.Tk()
    app = PCBInspectorGUI(root)
    root.mainloop()
