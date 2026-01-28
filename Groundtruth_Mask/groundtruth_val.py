import cv2
import numpy as np
import os

#Konfigurasi Path
base_folder = r"C:\Riset Skripsi\DATASET_ASLI\Val_T"
output_folder = r"C:\Riset Skripsi\DATASET_ASLI\Mask_GT_Val"

#Proses masing-masing subfolder
for subfolder in ['Ferrimagnetic', 'Ferromagnetic', 'Non_Magnetic']:
    folder_path = os.path.join(base_folder, subfolder)
    output_subfolder = os.path.join(output_folder, subfolder)
    os.makedirs(output_subfolder, exist_ok=True)

    if not os.path.exists(folder_path):
        print(f"[SKIP] Folder tidak ditemukan: {folder_path}")
        continue

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    print(f"\n==== Memproses folder: {subfolder} ({len(image_files)} file) ====")

    for image_file in image_files:
        img_path = os.path.join(folder_path, image_file)
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img_gray is None:
            print(f"[ERROR] Tidak dapat membaca gambar: {img_path}")
            continue

        img_inv = cv2.bitwise_not(img_gray)
        blur = cv2.GaussianBlur(img_inv, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.bitwise_not(mask)
        output_path = os.path.join(output_subfolder, f"mask_{image_file}")
        success = cv2.imwrite(output_path, mask)

        if success:
            print(f"[SAVED] {output_path}")
        else:
            print(f"[FAILED SAVE] {output_path}")
