import cv2
import numpy as np
import matplotlib.pyplot as plt

binary_img = cv2.imread('binary_sheet.png', cv2.IMREAD_GRAYSCALE)

if binary_img is None:
    print(" Error: binary_sheet.png 파일이 없습니다. preprocess.py를 먼저 실행하세요.")
    exit()

img_width = binary_img.shape[1]
kernel_length = int(img_width / 30)  
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

# 오선 추출
staff_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, horizontal_kernel)

# 이진화 이미지에서 오선 제거
notes_only = cv2.subtract(binary_img, staff_lines)

# 결과 확인 시각화
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.title("1. Binary Image")
plt.imshow(binary_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("2. Extracted Staff Lines")
plt.imshow(staff_lines, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("3. Notes Only (Staff Removed)")
plt.imshow(notes_only, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# 오선이 제거된 이미지 저장
cv2.imwrite('notes_only.png', notes_only)
cv2.imwrite('staff_lines.png', staff_lines)