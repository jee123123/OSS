import cv2
import matplotlib.pyplot as plt

# 악보 이미지(그레이스케일)
img_path = 'sheet.png'
gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if gray_img is None:
    print(f" Error: {img_path} 파일을 찾을 수 없습니다.")
    exit()

#악보 이진화
_, binary_img = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY_INV)

#결과 시각화
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Original Grayscale")
plt.imshow(gray_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Crisp Sharp Binary Image")
plt.imshow(binary_img, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

cv2.imwrite('binary_sheet.png', binary_img)