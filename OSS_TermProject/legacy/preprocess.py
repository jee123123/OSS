import cv2
import matplotlib.pyplot as plt
import argparse
import os

# 1. 터미널 명령행 인자(argparse) 설정
parser = argparse.ArgumentParser(description="Sheet Music Preprocessing Script")
parser.add_argument(
    "--image",
    type=str,
    default="sheet.jpg",  # 기본으로 불러올 이미지파일명
    help="Path to the sheet music image file"
)
args = parser.parse_args()
img_path = args.image

# 2. 이미지 불러오기 및 예외 처리
if not os.path.exists(img_path):
    print(f" Error: {img_path} 파일을 찾을 수 없습니다.")
    exit()

gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# 3. 악보 이진화
_, binary_img = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY_INV)

# 4. 결과 시각화
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

# 5. 이진화 이미지 저장
cv2.imwrite('binary_sheet.png', binary_img)
print(f" '{img_path}' 이진화 완료! 'binary_sheet.png'가 저장되었습니다.")