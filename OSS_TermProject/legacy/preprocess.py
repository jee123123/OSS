import cv2
import matplotlib.pyplot as plt
import argparse
import os

# 1. 스크립트(preprocess.py)가 위치한 디렉터리 경로 계산
script_dir = os.path.dirname(os.path.abspath(__file__))
default_img_path = os.path.join(script_dir, "sheet.png")

# 2. 터미널 명령행 인자(argparse) 설정
parser = argparse.ArgumentParser(description="Sheet Music Preprocessing Script")
parser.add_argument(
    "--image",
    type=str,
    default=default_img_path,  # 스크립트와 동일한 폴더의 sheet.jpg를 기본값으로 지정
    help="Path to the sheet music image file"
)
args = parser.parse_args()
img_path = args.image

# 사용자가 상대 경로를 직접 입력했을 경우를 대비해 스크립트 기준 경로 결합
if not os.path.isabs(img_path):
    img_path = os.path.join(script_dir, img_path)

# 3. 이미지 불러오기 및 예외 처리
if not os.path.exists(img_path):
    print(f" Error: {img_path} 파일을 찾을 수 없습니다.")
    exit()

gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# 4. 악보 이진화
_, binary_img = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY_INV)

# 5. 결과 시각화
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

# 6. 이진화 이미지 저장 (스크립트와 같은 폴더에 저장)
output_path = os.path.join(script_dir, 'binary_sheet.png')
cv2.imwrite(output_path, binary_img)
print(f" '{img_path}' 이진화 완료! '{output_path}'가 저장되었습니다.")