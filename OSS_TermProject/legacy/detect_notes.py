import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os

#파일 불러오기
parser = argparse.ArgumentParser(description="OMR Sheet Note Detection Script")
parser.add_argument(
    "--image",
    type=str,
    default="sheet.png",
    help="Path to the sheet music image file",
)
args = parser.parse_args()

image_path = args.image

# 이미지 파일 존재 여부 확인
if not os.path.exists(image_path):
    print(f" Error: File '{image_path}' not found.")
    exit()

# 이미지 및 이전 단계 결과물 불러오기
original_sheet = cv2.imread(image_path)
binary_img = cv2.imread('binary_sheet.png', cv2.IMREAD_GRAYSCALE)   # 이진화 악보
staff_lines = cv2.imread('staff_lines.png', cv2.IMREAD_GRAYSCALE)   # 오선 이미지

if binary_img is None or staff_lines is None or original_sheet is None:
    print(" Error: 필요한 이미지 파일(binary_sheet.png, staff_lines.png 등)을 찾을 수 없습니다.")
    exit()

img_h, img_w = binary_img.shape

row_sum = np.sum(staff_lines > 0, axis=1)
# 오선 범위
line_rows = np.where((row_sum > img_w * 0.4) & (row_sum < img_w * 0.95))[0]

# 연속된 행을 하나의 줄로 합침
lines_y = []
group = [line_rows[0]]
for r in line_rows[1:]:
    if r - group[-1] <= 1:
        group.append(r)
    else:
        lines_y.append(np.mean(group))
        group = [r]
lines_y.append(np.mean(group))

# 줄 간격으로 5개씩 시스템으로 묶기
gaps = np.diff(lines_y)
staff_space = float(np.median(gaps[gaps < np.median(gaps) * 1.8]))
print(f" 기준 오선 간격(Staff Space S): {staff_space:.2f} px")

systems = []      
cur = [lines_y[0]]
for y in lines_y[1:]:
    if y - cur[-1] < staff_space * 1.8:
        cur.append(y)
    else:
        systems.append(cur)
        cur = [y]
systems.append(cur)
systems = [s for s in systems if len(s) == 5]

S = staff_space
print(f" 검출된 오선 시스템: {len(systems)}개")

#음표머리 추출
kw = max(3, int(round(S * 0.8)) | 1)
kh = max(3, int(round(S * 0.6)) | 1)
head_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh))
heads_only = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, head_kernel)

#기둥 확인함수
def stem_length(x, y, w, h):
    best = 0
    cy = y + h // 2
    for cx in (x - 1, x, x + w - 1, x + w):         # 머리의 좌/우 가장자리 열
        for cxx in (cx - 1, cx, cx + 1):
            if not (0 <= cxx < img_w):
                continue
            for step in (-1, 1):
                n = 0
                yy = cy
                while 0 <= yy < img_h and binary_img[yy, cxx] > 0:
                    n += 1
                    yy += step
                best = max(best, n)
    return best

note_boxes = []   # (x, y, w, h, system_idx, step)

for si, lines in enumerate(systems):
    top, bottom = lines[0], lines[-1]

    y1 = int(max(0, top - S * 1.4))
    y2 = int(min(img_h, bottom + S * 1.9))

    xs = np.where(staff_lines[int(top):int(bottom) + 1, :].sum(axis=0) > 0)[0]
    sx0, sx1 = int(xs.min()), int(xs.max())

    band = binary_img[y1:y2, sx0:sx0 + int(S * 14)].copy()
    # 오선 제거한 상태로 분석
    band_no_staff = cv2.subtract(band, staff_lines[y1:y2, sx0:sx0 + int(S * 14)])
    n, lab, stats, _ = cv2.connectedComponentsWithStats(band_no_staff, connectivity=8)
    header_end = sx0
    for i in range(1, n):
        bx, by, bw, bh, area = stats[i]
        if bh >= S * 4.0 and bx < S * 4:            # 높은음자리표
            header_end = max(header_end, sx0 + bx + bw)
    header_end += int(S * 0.3)

   
    roi = np.zeros_like(heads_only)
    roi[y1:y2, header_end:sx1 - 2] = 255
    cand = cv2.bitwise_and(heads_only, roi)

    n, lab, stats, cents = cv2.connectedComponentsWithStats(cand, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]

        if not (S * 0.6 <= w <= S * 2.0 and S * 0.5 <= h <= S * 1.6):
            continue
        # 속이 꽉 찬 타원인지
        if area / float(w * h) < 0.55:
            continue
        # 기둥이 있어야 음표
        if stem_length(x, y, w, h) < S * 2.0:
            continue

        cy = y + h / 2.0
        local_S = (bottom - top) / 4.0          # 이 시스템의 실제 오선 간격
        step = int(round((bottom - cy) / (local_S / 2.0)))
        note_boxes.append((x, y, w, h, si, step))

# 같은 음표가 두 번 잡힌 경우 제거
note_boxes.sort(key=lambda b: (b[4], b[0]))
dedup = []
for b in note_boxes:
    if dedup and b[4] == dedup[-1][4] and abs(b[0] - dedup[-1][0]) < S * 0.8 \
            and abs(b[1] - dedup[-1][1]) < S * 0.8:
        continue
    dedup.append(b)
note_boxes = dedup

#결과
NAMES = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
def step_to_name(step):
    idx = 2 + step                      # E = 2
    octave = 4 + idx // 7
    return f"{NAMES[idx % 7]}{octave}"

for (x, y, w, h, si, step) in note_boxes:
    cv2.rectangle(original_sheet, (x - 3, y - 3), (x + w + 3, y + h + 3), (0, 255, 0), 1)
    cv2.putText(original_sheet, step_to_name(step), (x - 4, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1)

print(f"🎯 최종 음표 추출 완료: 총 {len(note_boxes)}개")
for si in range(len(systems)):
    seq = [step_to_name(b[5]) for b in note_boxes if b[4] == si]
    print(f"  {si + 1}번째 줄 ({len(seq)}개): {' '.join(seq)}")

#시각화
cv2.imwrite('detected_notes_perfect.png', original_sheet)
try:
    plt.figure(figsize=(12, 8))
    plt.title(f"Note Detection (Total Notes = {len(note_boxes)})")
    plt.imshow(cv2.cvtColor(original_sheet, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
except Exception:
    pass

# 다음 단계(MIDI 변환)에서 쓸 수 있도록 좌표 저장: x, y, w, h, 줄번호, 오선위치
np.save('note_boxes.npy', np.array(note_boxes))