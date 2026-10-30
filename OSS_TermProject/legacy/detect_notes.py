import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os

# ─────────────────────────────────────────────
# 1. 터미널 명령행 인자(argparse) 설정 및 파일 불러오기
# ─────────────────────────────────────────────
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

# 원본 이미지(JPG, PNG, BMP 등) 및 이전 단계 결과물 불러오기
original_sheet = cv2.imread(image_path)
binary_img = cv2.imread('binary_sheet.png', cv2.IMREAD_GRAYSCALE)   # 이진화 악보
staff_lines = cv2.imread('staff_lines.png', cv2.IMREAD_GRAYSCALE)   # 오선 이미지

if binary_img is None or staff_lines is None or original_sheet is None:
    print(" Error: 필요한 이미지 파일(binary_sheet.png, staff_lines.png 등)을 찾을 수 없습니다.")
    exit()

img_h, img_w = binary_img.shape

# ─────────────────────────────────────────────
# 2. 오선 시스템(5줄 1세트) 검출 + 오선 간격 S 계산
#    - 줄 하나하나의 Y 위치를 구해서 5개씩 묶는다.
#    - 음표/가사의 위치는 모두 "오선 기준 상대 좌표"로 판단한다.
# ─────────────────────────────────────────────
row_sum = np.sum(staff_lines > 0, axis=1)
# 폭의 40%~95%만 오선으로 인정 (95% 이상은 악보 테두리선)
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

systems = []      # 각 시스템: dict(lines=[y1..y5], x0, x1)
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

# ─────────────────────────────────────────────
# 3. 음표 머리(notehead) 후보 추출
#    - 오선(1px), 기둥(1~2px), 글자 획은 얇다.
#    - 속이 찬 타원 커널로 Opening 하면 "속이 찬 음표 머리"만 남는다.
#    → 음표를 통째로(기둥+꼬리) 잡던 기존 방식의 누락/오검출 원인을 제거
# ─────────────────────────────────────────────
kw = max(3, int(round(S * 0.8)) | 1)
kh = max(3, int(round(S * 0.6)) | 1)
head_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh))
heads_only = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, head_kernel)

# ─────────────────────────────────────────────
# 4. 세로 기둥(stem) 확인용 함수
#    음표 머리 좌/우 가장자리에서 위/아래로 이어지는 흰 픽셀의 길이를 잰다.
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# 5. 시스템별로 음표 머리 검출
# ─────────────────────────────────────────────
note_boxes = []   # (x, y, w, h, system_idx, step)

for si, lines in enumerate(systems):
    top, bottom = lines[0], lines[-1]

    # (a) 세로 ROI: 덧줄 음표는 허용, 코드명(위)/가사(아래)는 제외
    y1 = int(max(0, top - S * 1.4))
    y2 = int(min(img_h, bottom + S * 1.9))

    # (b) 가로 ROI: 이 시스템 오선의 좌우 끝 안쪽
    xs = np.where(staff_lines[int(top):int(bottom) + 1, :].sum(axis=0) > 0)[0]
    sx0, sx1 = int(xs.min()), int(xs.max())

    # (c) 머리부(음자리표/조표/박자표) 끝 위치를 시스템마다 자동 계산
    #     오선 시작점에서 "처음으로 나오는 세로 마디선/음표" 전까지가 머리부
    #     → 음자리표 같은 큰 덩어리(높이 > 4S)의 오른쪽 끝을 기준으로 삼는다.
    band = binary_img[y1:y2, sx0:sx0 + int(S * 14)].copy()
    # 오선 제거한 상태로 연결 요소 분석
    band_no_staff = cv2.subtract(band, staff_lines[y1:y2, sx0:sx0 + int(S * 14)])
    n, lab, stats, _ = cv2.connectedComponentsWithStats(band_no_staff, connectivity=8)
    header_end = sx0
    for i in range(1, n):
        bx, by, bw, bh, area = stats[i]
        if bh >= S * 4.0 and bx < S * 4:            # 높은음자리표
            header_end = max(header_end, sx0 + bx + bw)
    header_end += int(S * 0.3)

    # (d) 조표(♭)와 박자표(2/4)는 첫 음표 앞에 있으므로, 머리 검출 + 기둥 검사로 걸러냄
    roi = np.zeros_like(heads_only)
    roi[y1:y2, header_end:sx1 - 2] = 255
    cand = cv2.bitwise_and(heads_only, roi)

    n, lab, stats, cents = cv2.connectedComponentsWithStats(cand, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]

        # 음표 머리 크기 검사 (오선 간격 S 기준)
        if not (S * 0.6 <= w <= S * 2.0 and S * 0.5 <= h <= S * 1.6):
            continue
        # 속이 꽉 찬 타원인지 (bbox 대비 면적 비율)
        if area / float(w * h) < 0.55:
            continue
        # 기둥이 있어야 음표 (조표/박자표/글자 제거)
        if stem_length(x, y, w, h) < S * 2.0:
            continue

        # 오선 기준 위치: 맨 아래 줄 = 0, 한 칸(줄↔칸)마다 +1
        cy = y + h / 2.0
        local_S = (bottom - top) / 4.0          # 이 시스템의 실제 오선 간격
        step = int(round((bottom - cy) / (local_S / 2.0)))
        note_boxes.append((x, y, w, h, si, step))

# 같은 음표가 두 번 잡힌 경우 제거 (중심이 매우 가까운 것)
note_boxes.sort(key=lambda b: (b[4], b[0]))
dedup = []
for b in note_boxes:
    if dedup and b[4] == dedup[-1][4] and abs(b[0] - dedup[-1][0]) < S * 0.8 \
            and abs(b[1] - dedup[-1][1]) < S * 0.8:
        continue
    dedup.append(b)
note_boxes = dedup

# ─────────────────────────────────────────────
# 6. 결과 출력 (오선 위치 → 음 이름, 높은음자리표 기준: 맨 아래 줄 = E4)
#    조표(♭)는 아직 반영하지 않은 '계이름 후보'입니다.
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# 7. 시각화 및 저장
# ─────────────────────────────────────────────
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