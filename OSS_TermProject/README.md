# Score2MIDI

악보 이미지에서 음악 정보를 읽어 MIDI로 변환하는 오픈소스 프로젝트입니다.

## 개발 순서

1. 악보 이미지 전처리와 오선 검출
2. 오선 기준으로 음표 머리 위치 찾기
3. 음자리표·조표·박자에 따라 음표 해석
4. 여러 마디와 성부를 담는 중간 악보 구조 정의
5. MIDI 내보내기와 인식 결과 수정 기능

첫 실습은 인쇄된 악보 이미지에서 오선 다섯 줄을 찾아 결과 이미지를 저장하는 것입니다. 이 단계에서는 아직 음표를 읽거나 MIDI를 만들지 않습니다.

## 설치

Python 3.10 이상 권장.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 오선 검출 실행

```bash
python -m score2midi.staff_detect path/to/score.png --output work/staff-overlay.png
```

지원 입력은 우선 PNG/JPEG 같은 이미지입니다. PDF 지원은 이미지 처리 흐름이 안정된 다음 추가합니다.

## 현재 제한

- 수평으로 놓인 인쇄 악보를 대상으로 합니다.
- 손글씨, 기울어짐이 큰 스캔, 복잡한 배경은 아직 지원하지 않습니다.
- 오선 검출은 악보 인식 파이프라인의 첫 단계이며, 음표 판독 정확도를 뜻하지 않습니다.

## 기여

이 프로젝트는 수업 과제의 오픈소스 개발 과정을 위해 공개적으로 개선하는 것을 목표로 합니다. 라이선스와 기여 안내는 프로젝트 범위를 정한 뒤 추가합니다.
