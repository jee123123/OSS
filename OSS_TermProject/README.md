# Score2MIDI

악보 이미지에서 MIDI와 MusicXML 파일을 만드는 오픈소스 도구입니다. Audiveris로 악보를 판독한 뒤, `music21`로 변환 결과를 생성합니다.

## 작동 방식

1. Audiveris가 인쇄 악보 이미지를 판독합니다.
2. Audiveris가 MusicXML을 내보냅니다.
3. Score2MIDI가 보정된 MusicXML과 MIDI 파일을 함께 저장합니다.

인식 결과는 원본 악보의 품질과 표기 방식에 따라 달라질 수 있습니다. 특히 낮은 해상도나 흐린 이미지는 판독 결과가 부정확하거나 처리되지 않을 수 있습니다.

## 준비물

- Python 3.10 이상
- [Audiveris](https://github.com/Audiveris/audiveris): 현재 명령줄 버전에서는 별도로 설치해야 합니다. 이 저장소에는 Audiveris 실행 파일이 포함되어 있지 않습니다.

## 설치

프로젝트 폴더에서 가상환경을 만들고 활성화한 다음 Score2MIDI를 설치합니다.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
```

## GUI 실행

GUI 기능을 사용할 때는 선택 의존성을 설치한 다음 실행합니다. GUI는 다중 페이지 PDF 한 개를 직접 변환하거나, 한 곡의 악보 이미지 여러 장을 목록 순서대로 결합해 MIDI와 MusicXML 파일로 저장합니다. `위로`와 `아래로` 버튼으로 이미지 페이지 순서를 조정하고 결과 파일 이름과 저장 폴더를 지정할 수 있습니다. PDF와 이미지 파일은 함께 사용할 수 없습니다. 생성되는 MIDI의 악기 음색은 피아노로 지정됩니다.

```powershell
python -m pip install -e ".[gui]"
score2midi-gui
```

GUI를 처음 사용할 때 `Audiveris 선택` 버튼을 눌러 `Audiveris.exe`를 지정하세요. 선택한 경로는 사용자 설정에 저장되어 다음 실행부터 자동으로 사용됩니다. 아직 Audiveris를 포함한 설치 패키지는 제공하지 않습니다.

## 음표 인식 학습 데이터 만들기

검토가 끝난 MusicXML과 원본 악보를 이용해 음표 분류용 32x32 이미지와 `labels.csv`를 만들 수 있습니다. MusicXML의 오류도 라벨에 반영되므로 먼저 악보 편집기에서 MusicXML을 확인하세요.

```powershell
score2midi-build-dataset "score.pdf" "score.musicxml" "training_data"
```

결과 폴더의 `notehead`에는 MusicXML 음표와 위치가 일치하는 후보가 저장됩니다. 일치하지 않는 후보는 잘못된 음표를 비음표로 학습하는 일을 막기 위해 `review`에 따로 저장되므로, 모델 학습 전에 사람이 확인해야 합니다.

검토 GUI에서는 `1`을 누르면 음표, `2`를 누르면 비음표로 분류하며, `Space`로 건너뛰고 `Ctrl+Z`로 마지막 분류를 되돌릴 수 있습니다. 분류할 때마다 이미지 위치와 `labels.csv`가 즉시 저장됩니다.

빨간 상자는 판정할 후보의 중심을 표시합니다. 상자 중앙에 검거나 빈 타원형 음표 머리가 있으면 줄기와 오선이 함께 보이더라도 음표로 분류합니다. `비음표 다시 검토` 버튼으로 기존 비음표 라벨도 다시 확인할 수 있습니다.

이전 버전에서 만든 데이터셋에는 다음 명령으로 원본 악보 주변 이미지를 추가할 수 있습니다. 기존 라벨은 유지됩니다.

```powershell
python -m score2midi.ml_dataset "score.pdf" --add-context "training_data"
```

```powershell
python -m score2midi.dataset_reviewer "training_data"
```

검토된 `notehead`와 `other`가 각각 20개 이상이면 첫 PyTorch 분류 모델을 학습할 수 있습니다. 두 클래스의 개수가 다르면 적은 클래스에 맞춰 균형 있게 사용하며, 일부 데이터는 검증용으로 자동 분리됩니다.

```powershell
python -m score2midi.note_classifier "training_data" "models/notehead.pt"
```

한 악보에서 만든 소량의 데이터로 얻은 검증 정확도는 참고용입니다. 실제 변환에 연결하기 전 여러 악보로 학습하고, 학습에 사용하지 않은 악보에서도 성능을 확인해야 합니다.

학습 모델은 검토 GUI에서 추천 용도로 사용할 수 있습니다. 추천은 자동 저장되지 않으며, 화면을 확인한 뒤 맞으면 `Enter`로 적용합니다.

```powershell
python -m score2midi.dataset_reviewer "training_data" --model "models/notehead.pt"
```

## 사용법

Windows에서 Audiveris를 기본 경로에 설치한 경우:

```powershell
score2midi path\to\score.png outputs\score.mid --audiveris "C:\Program Files\Audiveris\Audiveris.exe"
```

Audiveris가 다른 위치에 설치되어 있으면 `--audiveris` 뒤에 해당 실행 파일 경로를 지정합니다. 입력 악보와 생성된 MIDI는 권리를 확인한 뒤 사용·배포하세요.

## 샘플 악보

저작권과 배포 허가가 확인되지 않은 악보 이미지는 저장소에 포함하지 않습니다. 시험하려면 사용 권한이 있거나 배포가 허용된 악보 이미지를 `samples` 폴더에 직접 넣어 사용하세요. `samples`와 `outputs`의 내용은 `.gitignore`에 따라 Git에 추가되지 않습니다.

## 라이선스

이 저장소의 Score2MIDI 코드는 MIT License로 배포합니다. 자세한 내용은 [LICENSE](LICENSE)를 확인하세요.

Audiveris와 Python 의존성은 각자의 라이선스로 배포됩니다. 구성 요소와 라이선스 정보는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 정리했습니다.

향후 Windows 데스크톱 배포판은 사용자가 Audiveris를 따로 설치하지 않아도 되도록 Audiveris를 별도 프로그램으로 함께 제공하는 방향으로 구성할 계획입니다. 현재는 설치 패키지가 제공되지 않으므로, 위 명령줄 사용법대로 Audiveris를 별도로 설치해야 합니다. Audiveris를 포함한 배포판을 만들 때는 포함한 정확한 버전, 해당 버전의 소스 코드와 라이선스 고지를 함께 제공합니다.

