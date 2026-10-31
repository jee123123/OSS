# Score2MIDI

악보 이미지에서 MIDI 파일을 만드는 오픈소스 명령줄 도구입니다. Audiveris로 악보를 MusicXML로 판독한 뒤, `music21`로 MIDI를 생성합니다.

## 작동 방식

1. Audiveris가 인쇄 악보 이미지를 판독합니다.
2. Audiveris가 MusicXML을 내보냅니다.
3. Score2MIDI가 MusicXML을 MIDI 파일로 변환합니다.

인식 결과는 원본 악보의 품질과 표기 방식에 따라 달라질 수 있습니다. 특히 낮은 해상도나 흐린 이미지는 판독 결과가 부정확하거나 처리되지 않을 수 있습니다.

## 준비물

- Python 3.10 이상
- [Audiveris](https://github.com/Audiveris/audiveris): 별도로 설치해야 합니다. 이 저장소에는 Audiveris 실행 파일이 포함되어 있지 않습니다.

## 설치

프로젝트 폴더에서 가상환경을 만들고 활성화한 다음 Score2MIDI를 설치합니다.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
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

Audiveris와 Python 의존성은 각자의 라이선스로 배포됩니다. 구성 요소와 라이선스 정보는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 정리했습니다. Audiveris는 별도로 설치하는 프로그램이며 이 저장소에 포함되어 있지 않습니다.

