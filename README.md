# white-eyes-flask

White Eyes 분석 서버. 명세: `white-eyes-docs`의 `features/analyzer.md`, `api/analyzer.md`.

현재는 골격만 있다: `GET /analyzer/v1/health` 하나.

## 실행

```bash
uv sync
cp .env.example .env   # INTERNAL_API_TOKEN, YOLO_WEIGHTS 채우기
uv run white-eyes-flask
curl -H "X-Internal-Token: <토큰>" localhost:5000/analyzer/v1/health
```

## 테스트

```bash
uv run pytest
```

## 의존성

`pyproject.toml`/`uv.lock`이 원본이고, `requirements.txt`(실행용)와 `requirements-dev.txt`(+pytest)는 버전까지 고정한 내보내기본이다. 의존성을 바꾼 뒤 다시 만든다.

```bash
uv export --no-dev --no-hashes --no-emit-project -o requirements.txt
uv export --no-hashes --no-emit-project -o requirements-dev.txt
```

Python 버전은 `.python-version`(3.12.15)에 고정.

`torch`는 CPU 전용으로 `2.6.0`에 고정했다(Windows 최신 torch 오류: `docs/troubleshooting/0002-torch-dll-init-windows.md`). Linux에서는 `uv`가 PyTorch CPU 저장소(`https://download.pytorch.org/whl/cpu`)에서 받는다. `requirements.txt`로 `pip`을 쓸 때는 같은 주소를 같이 준다.

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## 가중치

모델 가중치(`*.pt`)는 커밋하지 않는다(`weights/`는 `.gitignore`). `YOLO_WEIGHTS` 환경변수로 파일 경로를 주고, `DETECTOR`로 어느 모델의 가중치인지 알려 준다(기본 `visdrone`).

### 기본: VisDrone YOLOv8m (`DETECTOR=visdrone`)

위에서 내려다본 사람 몸을 찾는 모델이다. 출처는 https://huggingface.co/mshamrai/yolov8m-visdrone 의 `best.pt`이고, 시험에 쓴 파일은 아래 해시와 같다. 라이선스는 **PM이 정할 때까지 미확인**이다(Ultralytics AGPL-3.0, 학습 데이터 VisDrone은 연구 목적 공개). 그래서 이 저장소와 `white-eyes-assets`에는 올리지 않았다. 가중치도 pickle 형식이라 믿을 수 있는 파일만 쓴다.

```bash
curl -L -o weights/visdrone-yolov8m.pt https://huggingface.co/mshamrai/yolov8m-visdrone/resolve/main/best.pt
sha256sum weights/visdrone-yolov8m.pt   # Windows PowerShell: (Get-FileHash weights\visdrone-yolov8m.pt).Hash
# SHA-256: C1A19706FE83E8282795FB6C6CEC6A5E2A6E515C9B5FE4F558863D987EE24F8B (52,006,688바이트)
```

`.env`에는 `YOLO_WEIGHTS=./weights/visdrone-yolov8m.pt`처럼 적는다. 입력 크기는 `DETECT_IMGSZ`(기본 960), 같은 사람을 둘로 센 박스를 합치는 기준은 `DETECT_DEDUPE_OVERLAP`(기본 0.25, 0이면 합치지 않음)이다.

### 이전: YOLO-CROWD (`DETECTOR=yolo-crowd`)

가중치 `yolo-crowd.pt`는 비공개 저장소 `white-eyes1/white-eyes-assets`의 Release `yolo-crowd-v1`에 있다(출처: https://github.com/zaki1003/YOLO-CROWD README의 Postweight). `gh`로 로그인한 뒤 받고, **받은 뒤 해시를 꼭 확인한다**(가중치는 pickle 형식이라 `torch.load(weights_only=False)`로 읽으므로 믿을 수 있는 파일만 쓴다).

```bash
gh release download yolo-crowd-v1 --repo white-eyes1/white-eyes-assets --pattern "yolo-crowd.pt" --dir ./weights
sha256sum weights/yolo-crowd.pt      # Windows PowerShell: (Get-FileHash weights\yolo-crowd.pt).Hash
# SHA-256: 22aea94cad4eb181e94ea600b0013cf7307b1a4dd8788cc8454dc5ca5f573c39
```

`.env`에는 `DETECTOR=yolo-crowd`와 `YOLO_WEIGHTS=./weights/yolo-crowd.pt`처럼 적는다. 위에서 내려다본 영상에서는 정수리만 보여 사람을 많이 놓친다(40명이 보이는 장면에서 2명).

## vendor/ 출처

`vendor/yolo_crowd/`는 YOLO-CROWD가 쓰는 YOLOv5 모델 코드(`models/`, `utils/`)를 그대로 복사한 것이다. 수정하지 않았다.

- 출처: https://github.com/zaki1003/YOLO-CROWD (커밋 `fbe4f633a40f250e1fafe7ac777f280623c1445a`, 2025-11-04)
- 라이선스: `vendor/yolo_crowd/LICENSE` (YOLO-CROWD, MIT)
- 이 코드는 Ultralytics YOLOv5(https://github.com/ultralytics/yolov5)에서 파생됐다. YOLOv5 v5.0의 라이선스(GPL-3.0)를 `vendor/yolo_crowd/LICENSE.yolov5`에 함께 둔다. 파생 코드의 실제 적용 라이선스는 배포 전에 확인이 필요하다.
- 코드가 `import models`, `from utils...`처럼 최상위 패키지로 서로를 가져오므로, 불러올 때 `vendor/yolo_crowd`를 `sys.path`에 넣어야 한다.
