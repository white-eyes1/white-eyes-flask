# white-eyes-flask

White Eyes 분석 서버. 명세: `white-eyes-docs`의 `features/analyzer.md`, `api/analyzer.md`.

현재는 골격만 있다: `GET /analyzer/v1/health` 하나.

## 실행

```bash
uv sync
cp .env.example .env   # INTERNAL_TOKEN, YOLO_WEIGHTS 채우기
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

모델 가중치(`*.pt`)는 커밋하지 않는다. `YOLO_WEIGHTS` 환경변수로 파일 경로를 준다.

## vendor/ 출처

`vendor/yolo_crowd/`는 YOLO-CROWD가 쓰는 YOLOv5 모델 코드(`models/`, `utils/`)를 그대로 복사한 것이다. 수정하지 않았다.

- 출처: https://github.com/zaki1003/YOLO-CROWD (커밋 `fbe4f633a40f250e1fafe7ac777f280623c1445a`, 2025-11-04)
- 라이선스: `vendor/yolo_crowd/LICENSE` (YOLO-CROWD, MIT)
- 이 코드는 Ultralytics YOLOv5(https://github.com/ultralytics/yolov5)에서 파생됐다. YOLOv5 v5.0의 라이선스(GPL-3.0)를 `vendor/yolo_crowd/LICENSE.yolov5`에 함께 둔다. 파생 코드의 실제 적용 라이선스는 배포 전에 확인이 필요하다.
- 코드가 `import models`, `from utils...`처럼 최상위 패키지로 서로를 가져오므로, 불러올 때 `vendor/yolo_crowd`를 `sys.path`에 넣어야 한다.
