# 0002. Windows에서 최신 torch가 DLL 초기화 오류로 안 불러와짐

- 기록: 2026-10-20
- 관련: 이슈 #3, FNC-S-003
- 상태: 우회

## 증상
`import torch`에서 다음 오류가 난다. torch 2.14.1(PyPI 기본)에서 확인.
```
OSError: [WinError 1114] DLL 초기화 루틴을 실행할 수 없습니다. Error loading ".../torch/lib/c10.dll" or one of its dependencies.
```

## 원인
미확인(추정: 이 PC의 Visual C++ 런타임이 오래돼서 최신 torch가 요구하는 버전을 못 맞춤). 확인한 사실은 두 가지다.
- 이 PC의 `vcruntime140.dll`, `msvcp140.dll` 버전은 14.29.30156.0이다.
- 같은 PC에서 torch 2.6.0+cpu는 오류 없이 불러와진다.

## 해결
우회: `pyproject.toml`에서 `torch==2.6.0`, `torchvision==0.21.0`으로 고정했다. 2.6.0은 vendor 코드(YOLOv5)와 ultralytics 8.4.183에서 쓰는 범위(torch 2.6 이상)의 가장 낮은 버전이다.

## 재발 방지
- 새 PC에서 같은 오류가 나면 최신 Microsoft Visual C++ 재배포 패키지(x64)를 설치한 뒤 다시 시도한다. 이 방법은 시도하지 않아서 효과는 확인하지 못했다.
- torch 버전을 올릴 때는 Windows에서 `uv run python -c "import torch"`로 먼저 확인한다.
