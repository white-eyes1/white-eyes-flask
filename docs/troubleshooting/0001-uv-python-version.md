# 0001. uv가 .python-version의 3.12.15를 찾지 못함

- 기록: 2026-10-20
- 관련: 이슈 #1, PR #24
- 상태: 우회

## 증상
`uv sync`, `uv run pytest`를 실행하면 `.python-version`에 적힌 Python 3.12.15를 찾지 못해 실패한다.

## 원인
미확인(추정: 설치된 uv 0.11.0이 3.12.15라는 버전을 모른다). `uv self update`로 올리려 했으나 실행 중인 uv 프로세스 때문에 실패했다.

## 해결
우회: Python 버전을 `3.12`로 직접 지정한다.
```
uv sync --python 3.12
uv run --python 3.12 pytest
```
`.python-version`과 `pyproject.toml`(`requires-python = "==3.12.*"`)은 고치지 않았다.

## 재발 방지
uv를 새 버전으로 올릴 수 있으면(실행 중인 uv를 모두 끈 뒤 `uv self update`) 우회 없이 `uv sync`만으로 된다.
