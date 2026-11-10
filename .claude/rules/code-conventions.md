---
paths:
  - "**/*.py"
---
# 코드 컨벤션
- 코드를 쓰거나 고치기 전에 `../white-eyes-docs/code-conventions.md`(Python은 2·4·6·7장)와 `../white-eyes-docs/package-structure.md`를 Read로 읽고 따른다. 규칙 원본은 white-eyes-docs이고, 여기에 옮겨 적지 않는다.
- 두 문서를 읽지 못하면 작업을 멈추고 사람에게 알린다. 읽지 않은 채 진행하지 않는다.
- 형식(줄 길이·따옴표·import 순서)은 Black·isort가 맞춘다. 손으로 맞추지 말고 `uv run black .`·`uv run isort .`을 돌린다. CI가 `black --check`·`isort --check-only`로 확인한다.
- `vendor/`는 외부 코드라 고치지 않는다(형식 도구도 건너뛴다).
