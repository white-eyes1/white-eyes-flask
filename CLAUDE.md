# flask — 드론 군중 관제(분석 서버)

명세 목록: @../white-eyes-docs/INDEX.md
코드 컨벤션·패키지 구조: white-eyes-docs의 code-conventions.md, package-structure.md

## 일하는 순서
- 작업은 GitHub 이슈 하나 단위. 시작 전에 `gh issue view <번호>`로 이슈를 읽는다.
- 이슈의 FNC·API ID에 해당하는 white-eyes-docs 섹션(features/analyzer.md, api/analyzer.md)을 읽고 나서 코드를 쓴다.
- 명세와 다르게 만들어야 하면 멈추고 사람에게 묻는다. 명세를 임의로 해석하지 않는다.
- 검수 기준은 테스트로 만든다.

## 구조
- 분석 서버(포트 5000, 경로 `/analyzer/v1/**`). 관리자 앱(시작·정지 지시)과 웹 서버(실시간 화면)만 접근한다.
- `src/white_eyes_flask/`: 앱 코드. `tests/`: 테스트. `vendor/yolo_crowd/`: 외부 코드, 수정하지 않는다.
- `ops/`: 녹화 파일 정리·DB 백업 cron 스크립트(FNC-S-014·015). 분석 서버 앱과 따로 실행한다.
- spring과는 HTTP 계약(api/analyzer.md)으로만 연결한다. 서로의 코드를 가져다 쓰지 않는다.

## 명령
- 설치: `uv sync` (`.env`는 `cp .env.example .env` 후 값 채움)
- 실행: `uv run white-eyes-flask` (포트 5000, `.env`의 PORT. `.env`는 앱이 직접 읽음)
- 테스트: `uv run pytest`
- 의존성을 바꾸면 README의 `uv export` 두 줄로 requirements 파일을 다시 만든다.

## 규칙
- 키·토큰·경로는 환경변수로만. 새 변수는 .env.example에 이름만 추가한다.
- 더미 영상·시연 데이터·가중치 파일(`*.pt`)은 커밋하지 않는다. 가중치 경로는 `YOLO_WEIGHTS`.

## git
- 브랜치: feature/<이슈 번호>-<이름>. PR 대상은 항상 dev.
- main·dev에 직접 push하지 않는다.
- 커밋 메시지는 `white-eyes-docs/commit-convention.md`를 따른다.
