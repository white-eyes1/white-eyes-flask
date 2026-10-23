#!/bin/sh
# FNC-S-014 라이브 녹화 파일 정리 (스토리지 서버 cron, 매일 04:00)
#   0 4 * * * /path/to/ops/cleanup_recordings.sh
# 경로를 recordings/live로 고정해 dummy/ 는 건드리지 않는다.
LIVE_DIR="${MEDIA_ROOT:-/srv/media}/recordings/live"

find "$LIVE_DIR" -type f \( -name '*.mp4' -o -name '*.json.gz' \) -mtime +31 -delete
# -mindepth 1: 비어 있어도 recordings/live 자체는 지우지 않는다(명세와 다른 유일한 점)
find "$LIVE_DIR" -mindepth 1 -type d -empty -delete
