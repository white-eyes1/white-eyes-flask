"""FNC-S-007 RTSP 읽기: OpenCV(FFmpeg 백엔드) VideoCapture"""

import os
import re

import cv2

# 연결·읽기가 이 시간 안에 안 되면 실패로 본다. 정지 요청(최대 10초 대기)이 이보다 오래 막히지 않게 한다.
OPEN_TIMEOUT_MS = 5000
READ_TIMEOUT_MS = 5000
# RTSP로 보는 주소의 접두어(rtsps는 TLS 암호화 버전)
RTSP_SCHEMES = ("rtsp://", "rtsps://")
# 주소의 "//계정:비밀번호@" 부분. 계정에는 '/'나 '@'가 없다고 보고 첫 '@'까지만 잡는다
_CREDENTIALS = re.compile(r"//[^/@]*@")


def is_rtsp_uri(uri: str) -> bool:
    """RTSP 주소인지. 대소문자는 구분하지 않는다. (파일 경로 등 다른 입력과 구분하는 데 쓴다)"""
    return uri.lower().startswith(RTSP_SCHEMES)


def mask_uri(uri: str) -> str:
    """로그에 남길 주소. 주소에 든 계정·비밀번호(rtsp://user:pw@host)는 가린다."""
    return _CREDENTIALS.sub("//***@", uri)


def open_capture(uri: str):
    """RTSP를 연다. 열지 못해도 객체는 돌려주므로 isOpened()로 확인한다."""
    # 기본(UDP)은 패킷이 빠지면 화면이 깨지고 끊김 감지도 늦다. 이미 정해 둔 값이 있으면 그것을 쓴다.
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    return cv2.VideoCapture(
        uri,
        cv2.CAP_FFMPEG,
        [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            OPEN_TIMEOUT_MS,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            READ_TIMEOUT_MS,
        ],
    )
