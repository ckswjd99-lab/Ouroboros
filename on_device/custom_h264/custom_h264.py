# custom_h264.py ― custom_h264.c의 Python 래퍼

import ctypes
import numpy as np
import os

# MVInfo 구조체 정의 (custom_h264.c와 동일)
class MVInfo(ctypes.Structure):
    _fields_ = [
        ("dst_x", ctypes.c_int16),
        ("dst_y", ctypes.c_int16),
        ("motion_x", ctypes.c_int32),
        ("motion_y", ctypes.c_int32),
        ("motion_scale", ctypes.c_int32),
    ]

# libcustom_h264.so 동적 라이브러리 로드
_lib_path = os.path.join(os.path.dirname(__file__), "libcustom_h264.so")
lib = ctypes.CDLL(_lib_path)

# 함수 시그니처 지정
lib.mv_init.argtypes    = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
lib.mv_init.restype     = ctypes.c_int
lib.mv_process.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(MVInfo), ctypes.c_int]
lib.mv_process.restype  = ctypes.c_int
lib.mv_close.argtypes   = []
lib.mv_close.restype    = None
lib.mv_process_and_encode.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(MVInfo), ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),   # encoded_size_out
    ctypes.POINTER(ctypes.c_int)    # mv_cnt_out
]
lib.mv_process_and_encode.restype = ctypes.c_int

def mv_init(width: int, height: int, fps: int, x264_params: str = None) -> int:
    """라이브러리 초기화. x264_params를 지정하면 ffmpeg/x264 옵션을 직접 전달."""
    if x264_params is None:
        x264_params = ""
    return lib.mv_init(width, height, fps, x264_params.encode("utf-8"))

def mv_process(frame: np.ndarray, max_mv: int = 8192):
    """프레임(BGR np.ndarray)에서 모션 벡터 추출"""
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame은 uint8 BGR 3채널이어야 합니다.")
    mv_buf = (MVInfo * max_mv)()
    ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    n = lib.mv_process(ptr, mv_buf, max_mv)
    if n <= 0:
        return np.zeros((0, 5), dtype=np.int32)
    arr = np.zeros((n, 5), dtype=np.int32)
    for i in range(n):
        arr[i, 0] = mv_buf[i].dst_x
        arr[i, 1] = mv_buf[i].dst_y
        arr[i, 2] = mv_buf[i].motion_x
        arr[i, 3] = mv_buf[i].motion_y
        arr[i, 4] = mv_buf[i].motion_scale
    return arr

def mv_process_and_encode(frame: np.ndarray, max_mv: int = 8192, max_bytes: int = 2*1024*1024):
    """프레임(BGR)에서 모션벡터와 H.264 인코딩 바이트를 동시에 추출"""
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame은 uint8 BGR 3채널이어야 합니다.")
    mv_buf = (MVInfo * max_mv)()
    out_buf = (ctypes.c_uint8 * max_bytes)()
    ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    enc_size = ctypes.c_int()
    mv_cnt   = ctypes.c_int()
    ret = lib.mv_process_and_encode(ptr, mv_buf, max_mv,
                                    out_buf, max_bytes,
                                    ctypes.byref(enc_size), ctypes.byref(mv_cnt))
    if ret != 0:
        raise RuntimeError("encode error")
    encoded_bytes = bytes(out_buf[:enc_size.value])
    arr = np.zeros((mv_cnt.value, 5), dtype=np.int32) if mv_cnt.value > 0 else np.zeros((0, 5), dtype=np.int32)
    for i in range(mv_cnt.value):
        arr[i, 0] = mv_buf[i].dst_x
        arr[i, 1] = mv_buf[i].dst_y
        arr[i, 2] = mv_buf[i].motion_x
        arr[i, 3] = mv_buf[i].motion_y
        arr[i, 4] = mv_buf[i].motion_scale
    return encoded_bytes, arr

def mv_close():
    """라이브러리 자원 해제"""
    lib.mv_close()