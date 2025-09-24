from __future__ import annotations
"""
Realtime Radar GUI — 4s Capture / 1s Pause+Inference (PyQt5 + pyqtgraph + ONNXRuntime + TTS)

 (1) 실시간 RD(2D FFT + log) 영상
 (2) 터미널: CAPTURE=“Sensing...”/“Next Sign Sensing...”, PAUSE=“Ready For Inference”,
     추론 완료 시 각 클래스 로짓/확률 + Top-1
 (3) 배너: CAPTURE/PAUSE 상태, 추론 완료 시 Top-1 (KO/EN)
 (4) 상태창(PhaseStrip): (2)와 (3) 사이, 진행 상태만 표기 (Next Sign Sensing... / Ready For Inference)
Start / Pause / Resume / Stop 버튼: Stop은 프로그램 종료.

※ 모델은 '로짓 ONNX' 기준. (Softmax는 런타임에서 계산)
※ TTS: pyttsx3 기반, 한국어(가능 시) 음성 선택, 3초 하드캡(+길이 기반 속도 보정), <무동작> 미발화
"""

import os
import sys
import time
import threading
from dataclasses import dataclass
from enum import Enum, auto
from typing import Deque, List, Tuple
from collections import deque

import numpy as np
import cv2
from scipy.signal import windows

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

import onnxruntime as ort
import pyttsx3  # TTS

from ifxradarsdk import get_version_full
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwSequenceChirp

# ============================ User/Model Config ============================
NUM_CLASSES = 15

CLASSES_EN = [
    "Hello.",
    "Nice to meet you.",
    "Thank you.",
    "I respect you.",
    "sign language",
    "I love you.",
    "Take care.",
    "I'm sorry.",
    "Be happy.",
    "Welcome.",
    "Enjoy your meal.",
    "Aha, I see.",
    "I understand.",
    "Goodbye.",
    "<no gesture>",
]
CLASSES_KO = [
    "안녕하세요.",
    "만나서 반갑습니다.",
    "감사합니다.",
    "존경합니다.",
    "수어(수화)",
    "사랑합니다.",
    "몸 건강하세요.",
    "미안합니다.",
    "행복하세요.",
    "환영합니다.",
    "맛있게 드세요.",
    "아하, 알겠습니다.",
    "이해가 됩니다.",
    "안녕히 가세요.",
    "<무동작>",
]

# 숫자 라벨 표기 (터미널에 사용)
CLASSES_NUM = [f"Label{i:02d}" for i in range(NUM_CLASSES)]

# --- ONNX 경로: 환경변수 우선, 없으면 스크립트 옆 export/ 파일 사용
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ONNX = os.path.join(BASE_DIR, "export", "spectranet_logits.onnx")
ONNX_PATH = os.environ.get("SPECTRANET_ONNX", DEFAULT_ONNX)

# Capture/Pause 시간(초)
WINDOW_SEC = 4.0
PAUSE_SEC  = 2.0
UI_FPS     = 50

# 라이브 RD 표시 해상도(H, W)
RD_SHAPE   = (224, 130)

# ============================= Radar Config (Exact) ========================
RADAR_CONFIG = FmcwSimpleSequenceConfig(
    frame_repetition_time_s=0.08,
    chirp_repetition_time_s=300e-6,
    num_chirps=256,
    tdm_mimo=False,
    chirp=FmcwSequenceChirp(
        start_frequency_Hz=58e9,
        end_frequency_Hz=63.5e9,
        sample_rate_Hz=1e6,
        num_samples=128,
        rx_mask=7,
        tx_mask=1,
        tx_power_level=31,
        lp_cutoff_Hz=500000,
        hp_cutoff_Hz=80000,
        if_gain_dB=25,
    ),
)

# ============================ Math helpers =================================
def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x -= x.max()
    e = np.exp(x)
    return (e / (e.sum() + 1e-9)).astype(np.float32)

# ============================ Preprocess (Dataset spec) ====================
def min_max_normalize(x: np.ndarray) -> np.ndarray:
    x_min = float(np.min(x)); x_max = float(np.max(x))
    if x_max - x_min == 0:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - x_min) / (x_max - x_min)
    return np.nan_to_num(y).astype(np.float32)

def compute_rtm_dtm(frames_T_C_S_Rx: np.ndarray, rx_index: int, top_k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """frames_T_C_S_Rx: (T, 256, 128, 3)"""
    raw_stack = frames_T_C_S_Rx[:, :, :, rx_index]  # (T,256,128)
    rdi_stack: List[np.ndarray] = []
    doppler_window = None

    for t in range(raw_stack.shape[0]):
        raw_frame = raw_stack[t]  # (256,128)
        # Mean subtraction (2D)
        col_mean = np.mean(raw_frame, axis=0, keepdims=True)
        mean_sub = raw_frame - col_mean
        row_mean = np.mean(mean_sub, axis=1, keepdims=True)
        mean_sub = mean_sub - row_mean
        # Range Hamming + FFT
        X = mean_sub * windows.hamming(mean_sub.shape[1])[None, :]
        range_fft = np.fft.fft(X, axis=1, n=448)[:, :224]  # (256,224)
        # Doppler window
        if doppler_window is None:
            doppler_window = windows.hamming(range_fft.shape[0])[:, None]  # (256,1)
        doppler_input = range_fft * doppler_window
        # Doppler FFT + shift
        doppler_fft = np.fft.fftshift(np.fft.fft(doppler_input, axis=0), axes=0)
        doppler_mag = np.log1p(np.abs(doppler_fft)).astype(np.float32)
        rdi_stack.append(doppler_mag)

    rtm_list, dtm_list = [], []
    for rdi in rdi_stack:
        rtm_vec = np.sum(np.sort(rdi, axis=0)[-top_k:, :], axis=0)  # (224,)
        dtm_vec = np.sum(np.sort(rdi, axis=1)[:, -top_k:], axis=1)  # (256,)
        rtm_list.append(rtm_vec)
        dtm_list.append(dtm_vec)

    rtm = np.stack(rtm_list, axis=1)  # (224,T)
    dtm = np.stack(dtm_list, axis=1)  # (256,T)

    rtm_resized = cv2.resize(rtm, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    dtm_resized = cv2.resize(dtm, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return rtm_resized, dtm_resized

def make_six_channel_from_window(frames_raw: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """frames_raw: list of frames, each (Rx,256,128) -> (rtm, dtm) 각각 (224,224,3)"""
    arr = np.stack(frames_raw, axis=0)  # (T, Rx, 256, 128)
    T, Rx, C, S = arr.shape
    if Rx < 3:
        raise ValueError(f"Expect >=3 RX, got {Rx}")
    frames_T_C_S_Rx = np.transpose(arr[:, :3, :, :], (0, 2, 3, 1))  # (T,256,128,3)

    rtm_list, dtm_list = [], []
    for rx_idx in range(3):
        rtm, dtm = compute_rtm_dtm(frames_T_C_S_Rx, rx_idx, top_k=1)
        rtm_list.append(min_max_normalize(rtm))
        dtm_list.append(min_max_normalize(dtm))

    rtm_stack = np.stack(rtm_list, axis=-1).astype(np.float32)  # (224,224,3)
    dtm_stack = np.stack(dtm_list, axis=-1).astype(np.float32)  # (224,224,3)
    return rtm_stack, dtm_stack

# ============================ Live RD (light) ==============================
def rd_live_from_frame(frame: np.ndarray) -> np.ndarray:
    """빠른 RD 미리보기 생성 (RX0만 사용). 반환: (H,W)=RD_SHAPE, [0..1]"""
    arr = np.asarray(frame)  # (Rx,256,128)
    rx0 = arr[0]
    col_mean = np.mean(rx0, axis=0, keepdims=True)   # (1, S)
    rx0 = rx0 - col_mean
    row_mean = np.mean(rx0, axis=1, keepdims=True)   # (C, 1)
    rx0 = rx0 - row_mean
    C, S = rx0.shape
    w_r = windows.hamming(S)[None, :]
    w_d = windows.hamming(C)[:, None]
    Xr = np.fft.rfft(rx0 * w_r, n=S, axis=1)      # (C, S//2+1)
    Xrd = np.fft.fft(Xr * w_d, n=C, axis=0)
    mag = np.abs(np.fft.fftshift(Xrd, axes=0))
    rd = np.log1p(mag)
    rd -= rd.min(); d = rd.max(); rd = rd / d if d > 0 else rd
    rd[rd < 0.1] = 0
    H, W = RD_SHAPE
    rd = cv2.resize(rd.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    return rd

# ============================ Inference Worker =============================
@dataclass
class InferenceJob:
    window_frames: List[np.ndarray]  # each frame: (Rx,256,128)

class InferenceWorker(QtCore.QThread):
    # emits: probs, logits, top_ko, top_en
    result_ready = QtCore.pyqtSignal(np.ndarray, np.ndarray, str, str)

    def __init__(self):
        super().__init__()
        self._queue: Deque[InferenceJob] = deque()
        self._qlock = threading.Lock()
        self._running = True
        self._build_session()

    def _build_session(self):
        if not os.path.isfile(ONNX_PATH):
            raise FileNotFoundError(f"ONNX not found: {ONNX_PATH}")

        sess = None
        last_err = None
        avail = set(ort.get_available_providers())  # e.g., {'CPUExecutionProvider', ...}
        candidates = [
            ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            ["CPUExecutionProvider"],
        ]
        for provs in candidates:
            use = [p for p in provs if p in avail]
            if not use:
                continue
            try:
                so = ort.SessionOptions()
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                so.intra_op_num_threads = max(1, int(os.environ.get("ORT_INTRA_OP", "1")))
                so.inter_op_num_threads = max(1, int(os.environ.get("ORT_INTER_OP", "1")))
                sess = ort.InferenceSession(ONNX_PATH, providers=use, sess_options=so)
                print(f"[ORT] Using providers: {sess.get_providers()}")
                break
            except Exception as e:
                last_err = e
        if sess is None:
            raise RuntimeError(f"Failed to create ORT session for {ONNX_PATH}: {last_err}")

        self.session = sess
        ins = self.session.get_inputs()
        if len(ins) != 2:
            raise RuntimeError(f"Expect 2 inputs (rtm, dtm), got {len(ins)}")
        self.rtmin_name, self.dtmin_name = ins[0].name, ins[1].name
        self.out_name = self.session.get_outputs()[0].name  # logits (N, num_classes)

    def enqueue(self, job: InferenceJob):
        with self._qlock:
            self._queue.append(job)

    def run(self):
        while self._running:
            job = None
            with self._qlock:
                if self._queue:
                    job = self._queue.popleft()
            if job is None:
                self.msleep(5)
                continue

            rtm, dtm = make_six_channel_from_window(job.window_frames)
            rtm_b = rtm[np.newaxis, ...].astype(np.float32)
            dtm_b = dtm[np.newaxis, ...].astype(np.float32)

            logits = self.session.run([self.out_name], {
                self.rtmin_name: rtm_b,
                self.dtmin_name: dtm_b,
            })[0][0].astype(np.float32)  # (num_classes,)
            probs = softmax(logits)
            top = int(np.argmax(probs))
            self.result_ready.emit(probs, logits, CLASSES_KO[top], CLASSES_EN[top])

    def stop(self):
        self._running = False

# ============================ TTS Worker ===================================
class TTSWorker(QtCore.QThread):
    """
    비동기 TTS 재생기.
    - enqueue(text)로 한국어 문장을 넣으면 순차적으로 읽어줍니다.
    - 3초 하드캡(환경변수 TTS_MAX_SECS, 기본 2.9s)
    - 길이 기반 속도 보정 + 기본 rate(환경변수 TTS_RATE, 기본 210)
    - 가능한 경우 한국어 음성을 자동 선택
    - 중요: 모든 엔진 호출은 이 스레드 안에서만 수행 (타이머 스레드에서 stop 금지)
    """
    def __init__(self, rate: int | None = None, volume: float = 1.0):
        super().__init__()
        self._queue: Deque[str] = deque()
        self._qlock = threading.Lock()
        self._running = True

        self.max_secs = float(os.environ.get("TTS_MAX_SECS", "2.9"))
        self._default_rate = int(os.environ.get("TTS_RATE", str(rate if rate is not None else 210)))

        self._engine = pyttsx3.init()
        # 한국어 음성 우선 선택
        try:
            ko_voice_id = None
            for v in self._engine.getProperty("voices"):
                meta = " ".join([
                    getattr(v, "name", "") or "",
                    getattr(v, "id", "") or "",
                    " ".join(getattr(v, "languages", []) or []),
                ]).lower()
                if "ko" in meta or "korean" in meta or "kor" in meta:
                    ko_voice_id = v.id
                    break
            if ko_voice_id:
                self._engine.setProperty("voice", ko_voice_id)
        except Exception:
            pass

        self._engine.setProperty("rate", self._default_rate)
        self._engine.setProperty("volume", volume)

    def enqueue(self, text: str):
        if not text:
            return
        with self._qlock:
            self._queue.append(text)

    def _speak_with_timeout(self, text: str):
        # 글자수에 따른 속도 보정 (3초 내 수렴 유도)
        try:
            n = len(text)
            if n > 12:
                self._engine.setProperty("rate", min(self._default_rate + 40, 260))
            elif n > 8:
                self._engine.setProperty("rate", min(self._default_rate + 20, 240))
            else:
                self._engine.setProperty("rate", self._default_rate)
        except Exception:
            pass

        # 같은 스레드에서 루프 시작/반복/종료
        self._engine.say(text)
        start = time.time()
        try:
            self._engine.startLoop(False)  # 비차단 루프 시작
            # busy 동안 iterate()로 펌핑
            while self._running and self._engine.isBusy():
                now = time.time()
                if (now - start) >= self.max_secs:
                    # 하드캡: 루프 종료 시도
                    try:
                        self._engine.endLoop()
                    except Exception:
                        pass
                    # 루프 종료 직후 busy 잔여 시 1회 stop (같은 스레드!)
                    try:
                        if self._engine.isBusy():
                            self._engine.stop()
                    except Exception:
                        pass
                    return
                try:
                    self._engine.iterate()
                except Exception:
                    break
                # CPU 과점유 방지
                time.sleep(0.01)
        finally:
            # 자연 종료 케이스: 안전하게 endLoop
            try:
                self._engine.endLoop()
            except Exception:
                pass

    def run(self):
        while self._running:
            text = None
            with self._qlock:
                if self._queue:
                    text = self._queue.popleft()
            if text is None:
                self.msleep(20)
                continue
            try:
                self._speak_with_timeout(text)
            except Exception:
                # 엔진 오류 발생 시 다음 항목으로 진행
                pass

    def stop(self):
        self._running = False
        try:
            # 현재 루프를 깨우기 위해 같은 스레드 컨텍스트에서 stop/endLoop 시도
            self._engine.stop()
        except Exception:
            pass


# ============================ UI Widgets ==================================
class TerminalWidget(QtWidgets.QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.document().setMaximumBlockCount(3000)
        self.setFont(QtGui.QFont("Consolas", 15))

    def append_line(self, text: str):
        self.appendPlainText(text)
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())

class Banner(QtWidgets.QLabel):
    def __init__(self):
        super().__init__("—")
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setFont(QtGui.QFont("Noto Sans CJK KR", 45, QtGui.QFont.Bold))
        self.setStyleSheet(
            "padding:12px; border:2px solid #444; border-radius:12px;"
            "background-color:#0b0b0b; color:#f2f2f2;"
        )

    def set_text(self, text: str):
        self.setText(text)

class PhaseStrip(QtWidgets.QLabel):
    """
    (4) 상태 표시 전용 라벨.
    - CAPTURE: "Next Sign Sensing..." + 파란색 + 점 애니메이션(1~3개)
    - PAUSE_INFER: "Ready For Inference" + 노란색(고정)
    - USER PAUSE: '—' + 회색(고정)
    """
    class Mode(Enum):
        CAPTURE = auto()
        PAUSE_INFER = auto()
        PAUSED = auto()

    def __init__(self):
        super().__init__("—")
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setFont(QtGui.QFont("Noto Sans CJK KR", 18, QtGui.QFont.Medium))
        self.setStyleSheet(
            "padding:8px; border:1px solid #666; border-radius:8px;"
            "background:#111; color:#e0e0e0;"
        )
        self._mode = PhaseStrip.Mode.PAUSED
        self._dot = 0
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(400)  # 점 순환 속도
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

    def set_mode(self, mode: "PhaseStrip.Mode"):
        self._mode = mode
        self._dot = 0
        self._render()

    def _on_tick(self):
        if self._mode == PhaseStrip.Mode.CAPTURE:
            self._dot = (self._dot + 1) % 3
            self._render()

    def _render(self):
        if self._mode == PhaseStrip.Mode.CAPTURE:
            dots = "." * (self._dot + 1)
            self.setText(f"Next Sign Sensing{dots}")
            self.setStyleSheet(
                "padding:8px; border:1px solid #3b6ea5; border-radius:8px;"
                "background:#0f1b2b; color:#cfe4ff;"
            )
        elif self._mode == PhaseStrip.Mode.PAUSE_INFER:
            self.setText("Ready For Inference")
            self.setStyleSheet(
                "padding:8px; border:1px solid #8a6d00; border-radius:8px;"
                "background:#2a2200; color:#ffe38a;"
            )
        else:  # PAUSED
            self.setText("—")
            self.setStyleSheet(
                "padding:8px; border:1px solid #666; border-radius:8px;"
                "background:#111; color:#9a9a9a;"
            )

class LiveImage(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()
        v = self.addViewBox()
        v.setAspectLocked(True)
        self.img = pg.ImageItem(axisOrder='row-major')
        self.img.setLevels((0.0, 1.0))
        v.addItem(self.img)

        # viridis LUT 적용
        import matplotlib.cm as cm
        viridis = cm.get_cmap('viridis', 256)
        lut = (viridis(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.ubyte)
        self.img.setLookupTable(lut)

    def set_image(self, arr: np.ndarray):
        self.img.setImage(arr, autoLevels=False)

    def clear_image(self):
        self.img.clear()

# ============================ State Machine ===============================
class Phase(Enum):
    CAPTURE = auto()
    PAUSE_INFER = auto()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IFX Radar — 4s Capture / 1s Pause+Infer (ONNX + TTS)")
        self.resize(1380, 900)

        # Layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        h = QtWidgets.QHBoxLayout(central)

        # 좌: 라이브 영상(좁게)
        self.live = LiveImage(); h.addWidget(self.live, 2)
        # 우: 터미널 + (4)상태 + (3)배너 (넓게)
        right = QtWidgets.QWidget(); h.addWidget(right, 3)
        v = QtWidgets.QVBoxLayout(right)

        # --- Control bar (Start / Pause / Resume / Stop) ---
        ctrl = QtWidgets.QWidget(); v.addWidget(ctrl)
        ch = QtWidgets.QHBoxLayout(ctrl); ch.setContentsMargins(0,0,0,0)
        self.btn_start  = QtWidgets.QPushButton("Start")
        self.btn_pause  = QtWidgets.QPushButton("Pause")
        self.btn_resume = QtWidgets.QPushButton("Resume")
        self.btn_stop   = QtWidgets.QPushButton("Stop")
        ch.addWidget(self.btn_start); ch.addWidget(self.btn_pause)
        ch.addWidget(self.btn_resume); ch.addWidget(self.btn_stop)

        # TTS 토글 추가
        self.chk_tts = QtWidgets.QCheckBox("TTS (KO) On")
        self.chk_tts.setChecked(True)
        ch.addWidget(self.chk_tts); ch.addStretch(1)

        # 버튼 스타일
        for btn in [self.btn_start, self.btn_pause, self.btn_resume, self.btn_stop]:
            btn.setMinimumHeight(60)
            btn.setMinimumWidth(240)
            btn.setFont(QtGui.QFont("Noto Sans CJK KR", 15, QtGui.QFont.Bold))

        # (2) 터미널
        self.term = TerminalWidget(); v.addWidget(self.term, 3)

        # (4) 상태창
        self.phase_strip = PhaseStrip(); v.addWidget(self.phase_strip, 0)

        # (3) 배너 (결과 전용)
        self.banner = Banner(); v.addWidget(self.banner, 1)

        self.statusBar().showMessage("Idle — press Start")

        # Device
        self.device = DeviceFmcw()
        self.sequence = self.device.create_simple_sequence(RADAR_CONFIG)
        self.device.set_acquisition_sequence(self.sequence)
        self.term.append_line(f"SDK: {get_version_full()}")
        self.term.append_line(f"Sensor: {self.device.get_sensor_type()}")
        self.term.append_line("="*70)

        # Timers
        self.ui_timer = QtCore.QTimer(self)
        self.ui_timer.setInterval(int(1000 / UI_FPS))
        self.ui_timer.timeout.connect(self._on_ui_tick)

        self.acq_timer = QtCore.QTimer(self)
        self.acq_timer.setInterval(int(RADAR_CONFIG.frame_repetition_time_s * 1000))  # 80ms
        self.acq_timer.timeout.connect(self._poll_frame)

        self.phase_timer = QtCore.QTimer(self)
        self.phase_timer.setSingleShot(True)
        self.phase_timer.timeout.connect(self._next_phase)

        # Inference worker
        self.inf = InferenceWorker()
        self.inf.result_ready.connect(self._on_infer_done)
        self.inf.start()

        # TTS 워커
        self.tts = TTSWorker(rate=None, volume=1.0)
        self.tts.start()
        self._last_spoken = None
        self._tts_blocklist = {"<무동작>"}  # 무동작은 읽지 않음
        # TTS 별칭(축약) 맵 — 필요 시 자유롭게 추가/수정
        self._tts_alias = {
            "수어(수화)": "수화",
            "맛있게 드세요.": "맛있게 드세요",
            "만나서 반갑습니다.": "만나서 반갑습니다",
            "아하, 알겠습니다.": "아하, 알겠습니다",
            "이해가 됩니다.": "이해가 됩니다",
            "몸 건강하세요.": "몸 건강하세요.",
            "안녕하세요.": "안녕하세요",
            "감사합니다.": "감사합니다",
            "환영합니다.": "환영합니다",
            "행복하세요.": "행복하세요",
            "존경합니다.": "존경합니다",
            "사랑합니다.": "사랑합니다",
            "미안합니다.": "미안합니다",
            "안녕히 가세요.": "안녕히 가세요",
        }

        # Buffers/state
        self.frames_buf: List[np.ndarray] = []
        self.live_queue: Deque[np.ndarray] = deque()
        self.phase: Phase | None = None
        self._blink = False
        self._cycle = 0
        self._paused = True  # idle start

        # Initial UI
        self.banner.set_text("Press Start!!")
        self.phase_strip.set_mode(PhaseStrip.Mode.PAUSED)

        # Connect buttons
        self.btn_start.clicked.connect(self._on_start_clicked)
        self.btn_pause.clicked.connect(self._on_pause_clicked)
        self.btn_resume.clicked.connect(self._on_resume_clicked)
        self.btn_stop.clicked.connect(self._on_stop_clicked)

        # 라벨 매핑 표 출력
        for i, (en, ko) in enumerate(zip(CLASSES_EN, CLASSES_KO)):
            self.term.append_line(f"Label {i:02d}: \"{en}\" / {ko}")
        self.term.append_line("="*70)

    # ------------------------ Phase control ------------------------
    def _set_phase(self, phase: Phase):
        if self._paused:
            return
        self.phase = phase
        if phase == Phase.CAPTURE:
            try:
                self.device.set_acquisition_sequence(self.sequence)
            except Exception as e:
                self.term.append_line(f"[WARN] set_acquisition_sequence: {e}")

            self.frames_buf.clear()
            self.live_queue.clear()

            self.phase_strip.set_mode(PhaseStrip.Mode.CAPTURE)

            if self._cycle == 0:
                self.term.append_line("[CAPTURE] Sensing started (4s)")
            else:
                self.term.append_line("="*70)
                self.term.append_line("[CAPTURE] Next Sign Sensing... (4s)")

            if not self.ui_timer.isActive():
                self.ui_timer.start()
            if not self.acq_timer.isActive():
                self.acq_timer.start()
            self.phase_timer.start(int(WINDOW_SEC * 1000))

        elif phase == Phase.PAUSE_INFER:
            try:
                self.acq_timer.stop()
                self.device.stop_acquisition()
            except Exception:
                pass

            self.phase_strip.set_mode(PhaseStrip.Mode.PAUSE_INFER)
            self.term.append_line(f"[PAUSE] Ready For Inference ({PAUSE_SEC:.0f}s)")

            if self.frames_buf:
                self.inf.enqueue(InferenceJob(window_frames=list(self.frames_buf)))

            self.phase_timer.start(int(PAUSE_SEC * 1000))
            self._cycle += 1

    def _next_phase(self):
        if self._paused:
            return
        self._set_phase(Phase.CAPTURE if self.phase == Phase.PAUSE_INFER else Phase.PAUSE_INFER)

    # ------------------------ Acquisition & UI ----------------------
    @QtCore.pyqtSlot()
    def _poll_frame(self):
        try:
            frame_list = self.device.get_next_frame()  # iterable
        except Exception as e:
            self.term.append_line(f"[ERR] get_next_frame: {e}")
            return
        for fr in frame_list:
            self.frames_buf.append(np.asarray(fr))        # (Rx,256,128)
            self.live_queue.append(rd_live_from_frame(fr))

    @QtCore.pyqtSlot()
    def _on_ui_tick(self):
        if self._paused:
            return
        self._blink = not self._blink
        if self.phase == Phase.CAPTURE:
            if self._blink:
                msg = "Next Sign Sensing…" if self._cycle >= 1 else "Sensing…"
                self.statusBar().showMessage(msg)
        else:
            if self._blink:
                self.statusBar().showMessage("Ready For Inference")

        # 라이브 영상 업데이트 (PAUSE 중엔 큐 소진 후 빈화면)
        if self.live_queue:
            rd = self.live_queue.popleft()
            self.live.set_image(rd)
        else:
            if self.phase == Phase.PAUSE_INFER:
                self.live.clear_image()

    @QtCore.pyqtSlot(np.ndarray, np.ndarray, str, str)
    def _on_infer_done(self, probs: np.ndarray, logits: np.ndarray, top_ko: str, top_en: str):
        if self._paused:
            return
        ts = time.strftime("%H:%M:%S")

        lines = [
            f'{CLASSES_NUM[i]} ("{CLASSES_EN[i]}"): '
            f'logit={logits[i]:+7.3f}  prob={probs[i]:.3f}'
            for i in range(len(probs))
        ]
        self.term.append_line(f'[{ts}] TOP = {top_en} / {top_ko}\n  ' + "\n  ".join(lines))

        # 배너에는 결과(한/영)만 표기
        self.banner.set_text(f"{top_ko}\n{top_en}")

        # === TTS: 한국어 Top-1 읽기 (무동작 미발화, 중복 방지, Paused 시 미재생) ===
        if (not self._paused) and self.chk_tts.isChecked():
            if top_ko not in self._tts_blocklist:
                say_text = self._tts_alias.get(top_ko, top_ko)  # 별칭 치환(축약)
                if say_text:
                    self.tts.enqueue(say_text)

    # ------------------------ Controls -----------------------------
    def _on_start_clicked(self):
        if not self._paused:
            return
        self._paused = False
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_resume.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.statusBar().showMessage("Running")
        self.phase_strip.set_mode(PhaseStrip.Mode.CAPTURE)
        self._cycle = 0 if self.phase is None else self._cycle
        self._set_phase(Phase.CAPTURE)

    def _on_pause_clicked(self):
        if self._paused:
            return
        self._paused = True
        try:
            self.phase_timer.stop()
            self.acq_timer.stop()
            self.ui_timer.stop()
            self.device.stop_acquisition()
        except Exception:
            pass
        self.statusBar().showMessage("Paused — press Resume")

        self.banner.set_text("Paused")                      # (3)
        self.phase_strip.set_mode(PhaseStrip.Mode.PAUSED)   # (4)

        self.term.append_line("[PAUSE] All stopped by user")
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(True)
        self.btn_stop.setEnabled(True)

    def _on_resume_clicked(self):
        if not self._paused:
            return
        self._paused = False
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_resume.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.statusBar().showMessage("Running")
        self.phase_strip.set_mode(PhaseStrip.Mode.CAPTURE)
        self._set_phase(Phase.CAPTURE)

    def _on_stop_clicked(self):
        self.close()

    # ------------------------ Lifecycle ----------------------------
    def closeEvent(self, e: QtGui.QCloseEvent):
        try:
            self.phase_timer.stop(); self.acq_timer.stop(); self.ui_timer.stop()
            self.inf.stop(); self.inf.wait(1000)
            try:
                self.device.stop_acquisition()
            except Exception:
                pass
            try:
                self.device.close()
            except Exception:
                pass
            # TTS 정리
            try:
                self.tts.stop()
                self.tts.wait(1000)
            except Exception:
                pass
        finally:
            return super().closeEvent(e)

# ============================ Entrypoint ==================================
def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder='row-major')
    win = MainWindow(); win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
