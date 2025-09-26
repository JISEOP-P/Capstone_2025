from __future__ import annotations
import os, sys, time, threading, platform, wave
from dataclasses import dataclass
from enum import Enum, auto
from typing import Deque, List, Tuple
from collections import deque

import numpy as np
import cv2
from scipy.signal import windows

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtMultimedia import QSoundEffect
import pyqtgraph as pg
import matplotlib  # for colormap

import onnxruntime as ort
import pyttsx3

from ifxradarsdk import get_version_full
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwSequenceChirp

# ============================ User/Model Config ============================
NUM_CLASSES = 15
CLASSES_EN = [
    "Hello.","Nice to meet you.","Thank you.","I respect you.","sign language",
    "I love you.","Take care.","I'm sorry.","Be happy.","Welcome.",
    "Enjoy your meal.","Aha, I see.","I understand.","Goodbye.","<no gesture>",
]
CLASSES_KO = [
    "안녕하세요.","만나서 반갑습니다.","감사합니다.","존경합니다.","수어(수화)",
    "사랑합니다.","몸 건강하세요.","미안합니다.","행복하세요.","환영합니다.",
    "맛있게 드세요.","아하, 알겠습니다.","이해가 됩니다.","안녕히 가세요.","<무동작>",
]
CLASSES_NUM = [f"Label{i:02d}" for i in range(NUM_CLASSES)]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ONNX = os.path.join(BASE_DIR, "export", "spectranet_logits.onnx")
ONNX_PATH = os.environ.get("SPECTRANET_ONNX", DEFAULT_ONNX)

WINDOW_SEC = 4.0
PAUSE_SEC  = 4.0
UI_FPS     = 50

# ---------------- display aspect targets (width:height) --------------------
ASPECT_RDI_XY = 128 / 256   # RDI 표시 비(W/H)
ASPECT_SQUARE = 1.0         # RTM/DTM 정사각

# 물리 범위
VEL_MAX   = 4.3
RANGE_MAX = 1.74
TIME_MAX  = 4.0

# 눈금(major ticks)
VEL_TICKS_MAJOR   = [-4,-3,-2,-1,0,1,2,3,4]
TIME_TICKS_MAJOR  = [1,2,3]
RANGE_TICKS_MAJOR = [0.0, 0.5, 1.0, 1.5]

# ============================= Radar Config ===============================
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
        rx_mask=7, tx_mask=1, tx_power_level=31,
        lp_cutoff_Hz=500000, hp_cutoff_Hz=80000, if_gain_dB=25,
    ),
)

# ============================ 공통 UI 값 ===================================
UNIFIED_BAR_H  = 50
BTN_W = 160
HEADER_FONT_PT = 20
PHASE_FONT_PT  = 20  # 원하면 HEADER_FONT_PT와 동일 사용

# === 공통 폰트 헬퍼 ===
def header_font(point_size: int) -> QtGui.QFont:
    f = QtGui.QFont("Noto Sans CJK KR", int(point_size))
    f.setWeight(QtGui.QFont.Bold)  # 헤더와 동일 굵기
    return f

# === 간단 톤 WAV 생성 ===
def ensure_beep_wav(path: str, freq=1000, dur_ms=120, sr=44100, amplitude=0.35):
    if os.path.isfile(path):
        return
    t = np.linspace(0, dur_ms/1000.0, int(sr*dur_ms/1000.0), False, dtype=np.float64)
    tone = amplitude * np.sin(2*np.pi*freq*t)
    audio = (tone * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())

# ============================ Helpers =====================================
def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64); x -= x.max()
    e = np.exp(x); return (e / (e.sum() + 1e-9)).astype(np.float32)

def min_max_normalize(x: np.ndarray) -> np.ndarray:
    x_min = float(np.min(x)); x_max = float(np.max(x))
    if x_max - x_min == 0:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - x_min) / (x_max - x_min)
    return np.nan_to_num(y).astype(np.float32)

# ============================ RTM/DTM =====================================
RANGE_SAMPLES   = 128
DOPPLER_CHIRPS  = 256
RANGE_FFT_N     = 448
RANGE_KEEP_N    = 224
_W_R = windows.hamming(RANGE_SAMPLES, sym=False)[None, :].astype(np.float32)
_W_D = windows.hamming(DOPPLER_CHIRPS,  sym=False)[:, None].astype(np.float32)

def compute_rtm_dtm(frames_T_C_S_Rx: np.ndarray, rx_index: int, top_k: int = 1
                    ) -> tuple[np.ndarray, np.ndarray]:
    raw_stack = frames_T_C_S_Rx[:, :, :, rx_index]  # (T,256,128)
    T = raw_stack.shape[0]
    rtm_list, dtm_list = [], []
    doppler_window = windows.hamming(DOPPLER_CHIRPS, sym=False)[:, None].astype(np.float32)

    for t in range(T):
        x = raw_stack[t].astype(np.float32)
        x = x - np.mean(x, axis=0, keepdims=True)
        x = x - np.mean(x, axis=1, keepdims=True)

        xr = x * _W_R
        rf = np.fft.fft(xr, axis=1, n=RANGE_FFT_N)[:, :RANGE_KEEP_N]  # (256,224)
        xd = rf * doppler_window
        df = np.fft.fftshift(np.fft.fft(xd, axis=0), axes=0)

        rdi = np.log1p(np.abs(df)).astype(np.float32)

        r_sorted = np.sort(rdi, axis=0)
        rtm_vec  = np.sum(r_sorted[-top_k:, :], axis=0)
        c_sorted = np.sort(rdi, axis=1)
        dtm_vec  = np.sum(c_sorted[:, -top_k:], axis=1)

        rtm_list.append(rtm_vec)
        dtm_list.append(dtm_vec)

    rtm = np.stack(rtm_list, axis=1)                                  # (224,T)
    dtm = np.stack(dtm_list, axis=1)                                  # (256,T)

    rtm_resized = cv2.resize(rtm, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    dtm_resized = cv2.resize(dtm, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return rtm_resized, dtm_resized

def make_six_channel_from_window(frames_raw: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
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

# ============================ Live RDI =====================================
def rd_live_from_frame(frame: np.ndarray) -> np.ndarray:
    """
    RDI: x=Distance, y=Velocity. 회전/반전 없음. 반환 범위 0~1.
    """
    arr = np.asarray(frame)                # (Rx,256,128)
    x   = arr[0].astype(np.float32)        # (256,128)

    x = x - np.mean(x, axis=0, keepdims=True)
    x = x - np.mean(x, axis=1, keepdims=True)

    Xr = x * _W_R
    rf = np.fft.fft(Xr, axis=1, n=RANGE_FFT_N)[:, :RANGE_KEEP_N]      # (256,224)
    Xd = rf * _W_D
    df = np.fft.fftshift(np.fft.fft(Xd, axis=0), axes=0)              # (256,224)

    mag = np.log1p(np.abs(df)).astype(np.float32)
    mag -= mag.min(); m = mag.max(); mag = (mag/m) if m > 0 else mag
    return mag

# ============================ Inference Worker =============================
@dataclass
class InferenceJob:
    window_frames: List[np.ndarray]

class InferenceWorker(QtCore.QThread):
    result_ready = QtCore.pyqtSignal(np.ndarray, np.ndarray, str, str, np.ndarray, np.ndarray)
    def __init__(self):
        super().__init__()
        self._queue: Deque[InferenceJob] = deque()
        self._qlock = threading.Lock()
        self._running = True
        self._build_session()
    def _build_session(self):
        if not os.path.isfile(ONNX_PATH):
            raise FileNotFoundError(f"ONNX not found: {ONNX_PATH}")
        sess = None; last_err = None
        avail = set(ort.get_available_providers())
        candidates = [
            ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            ["CPUExecutionProvider"],
        ]
        for provs in candidates:
            use = [p for p in provs if p in avail]
            if not use: continue
            try:
                so = ort.SessionOptions()
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                so.intra_op_num_threads = max(1, int(os.environ.get("ORT_INTRA_OP", "1")))
                so.inter_op_num_threads = max(1, int(os.environ.get("ORT_INTER_OP", "1")))
                sess = ort.InferenceSession(ONNX_PATH, providers=use, sess_options=so)
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
        self.out_name = self.session.get_outputs()[0].name
    def enqueue(self, job: InferenceJob):
        with self._qlock:
            self._queue.append(job)
    def run(self):
        while self._running:
            job = None
            with self._qlock:
                if self._queue: job = self._queue.popleft()
            if job is None:
                self.msleep(5); continue
            rtm, dtm = make_six_channel_from_window(job.window_frames)
            rtm0 = rtm[..., 0]; dtm0 = dtm[..., 0]
            rtm_b = rtm[np.newaxis, ...].astype(np.float32)
            dtm_b = dtm[np.newaxis, ...].astype(np.float32)
            logits = self.session.run([self.out_name], {
                self.rtmin_name: rtm_b,
                self.dtmin_name: dtm_b,
            })[0][0].astype(np.float32)
            probs = softmax(logits); top = int(np.argmax(probs))
            self.result_ready.emit(probs, logits, CLASSES_KO[top], CLASSES_EN[top], rtm0, dtm0)
    def stop(self):
        self._running = False

# ============================ TTS Worker ===================================
class TTSWorker(QtCore.QThread):
    def __init__(self, rate: int | None = None, volume: float = 1.0):
        super().__init__()
        self._queue: Deque[str] = deque()
        self._qlock = threading.Lock()
        self._running = True
        self.max_secs = float(os.environ.get("TTS_MAX_SECS", "2.9"))
        self._default_rate = int(os.environ.get("TTS_RATE", str(rate if rate is not None else 210)))
        self._engine = pyttsx3.init()
        try:
            ko_voice_id = None
            for v in self._engine.getProperty("voices"):
                meta = " ".join([
                    getattr(v, "name", "") or "",
                    getattr(v, "id", "") or "",
                    " ".join(getattr(v, "languages", []) or []),
                ]).lower()
                if "ko" in meta or "korean" in meta or "kor" in meta:
                    ko_voice_id = v.id; break
            if ko_voice_id: self._engine.setProperty("voice", ko_voice_id)
        except Exception: pass
        self._engine.setProperty("rate", self._default_rate)
        self._engine.setProperty("volume", volume)
    def enqueue(self, text: str):
        if not text: return
        with self._qlock: self._queue.append(text)
    def _speak_with_timeout(self, text: str):
        try:
            n = len(text)
            if n > 12:   self._engine.setProperty("rate", min(self._default_rate + 40, 260))
            elif n > 8: self._engine.setProperty("rate", min(self._default_rate + 20, 240))
            else:       self._engine.setProperty("rate", self._default_rate)
        except Exception: pass
        self._engine.say(text); start = time.time()
        try:
            self._engine.startLoop(False)
            while self._running and self._engine.isBusy():
                if (time.time() - start) >= self.max_secs:
                    try: self._engine.endLoop()
                    except Exception: pass
                    try:
                        if self._engine.isBusy(): self._engine.stop()
                    except Exception: pass
                    return
                try: self._engine.iterate()
                except Exception: break
                time.sleep(0.01)
        finally:
            try: self._engine.endLoop()
            except Exception: pass
    def run(self):
        while self._running:
            text = None
            with self._qlock:
                if self._queue: text = self._queue.popleft()
            if text is None:
                self.msleep(20); continue
            try: self._speak_with_timeout(text)
            except Exception: pass
    def stop(self):
        self._running = False
        try:
            self._engine.stop()
        except Exception: pass

# ============================ Axes & Image =================================
_MAJOR_SPACING = 1.0
_MINOR_SPACING = 0.5

class FixedAxisItem(pg.AxisItem):
    def __init__(self, orientation: str,
                 major: List[float], minor: List[float] | None = None,
                 label_map: dict[float, str] | None = None):
        super().__init__(orientation=orientation)
        self._major = list(major)
        self._minor = list(minor or [])
        self._labels = {v: (f"{v:g}") for v in self._major}
        if label_map:
            self._labels.update(label_map)

    def tickValues(self, minVal, maxVal, size):
        out = []
        if self._minor:
            out.append((_MINOR_SPACING, [v for v in self._minor if (minVal-1e9) <= v <= (maxVal+1e9)]))
        if self._major:
            out.append((_MAJOR_SPACING, [v for v in self._major if (minVal-1e9) <= v <= (maxVal+1e9)]))
        return out

    def tickStrings(self, values, scale, spacing):
        if spacing == _MAJOR_SPACING:
            return [self._labels.get(v, "") for v in values]
        return [""] * len(values)

class AxisImage(pg.GraphicsLayoutWidget):
    """
    set_image 후 좌표 고정, ViewBox 픽셀 종횡비를 display_aspect_xy(W/H)로 고정.
    """
    def __init__(self, x_range: Tuple[float,float], y_range: Tuple[float,float],
                 x_label: str, y_label: str,
                 x_major: List[float], y_major: List[float],
                 x_minor: List[float] | None = None, y_minor: List[float] | None = None,
                 lock_aspect: bool = True,
                 display_aspect_xy: float = 1.0,
                 label_offset_px: int = 26, tick_font_pt: float = 9.0,
                 tick_text_offset_px: int = 12,
                 extra_margins: Tuple[int,int,int,int] = (12, 22, 12, 22),
                 rotation_k: int = 0, transpose: bool = False,
                 flip_x: bool = False, flip_y: bool = False):
        super().__init__()
        self._rotation_k = int(rotation_k) % 4
        self._transpose = bool(transpose)
        self._flip_x = bool(flip_x)
        self._flip_y = bool(flip_y)
        self._display_aspect = float(display_aspect_xy)

        self.setContentsMargins(0,0,0,0)
        self.ci.layout.setContentsMargins(0,0,0,0)
        self.ci.layout.setSpacing(0)

        axis_bottom = FixedAxisItem('bottom', major=x_major, minor=(x_minor or []))
        axis_left   = FixedAxisItem('left',   major=y_major, minor=(y_minor or []))

        self.plot: pg.PlotItem = self.addPlot(axisItems={'bottom': axis_bottom, 'left': axis_left})
        self.plot.setMenuEnabled(False); self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setClipToView(True); self.plot.hideButtons(); self.plot.showGrid(x=False, y=False)

        l,t,r,b = extra_margins
        self.plot.layout.setContentsMargins(l,t,r,b)
        tick_font = QtGui.QFont("Noto Sans CJK KR", int(tick_font_pt))
        for name in ('bottom','left'):
            ax = self.plot.getAxis(name)
            ax.setStyle(tickFont=tick_font, tickTextOffset=tick_text_offset_px,
                        autoExpandTextSpace=True, showValues=True)
        self.plot.getAxis('bottom').setLabel(x_label, **{'color':'#aaa', 'font-size':'13pt'})
        self.plot.getAxis('left').setLabel(y_label, **{'color':'#aaa', 'font-size':'13pt'})
        self.plot.getAxis('bottom').setHeight(46)
        self.plot.getAxis('left').setWidth(58)

        self.vb: pg.ViewBox = self.plot.getViewBox()
        self.vb.setBorder(None)

        self.img = pg.ImageItem(axisOrder='row-major')
        self.img.setLevels((0.0, 1.0))
        lut = (matplotlib.colormaps['turbo'](np.linspace(0,1,256))[:,:3]*255).astype(np.ubyte)
        self.img.setLookupTable(lut)
        self.plot.addItem(self.img)

        self._x_range = x_range; self._y_range = y_range
        self._apply_ranges_and_aspect(lock_aspect=True)

        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def _apply_ranges_and_aspect(self, lock_aspect: bool):
        x0,x1 = self._x_range; y0,y1 = self._y_range
        self.plot.setLimits(xMin=min(x0,x1), xMax=max(x0,x1),
                            yMin=min(y0,y1), yMax=max(y0,y1))
        self.vb.enableAutoRange(x=False, y=False)
        self.vb.setDefaultPadding(0.0)
        self.vb.setRange(xRange=(x0,x1), yRange=(y0,y1), padding=0.0)

        xspan = (x1-x0); yspan = (y1-y0)
        if lock_aspect:
            ratio = max(1e-9, self._display_aspect * (yspan / xspan))
            self.vb.setAspectLocked(True, ratio=ratio)
        else:
            self.vb.setAspectLocked(False)

    def _orient(self, arr: np.ndarray) -> np.ndarray:
        x = arr
        if self._transpose: x = x.T
        if self._rotation_k: x = np.rot90(x, k=self._rotation_k)
        if self._flip_x: x = np.flip(x, axis=1)
        if self._flip_y: x = np.flip(x, axis=0)
        return x

    def set_image(self, arr: np.ndarray):
        arr2 = self._orient(arr)
        self.img.setImage(arr2, autoLevels=False, autoDownsample=True)
        x0,x1 = self._x_range; y0,y1 = self._y_range
        self.img.setRect(QtCore.QRectF(x0, y0, (x1-x0), (y1-y0)))
        self._apply_ranges_and_aspect(lock_aspect=True)

# ============================ Simple Widgets ===============================
class TerminalWidget(QtWidgets.QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.document().setMaximumBlockCount(3000)
        self.setFont(QtGui.QFont("Consolas", 12))
    def append_line(self, text: str):
        self.appendPlainText(text)
        sb = self.verticalScrollBar(); sb.setValue(sb.maximum())

class Banner(QtWidgets.QLabel):
    def __init__(self):
        super().__init__("—")
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setFont(QtGui.QFont("Noto Sans CJK KR", 50, QtGui.QFont.Bold))
        self.setStyleSheet("padding:12px; border:5px solid #444; border-radius:12px;"
                           "background-color:#0b0b0b; color:#f2f2f2;")
    def set_text(self, text: str):
        self.setText(text)

class PhaseStrip(QtWidgets.QLabel):
    class Mode(Enum): CAPTURE = auto(); PAUSE_INFER = auto(); PAUSED = auto()
    def __init__(self):
        super().__init__("—")
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setFont(header_font(HEADER_FONT_PT))  # 헤더와 동일 폰트
        self.setStyleSheet("padding:6px; border:1px solid #666; border-radius:8px;"
                           "background:#111; color:#e0e0e0;")
        self.setFixedHeight(UNIFIED_BAR_H)
        self._mode = PhaseStrip.Mode.PAUSED
        self._dot = 0
        self._timer = QtCore.QTimer(self); self._timer.setInterval(400)
        self._timer.timeout.connect(self._on_tick); self._timer.start()
    def set_mode(self, mode: "PhaseStrip.Mode"):
        self._mode = mode; self._dot = 0; self._render()
    def _on_tick(self):
        if self._mode == PhaseStrip.Mode.CAPTURE or self._mode == PhaseStrip.Mode.PAUSE_INFER:
            self._dot = (self._dot + 1) % 3; self._render()
    def _render(self):
        if self._mode == PhaseStrip.Mode.CAPTURE:
            dots = "." * (self._dot + 1)
            self.setText(f"Next Sign Sensing{dots}")
            self.setStyleSheet("padding:6px; border:1px solid #3b6ea5; border-radius:8px;"
                               "background:#0f1b2b; color:#cfe4ff;")
        elif self._mode == PhaseStrip.Mode.PAUSE_INFER:
            dots = "." * (self._dot + 1)
            self.setText(f"Model Inference{dots}")
            self.setStyleSheet("padding:6px; border:1px solid #8a6d00; border-radius:8px;"
                               "background:#2a2200; color:#ffe38a;")
        else:
            self.setText("—")
            self.setStyleSheet("padding:6px; border:1px solid #666; border-radius:8px;"
                               "background:#111; color:#9a9a9a;")

# ============================ Header Bar (원래 모양 복원) ============================
class HeaderTheme(Enum):
    DEFAULT = auto()  # 검정 배경 + 흰 글자
    BLUE    = auto()  # CAPTURE 톤
    YELLOW  = auto()  # PAUSE_INFER 톤

def _theme_colors(theme: HeaderTheme):
    if theme == HeaderTheme.BLUE:
        return {"bg": QtGui.QColor(15, 27, 43), "fg": QtGui.QColor(207, 228, 255)}  # #0f1b2b / #cfe4ff
    if theme == HeaderTheme.YELLOW:
        return {"bg": QtGui.QColor(42, 34, 0), "fg": QtGui.QColor(255, 227, 138)}  # #2a2200 / #ffe38a
    return {"bg": QtGui.QColor(0, 0, 0), "fg": QtGui.QColor(255, 255, 255)}        # DEFAULT

class HeaderBar(QtWidgets.QWidget):
    """
    - 원래처럼 '직사각형 + 딱 맞는 높이' 유지
    - 테두리/라운드/여백 추가 없음
    - 상태(Phase)에 따라 배경/글자색만 변경
    """
    def __init__(self, left_text: str = "", right_text: str | None = None,
                 height: int = UNIFIED_BAR_H, font_pt: int = HEADER_FONT_PT):
        super().__init__()
        self._font_pt = font_pt
        self._theme = HeaderTheme.DEFAULT

        self.setAutoFillBackground(True)
        self.setMinimumHeight(height)
        self.setMaximumHeight(height)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(8, 0, 8, 0)   # 상하 여백 0으로 딱 붙임
        lay.setSpacing(8)

        def mk_label(txt: str) -> QtWidgets.QLabel:
            lb = QtWidgets.QLabel(txt)
            lb.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
            lb.setFont(header_font(self._font_pt))
            return lb

        self.left = mk_label(left_text)
        self.right = None

        if right_text is None:
            self.left.setAlignment(QtCore.Qt.AlignCenter)
            self.left.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            lay.addStretch(1)
            lay.addWidget(self.left)
            lay.addStretch(1)
        else:
            lay.addWidget(self.left, 0, QtCore.Qt.AlignLeft)
            lay.addStretch(1)
            self.right = mk_label(right_text)
            self.right.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)
            lay.addWidget(self.right, 0, QtCore.Qt.AlignRight)

        self._apply_theme()

    def _apply_theme(self):
        colors = _theme_colors(self._theme)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), colors["bg"])
        self.setPalette(pal)

        def apply_label_color(lb: QtWidgets.QLabel | None):
            if lb is None: return
            lb.setStyleSheet(f"color: rgb({colors['fg'].red()},{colors['fg'].green()},{colors['fg'].blue()});")

        apply_label_color(self.left)
        apply_label_color(self.right)

    def set_theme(self, theme: HeaderTheme):
        if theme == self._theme:
            return
        self._theme = theme
        self._apply_theme()

# ============================ State Machine ================================
class Phase(Enum): CAPTURE = auto(); PAUSE_INFER = auto()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"IFX Radar — {WINDOW_SEC:.0f}s Capture / {PAUSE_SEC:.0f}s Pause+Infer (ONNX + TTS)")
        self.resize(960, 540)

        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        grid = QtWidgets.QGridLayout(central)
        grid.setContentsMargins(8,8,8,8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        # ===== Row 0: 헤더/버튼 =====
        self.left_header = HeaderBar("Real–Time Range–Doppler Image (Rx0)",
                                     height=UNIFIED_BAR_H, font_pt=HEADER_FONT_PT)
        grid.addWidget(self.left_header, 0, 0, 1, 1)

        self.rtm_header = HeaderBar("Range–Time Map (Rx0)",
                                    height=UNIFIED_BAR_H, font_pt=HEADER_FONT_PT)
        grid.addWidget(self.rtm_header, 0, 1, 1, 1)

        self.ctrl = QtWidgets.QWidget()
        ch = QtWidgets.QHBoxLayout(self.ctrl); ch.setContentsMargins(0,0,0,0); ch.setSpacing(8)
        self.btn_start  = QtWidgets.QPushButton("Start")
        self.btn_pause  = QtWidgets.QPushButton("Pause")
        self.btn_resume = QtWidgets.QPushButton("Resume")
        self.btn_stop   = QtWidgets.QPushButton("Stop")
        for btn in [self.btn_start,self.btn_pause,self.btn_resume,self.btn_stop]:
            btn.setFixedSize(BTN_W, UNIFIED_BAR_H)
            btn.setFont(QtGui.QFont("Noto Sans CJK KR", 20, QtGui.QFont.Bold))
            ch.addWidget(btn)
        self.chk_tts = QtWidgets.QCheckBox("TTS (KO) On"); self.chk_tts.setChecked(True)
        ch.addWidget(self.chk_tts); ch.addStretch(1)
        grid.addWidget(self.ctrl, 0, 2, 1, 1)

        # ===== Row 1: RDI/RTM/Console =====
        self.live = AxisImage(
            x_range=(0.0, RANGE_MAX),
            y_range=(-VEL_MAX, VEL_MAX),
            x_label="Distance (m)", y_label="Velocity (m/s)",
            x_major=RANGE_TICKS_MAJOR, y_major=VEL_TICKS_MAJOR,
            lock_aspect=True, display_aspect_xy=ASPECT_RDI_XY, tick_font_pt=10.0
        )
        grid.addWidget(self.live, 1, 0, 3, 1)

        self.rtm_view = AxisImage(
            x_range=(0.0, TIME_MAX),
            y_range=(-0.06, RANGE_MAX),
            x_label="Time (s)", y_label="Distance (m)",
            x_major=TIME_TICKS_MAJOR, y_major=RANGE_TICKS_MAJOR,
            lock_aspect=True, display_aspect_xy=ASPECT_SQUARE, tick_font_pt=10.0
        )
        self.rtm_view.vb.invertY(True)
        grid.addWidget(self.rtm_view, 1, 1, 1, 1)

        right_stack = QtWidgets.QWidget()
        rs = QtWidgets.QVBoxLayout(right_stack); rs.setContentsMargins(0,0,0,0); rs.setSpacing(8)
        self.term = TerminalWidget();  rs.addWidget(self.term, 1)
        grid.addWidget(right_stack, 1, 2, 1, 1)

        # ===== Row 2: DTM 헤더 / PhaseStrip =====
        self.dtm_header = HeaderBar("Doppler–Time Map (Rx0)",
                                    height=UNIFIED_BAR_H, font_pt=HEADER_FONT_PT)
        grid.addWidget(self.dtm_header, 2, 1, 1, 1)

        self.phase_strip = PhaseStrip()
        grid.addWidget(self.phase_strip, 2, 2, 1, 1)

        # ===== Row 3: DTM / Banner =====
        self.dtm_view = AxisImage(
            x_range=(0.0, TIME_MAX),
            y_range=(-VEL_MAX, VEL_MAX),
            x_label="Time (s)", y_label="Velocity (m/s)",
            x_major=TIME_TICKS_MAJOR, y_major=VEL_TICKS_MAJOR,
            lock_aspect=True, display_aspect_xy=ASPECT_SQUARE, tick_font_pt=10.0
        )
        grid.addWidget(self.dtm_view, 3, 1, 1, 1)

        self.banner = Banner()
        grid.addWidget(self.banner, 3, 2, 1, 1)

        # ===== Column/Row stretch =====
        grid.setColumnStretch(0, 3)  # RDI
        grid.setColumnStretch(1, 2)  # RTM/DTM
        grid.setColumnStretch(2, 4)  # Controls/Console/Etc

        grid.setRowStretch(0, 0)  # 헤더/버튼
        grid.setRowStretch(1, 1)  # RTM 플롯
        grid.setRowStretch(2, 0)  # 중간 헤더/스트립
        grid.setRowStretch(3, 1)  # DTM 플롯

        self.statusBar().setStyleSheet("font-size: 12pt;")
        self.statusBar().showMessage("Idle — press Start")

        # ====== Beep 준비: WAV 생성 + QSoundEffect 설정 ======
        self._beep_path = os.path.join(BASE_DIR, "beep_1000Hz_120ms.wav")
        try:
            ensure_beep_wav(self._beep_path, freq=1000, dur_ms=120, sr=44100)
        except Exception as e:
            print(f"[WARN] beep wav create failed: {e}")
        self._beep = QSoundEffect(self)
        try:
            self._beep.setSource(QtCore.QUrl.fromLocalFile(self._beep_path))
            self._beep.setVolume(0.9)  # 0.0~1.0
            # macOS/일부 환경에선 첫 play()가 늦게 로드되니 미리 로드 유도
            self._beep.setLoopCount(1)
        except Exception as e:
            print(f"[WARN] QSoundEffect init failed: {e}")

        # Device
        self.device = DeviceFmcw()
        self.sequence = self.device.create_simple_sequence(RADAR_CONFIG)
        self.device.set_acquisition_sequence(self.sequence)
        self.term.append_line(f"SDK: {get_version_full()} / Sensor: {self.device.get_sensor_type()}")
        self.term.append_line("Inference Model: [SpectraNet]")
        self.term.append_line("="*70)

        # Timers
        self.ui_timer = QtCore.QTimer(self); self.ui_timer.setInterval(int(1000 / UI_FPS))
        self.ui_timer.timeout.connect(self._on_ui_tick)
        self.acq_timer = QtCore.QTimer(self)
        self.acq_timer.setInterval(int(RADAR_CONFIG.frame_repetition_time_s * 1000))
        self.acq_timer.timeout.connect(self._poll_frame)
        self.phase_timer = QtCore.QTimer(self); self.phase_timer.setSingleShot(True)
        self.phase_timer.timeout.connect(self._next_phase)

        # Workers
        self.inf = InferenceWorker()
        self.inf.result_ready.connect(self._on_infer_done)
        self.inf.start()

        self.tts = TTSWorker(rate=None, volume=1.0); self.tts.start()
        self._tts_blocklist = {"<무동작>"}; self._tts_alias = {}

        # State
        self.frames_buf: List[np.ndarray] = []
        self.live_queue: Deque[np.ndarray] = deque()
        self.phase: Phase | None = None
        self._blink = False; self._cycle = 0; self._paused = True

        self._live_preview_ctr = 0; self.LIVE_PREVIEW_EVERY = 2

        self.banner.set_text("Press Start!!")
        self.phase_strip.set_mode(PhaseStrip.Mode.PAUSED)

        self.btn_start.clicked.connect(self._on_start_clicked)
        self.btn_pause.clicked.connect(self._on_pause_clicked)
        self.btn_resume.clicked.connect(self._on_resume_clicked)
        self.btn_stop.clicked.connect(self._on_stop_clicked)

        for i, (en, ko) in enumerate(zip(CLASSES_EN, CLASSES_KO)):
            self.term.append_line(f"Label {i:02d}: \"{en}\" / {ko}")
        self.term.append_line("="*70)

        # 초기 헤더 테마
        self._apply_header_theme_for_phase(None)

    # ------------------------ Beep 재생 -----------------
    def _play_beep(self):
        try:
            if self._beep.source().isEmpty():
                raise RuntimeError("QSoundEffect source empty")
            self._beep.stop()     # 혹시 재생중이면 중복 방지
            self._beep.play()
            # QSoundEffect가 로드를 못하면 아래로 폴백
            if self._beep.isLoaded() is False:
                raise RuntimeError("QSoundEffect not loaded")
        except Exception:
            # 폴백: winsound → QApplication.beep()
            try:
                if platform.system().lower().startswith("win"):
                    import winsound
                    winsound.Beep(1000, 120)
                else:
                    QtWidgets.QApplication.beep()
            except Exception:
                try:
                    QtWidgets.QApplication.beep()
                except Exception:
                    pass

    # ------------------------ Header theme control -----------------
    def _apply_header_theme_for_phase(self, phase: Phase | None):
        """
        CAPTURE:    RDI 헤더=파랑, RTM/DTM=검정
        PAUSE_INFER: RDI 헤더=검정, RTM/DTM=노랑
        기타:       모두 검정
        """
        if phase == Phase.CAPTURE:
            self.left_header.set_theme(HeaderTheme.BLUE)
            self.rtm_header.set_theme(HeaderTheme.DEFAULT)
            self.dtm_header.set_theme(HeaderTheme.DEFAULT)
        elif phase == Phase.PAUSE_INFER:
            self.left_header.set_theme(HeaderTheme.DEFAULT)
            self.rtm_header.set_theme(HeaderTheme.YELLOW)
            self.dtm_header.set_theme(HeaderTheme.YELLOW)
        else:
            self.left_header.set_theme(HeaderTheme.DEFAULT)
            self.rtm_header.set_theme(HeaderTheme.DEFAULT)
            self.dtm_header.set_theme(HeaderTheme.DEFAULT)

    # ------------------------ Phase control ------------------------
    def _set_phase(self, phase: Phase):
        if self._paused: return
        self.phase = phase
        self._apply_header_theme_for_phase(phase)

        if phase == Phase.CAPTURE:
            # 센싱 시작 ‘삑’ 비프음
            self._play_beep()

            try: self.device.set_acquisition_sequence(self.sequence)
            except Exception as e: self.term.append_line(f"[WARN] set_acquisition_sequence: {e}")
            self.frames_buf.clear(); self.live_queue.clear()
            self.phase_strip.set_mode(PhaseStrip.Mode.CAPTURE)
            if self._cycle == 0:
                self.term.append_line("[CAPTURE] Sensing started (4s)")
            else:
                self.term.append_line("="*70)
                self.term.append_line("[CAPTURE] Next Sign Sensing... (4s)")
            if not self.ui_timer.isActive(): self.ui_timer.start()
            if not self.acq_timer.isActive(): self.acq_timer.start()
            self.phase_timer.start(int(WINDOW_SEC * 1000))
        elif phase == Phase.PAUSE_INFER:
            try:
                self.acq_timer.stop(); self.device.stop_acquisition()
            except Exception: pass
            self.phase_strip.set_mode(PhaseStrip.Mode.PAUSE_INFER)
            self.term.append_line(f"[PAUSE] Preprocessing & Model Inference ({PAUSE_SEC:.0f}s)")
            if self.frames_buf:
                self.inf.enqueue(InferenceJob(window_frames=list(self.frames_buf)))
            self.phase_timer.start(int(PAUSE_SEC * 1000)); self._cycle += 1

    def _next_phase(self):
        if self._paused: return
        self._set_phase(Phase.CAPTURE if self.phase == Phase.PAUSE_INFER else Phase.PAUSE_INFER)

    # ------------------------ Acquisition & UI ----------------------
    @QtCore.pyqtSlot()
    def _poll_frame(self):
        try:
            frame_list = self.device.get_next_frame()
        except Exception as e:
            self.term.append_line(f"[ERR] get_next_frame: {e}")
            return
        for fr in frame_list:
            self.frames_buf.append(np.asarray(fr))
            self._live_preview_ctr += 1
            if (self._live_preview_ctr % self.LIVE_PREVIEW_EVERY) == 0:
                while len(self.live_queue) > 0: self.live_queue.popleft()
                self.live_queue.append(rd_live_from_frame(fr))

    @QtCore.pyqtSlot()
    def _on_ui_tick(self):
        if self._paused: return
        self._blink = not self._blink
        if self.phase == Phase.CAPTURE:
            if self._blink:
                msg = "Next Sign Sensing…" if self._cycle >= 1 else "Sensing…"
                self.statusBar().showMessage(msg)
        else:
            if self._blink: self.statusBar().showMessage("Model Inference")
        if self.live_queue:
            rd = self.live_queue.popleft()
            self.live.set_image(rd)

    @QtCore.pyqtSlot(np.ndarray, np.ndarray, str, str, np.ndarray, np.ndarray)
    def _on_infer_done(self, probs: np.ndarray, logits: np.ndarray, top_ko: str, top_en: str,
                       rtm0: np.ndarray, dtm0: np.ndarray):
        if self._paused: return
        ts = time.strftime("%H:%M:%S")
        lines = [
            f'{CLASSES_NUM[i]} ("{CLASSES_EN[i]}"): logit={logits[i]:+7.3f}  prob={probs[i]:.3f}'
            for i in range(len(probs))
        ]
        self.term.append_line(f'[{ts}] TOP = {top_en} / {top_ko}\n  ' + "\n  ".join(lines))
        self.banner.set_text(f"{top_ko}\n{top_en}")
        try:
            self.rtm_view.set_image(rtm0)
            self.dtm_view.set_image(dtm0)
        except Exception:
            pass
        if self.chk_tts.isChecked() and (top_ko not in self._tts_blocklist):
            self.tts.enqueue(top_ko)

    # ------------------------ Controls -----------------------------
    def _on_start_clicked(self):
        if not self._paused: return
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
        if self._paused: return
        self._paused = True
        try:
            self.phase_timer.stop(); self.acq_timer.stop(); self.ui_timer.stop()
            self.device.stop_acquisition()
        except Exception: pass
        self.statusBar().showMessage("Paused — press Resume")
        self.banner.set_text("Paused")
        self.phase_strip.set_mode(PhaseStrip.Mode.PAUSED)
        self.term.append_line("[PAUSE] All stopped by user")
        self.btn_start.setEnabled(False); self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(True); self.btn_stop.setEnabled(True)
        self._apply_header_theme_for_phase(None)

    def _on_resume_clicked(self):
        if not self._paused: return
        self._paused = False
        self.btn_start.setEnabled(False); self.btn_pause.setEnabled(True)
        self.btn_resume.setEnabled(False); self.btn_stop.setEnabled(True)
        self.statusBar().showMessage("Running")
        self.phase_strip.set_mode(PhaseStrip.Mode.CAPTURE)
        self._set_phase(Phase.CAPTURE)

    def _on_stop_clicked(self):
        self.close()

    def closeEvent(self, e: QtGui.QCloseEvent):
        try:
            self.phase_timer.stop(); self.acq_timer.stop(); self.ui_timer.stop()
            self.inf.stop(); self.inf.wait(1000)
            try: self.device.stop_acquisition()
            except Exception: pass
            try: self.device.close()
            except Exception: pass
            try: self.tts.stop(); self.tts.wait(1000)
            except Exception: pass
        finally:
            return super().closeEvent(e)

# ============================ Entrypoint ==================================
def main():
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder='row-major', antialias=True)
    win = MainWindow(); win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
