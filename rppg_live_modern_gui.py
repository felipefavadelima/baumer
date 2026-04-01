from __future__ import annotations

"""
rPPG ao vivo com interface moderna (PySide6 + pyqtgraph).

Recursos principais
-------------------
- Captura ao vivo via câmera Baumer (neoapi) ou webcam OpenCV.
- Seleção de ROI diretamente sobre o vídeo com o mouse.
- Exibição contínua do sinal rPPG usando o algoritmo POS.
- FFT/BPM atualizada periodicamente (padrão: a cada 10 s).
- ROI sempre desenhada sobre a imagem.
- Processamento desacoplado da GUI por thread dedicada.

Dependências
------------
pip install numpy scipy opencv-python PySide6 pyqtgraph
# Opcional para câmera Baumer:
# instalar neoapi conforme o SDK da Baumer

Uso
---
python rppg_live_modern_gui.py
"""

import math
import sys
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import pyqtgraph as pg
from scipy.signal import butter, detrend, filtfilt
from scipy.fft import rfft, rfftfreq
from PySide6 import QtCore, QtGui, QtWidgets


# ==============================
# Configurações padrão
# ==============================
DEFAULT_SOURCE = "Baumer (neoAPI)"
DEFAULT_CAMERA_INDEX = 0
DEFAULT_TARGET_FPS = 45.0
DEFAULT_MAX_DISPLAY_HEIGHT = 720
DEFAULT_SIGNAL_WINDOW_S = 15.0
DEFAULT_PROCESS_BUFFER_S = 30.0
DEFAULT_FFT_WINDOW_S = 10.0
DEFAULT_FFT_UPDATE_EVERY_S = 10.0
DEFAULT_POS_WINDOW_S = 1.6
DEFAULT_BANDPASS_LOW_HZ = 0.75
DEFAULT_BANDPASS_HIGH_HZ = 3.0
DEFAULT_BPM_MIN = 40.0
DEFAULT_BPM_MAX = 180.0
DEFAULT_ROI = (100, 100, 220, 220)


# ==============================
# Utilidades de processamento
# ==============================

def clamp_roi(roi: Tuple[int, int, int, int], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    x, y, w, h = [int(v) for v in roi]
    if width <= 0 or height <= 0:
        return None
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    if w < 2 or h < 2:
        return None
    return x, y, w, h


def estimate_fps(timestamps: np.ndarray, fallback: float) -> float:
    if timestamps.size < 3:
        return fallback
    dt = np.diff(timestamps)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return fallback
    fps = 1.0 / np.median(dt)
    if not np.isfinite(fps) or fps <= 0:
        return fallback
    return float(fps)


def resample_rgb(rgb: np.ndarray, timestamps: np.ndarray, fps: float) -> Tuple[np.ndarray, np.ndarray]:
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError("rgb deve ter shape (N, 3)")
    if timestamps.size != rgb.shape[0]:
        raise ValueError("timestamps e rgb devem ter o mesmo tamanho")
    if timestamps.size < 2:
        return rgb.copy(), timestamps.copy()

    t0 = float(timestamps[0])
    t1 = float(timestamps[-1])
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return rgb.copy(), timestamps.copy()

    step = 1.0 / max(fps, 1e-6)
    t_uniform = np.arange(t0, t1 + 0.5 * step, step, dtype=np.float64)
    if t_uniform.size < 2:
        return rgb.copy(), timestamps.copy()

    rgb_uniform = np.empty((t_uniform.size, 3), dtype=np.float64)
    for c in range(3):
        rgb_uniform[:, c] = np.interp(t_uniform, timestamps, rgb[:, c])
    return rgb_uniform, t_uniform


def pos_algorithm(rgb: np.ndarray, fps: float, window_sec: float = DEFAULT_POS_WINDOW_S) -> np.ndarray:
    """
    Implementação deslizante do algoritmo POS.
    rgb: array (N, 3) com colunas [R, G, B].
    """
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError("rgb deve ter shape (N, 3)")

    n = rgb.shape[0]
    window = max(2, int(round(window_sec * fps)))
    if n < window + 1:
        return np.zeros(n, dtype=np.float64)

    x = rgb.T.astype(np.float64, copy=False)
    h_out = np.zeros(n, dtype=np.float64)
    proj_mat = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]], dtype=np.float64)

    for end in range(window, n + 1):
        c = x[:, end - window : end]
        mean = np.mean(c, axis=1, keepdims=True)
        mean[mean == 0.0] = 1.0
        c_norm = c / mean - 1.0

        s = proj_mat @ c_norm
        s1 = s[0]
        s2 = s[1]
        std2 = np.std(s2)
        alpha = 0.0 if std2 < 1e-12 else np.std(s1) / std2
        h = s1 + alpha * s2
        h -= np.mean(h)
        h_out[end - window : end] += h

    return h_out


def bandpass_filter(signal: np.ndarray, fps: float, low_hz: float, high_hz: float, order: int = 3) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size < 16:
        return signal - np.mean(signal) if signal.size else signal

    nyq = 0.5 * fps
    high_hz = min(high_hz, 0.99 * nyq)
    low_hz = max(0.01, low_hz)
    if low_hz >= high_hz:
        return signal - np.mean(signal)

    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="bandpass")
    padlen = 3 * (max(len(a), len(b)) - 1)
    centered = detrend(signal)
    if centered.size <= padlen + 1:
        return centered
    return filtfilt(b, a, centered)


def compute_fft_bpm(
    signal: np.ndarray,
    fps: float,
    bpm_min: float,
    bpm_max: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size < 16:
        return np.array([]), np.array([]), math.nan, math.nan

    windowed = (signal - np.mean(signal)) * np.hanning(signal.size)
    spec = np.abs(rfft(windowed))
    freqs_hz = rfftfreq(signal.size, d=1.0 / fps)
    freqs_bpm = freqs_hz * 60.0

    band = (freqs_bpm >= bpm_min) & (freqs_bpm <= bpm_max)
    if not np.any(band):
        return freqs_bpm, spec, math.nan, math.nan

    spec_band = spec[band]
    bpm_band = freqs_bpm[band]
    peak_idx = int(np.argmax(spec_band))
    bpm_peak = float(bpm_band[peak_idx])

    peak_mask = np.abs(bpm_band - bpm_peak) <= 3.0
    p_signal = float(np.sum(spec_band[peak_mask] ** 2))
    p_noise = float(np.sum(spec_band[~peak_mask] ** 2))
    snr_db = math.inf if p_noise <= 1e-15 else 10.0 * math.log10((p_signal + 1e-12) / (p_noise + 1e-12))
    return freqs_bpm, spec, bpm_peak, snr_db


# ==============================
# Backends de captura
# ==============================

@dataclass
class FramePacket:
    display_bgr8: np.ndarray
    analysis_bgr: np.ndarray
    source_size: Tuple[int, int]
    timestamp_s: float


class BaseBackend:
    name = "Base"

    def open(self) -> None:
        raise NotImplementedError

    def read(self) -> Optional[FramePacket]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class OpenCVBackend(BaseBackend):
    name = "Webcam (OpenCV)"

    def __init__(self, camera_index: int = 0, target_fps: float = 30.0, max_display_height: int = 720):
        self.camera_index = int(camera_index)
        self.target_fps = float(target_fps)
        self.max_display_height = int(max_display_height)
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a webcam índice {self.camera_index}.")
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

    def read(self) -> Optional[FramePacket]:
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        display = self._resize_display(frame)
        h, w = frame.shape[:2]
        return FramePacket(
            display_bgr8=display,
            analysis_bgr=frame,
            source_size=(w, h),
            timestamp_s=time.perf_counter(),
        )

    def _resize_display(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = min(1.0, self.max_display_height / max(1, h))
        if scale >= 0.999:
            return frame
        return cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LINEAR)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class BaumerNeoAPIBackend(BaseBackend):
    name = "Baumer (neoAPI)"

    def __init__(self, target_fps: float = 45.0, max_display_height: int = 720):
        self.target_fps = float(target_fps)
        self.max_display_height = int(max_display_height)
        self.neoapi = None
        self.cam = None
        self.analysis_conversion = None

    def open(self) -> None:
        try:
            import neoapi  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "neoapi não está disponível neste Python. Instale o SDK/neoAPI da Baumer para usar esta opção."
            ) from exc

        self.neoapi = neoapi
        self.cam = neoapi.Cam()
        self.cam.Connect()

        try:
            self.cam.f.TriggerMode.value = neoapi.TriggerMode_Off
        except Exception:
            pass
        try:
            self.cam.f.ExposureAuto.value = neoapi.ExposureAuto_Off
        except Exception:
            pass
        try:
            self.cam.f.GainAuto.value = neoapi.GainAuto_Off
        except Exception:
            pass
        try:
            self.cam.f.AcquisitionFrameRateEnable.value = True
            self.cam.f.AcquisitionFrameRate.value = self.target_fps
        except Exception:
            pass

        # Mantém a câmera em Bayer 12-bit quando possível, como no script de captura.
        try:
            self.cam.f.PixelFormat.value = neoapi.PixelFormat_BayerRG12p
        except Exception:
            pass

        # Decide qual conversão usar para análise: tenta 16-bit primeiro, cai para 8-bit.
        self.analysis_conversion = "BGR16"
        probe = self.cam.GetImage()
        if probe.IsEmpty():
            raise RuntimeError("Falha ao obter frame inicial da câmera Baumer.")
        try:
            arr = probe.Convert(self.analysis_conversion).GetNPArray()
            if arr is None or arr.size == 0:
                raise RuntimeError("Conversão BGR16 vazia.")
        except Exception:
            self.analysis_conversion = "BGR8"

    def read(self) -> Optional[FramePacket]:
        if self.cam is None:
            return None
        img = self.cam.GetImage()
        if img.IsEmpty():
            return None

        display = img.Convert("BGR8").GetNPArray()
        if display is None or display.size == 0:
            return None

        try:
            analysis = img.Convert(self.analysis_conversion).GetNPArray()
            if analysis is None or analysis.size == 0:
                analysis = display
        except Exception:
            analysis = display

        if analysis.ndim == 2:
            analysis = cv2.cvtColor(analysis, cv2.COLOR_GRAY2BGR)
        if display.ndim == 2:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

        display_small = self._resize_display(display)
        h, w = analysis.shape[:2]
        return FramePacket(
            display_bgr8=display_small,
            analysis_bgr=analysis,
            source_size=(w, h),
            timestamp_s=time.perf_counter(),
        )

    def _resize_display(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = min(1.0, self.max_display_height / max(1, h))
        if scale >= 0.999:
            return frame
        return cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LINEAR)

    def close(self) -> None:
        if self.cam is not None:
            try:
                if self.cam.IsConnected():
                    self.cam.Disconnect()
            except Exception:
                pass
            self.cam = None


# ==============================
# Thread de captura
# ==============================

class CaptureWorker(QtCore.QObject):
    frameReady = QtCore.Signal(object, float, tuple, object)
    statusMessage = QtCore.Signal(str)
    errorMessage = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(
        self,
        source_name: str,
        camera_index: int,
        target_fps: float,
        max_display_height: int,
    ):
        super().__init__()
        self.source_name = source_name
        self.camera_index = int(camera_index)
        self.target_fps = float(target_fps)
        self.max_display_height = int(max_display_height)
        self._running = False
        self._backend: Optional[BaseBackend] = None
        self._roi_lock = threading.Lock()
        self._roi: Optional[Tuple[int, int, int, int]] = DEFAULT_ROI

    @QtCore.Slot(tuple)
    def set_roi(self, roi: Tuple[int, int, int, int]) -> None:
        with self._roi_lock:
            self._roi = tuple(int(v) for v in roi)

    @QtCore.Slot()
    def clear_roi(self) -> None:
        with self._roi_lock:
            self._roi = None

    def _get_roi(self) -> Optional[Tuple[int, int, int, int]]:
        with self._roi_lock:
            return None if self._roi is None else tuple(self._roi)

    @QtCore.Slot()
    def stop(self) -> None:
        self._running = False

    @QtCore.Slot()
    def run(self) -> None:
        self._running = True
        try:
            if self.source_name == "Baumer (neoAPI)":
                backend = BaumerNeoAPIBackend(
                    target_fps=self.target_fps,
                    max_display_height=self.max_display_height,
                )
            else:
                backend = OpenCVBackend(
                    camera_index=self.camera_index,
                    target_fps=self.target_fps,
                    max_display_height=self.max_display_height,
                )

            self._backend = backend
            backend.open()
            self.statusMessage.emit(f"Fonte ativa: {backend.name}")

            while self._running:
                packet = backend.read()
                if packet is None:
                    continue

                roi = self._get_roi()
                mean_rgb = None
                if roi is not None:
                    w_src, h_src = packet.source_size
                    roi_ok = clamp_roi(roi, w_src, h_src)
                    if roi_ok is not None:
                        x, y, w, h = roi_ok
                        crop = packet.analysis_bgr[y : y + h, x : x + w]
                        if crop.size > 0:
                            mean_bgr = crop.mean(axis=(0, 1))
                            mean_rgb = np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]], dtype=np.float64)

                self.frameReady.emit(packet.display_bgr8, packet.timestamp_s, packet.source_size, mean_rgb)

        except Exception as exc:
            self.errorMessage.emit(str(exc))
        finally:
            try:
                if self._backend is not None:
                    self._backend.close()
            finally:
                self.finished.emit()


# ==============================
# Widget de vídeo com ROI no mouse
# ==============================

class VideoWidget(QtWidgets.QFrame):
    roiChanged = QtCore.Signal(tuple)
    roiCleared = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 420)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setMouseTracking(True)

        self._pixmap: Optional[QtGui.QPixmap] = None
        self._display_size = QtCore.QSize(0, 0)
        self._image_rect = QtCore.QRectF()
        self._source_size = (0, 0)
        self._roi: Optional[Tuple[int, int, int, int]] = DEFAULT_ROI
        self._dragging = False
        self._drag_start_src: Optional[QtCore.QPointF] = None
        self._drag_current_src: Optional[QtCore.QPointF] = None

    def set_frame(self, frame_bgr8: np.ndarray, source_size: Tuple[int, int]) -> None:
        self._source_size = tuple(int(v) for v in source_size)
        rgb = cv2.cvtColor(frame_bgr8, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888).copy()
        self._pixmap = QtGui.QPixmap.fromImage(qimg)
        self._display_size = QtCore.QSize(w, h)
        self.update()

    def roi(self) -> Optional[Tuple[int, int, int, int]]:
        return None if self._roi is None else tuple(self._roi)

    def set_roi(self, roi: Optional[Tuple[int, int, int, int]]) -> None:
        self._roi = roi
        self.update()

    def clear_roi(self) -> None:
        self._roi = None
        self.roiCleared.emit()
        self.update()

    def _get_drawn_image_rect(self) -> QtCore.QRectF:
        if self._pixmap is None:
            return QtCore.QRectF()
        scaled = self._pixmap.size().scaled(self.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        x = (self.width() - scaled.width()) / 2.0
        y = (self.height() - scaled.height()) / 2.0
        return QtCore.QRectF(x, y, scaled.width(), scaled.height())

    def _widget_to_source(self, pos: QtCore.QPointF) -> Optional[QtCore.QPointF]:
        rect = self._get_drawn_image_rect()
        if rect.isNull() or not rect.contains(pos):
            return None
        sw, sh = self._source_size
        if sw <= 0 or sh <= 0:
            return None
        x_rel = (pos.x() - rect.left()) / rect.width()
        y_rel = (pos.y() - rect.top()) / rect.height()
        x_src = x_rel * sw
        y_src = y_rel * sh
        x_src = max(0.0, min(x_src, sw - 1))
        y_src = max(0.0, min(y_src, sh - 1))
        return QtCore.QPointF(x_src, y_src)

    def _source_to_widget_rect(self, roi: Tuple[int, int, int, int]) -> QtCore.QRectF:
        rect = self._get_drawn_image_rect()
        sw, sh = self._source_size
        if rect.isNull() or sw <= 0 or sh <= 0:
            return QtCore.QRectF()
        x, y, w, h = roi
        return QtCore.QRectF(
            rect.left() + (x / sw) * rect.width(),
            rect.top() + (y / sh) * rect.height(),
            (w / sw) * rect.width(),
            (h / sh) * rect.height(),
        )

    def _normalized_roi_from_drag(self) -> Optional[Tuple[int, int, int, int]]:
        if self._drag_start_src is None or self._drag_current_src is None:
            return None
        x0 = self._drag_start_src.x()
        y0 = self._drag_start_src.y()
        x1 = self._drag_current_src.x()
        y1 = self._drag_current_src.y()

        x = int(round(min(x0, x1)))
        y = int(round(min(y0, y1)))
        w = int(round(abs(x1 - x0)))
        h = int(round(abs(y1 - y0)))
        roi = clamp_roi((x, y, w, h), self._source_size[0], self._source_size[1])
        if roi is None:
            return None
        if roi[2] < 8 or roi[3] < 8:
            return None
        return roi

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            p = self._widget_to_source(event.position())
            if p is not None:
                self._dragging = True
                self._drag_start_src = p
                self._drag_current_src = p
                self.update()
                return
        elif event.button() == QtCore.Qt.MouseButton.RightButton:
            self.clear_roi()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._dragging:
            p = self._widget_to_source(event.position())
            if p is not None:
                self._drag_current_src = p
                self.update()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._dragging and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._dragging = False
            p = self._widget_to_source(event.position())
            if p is not None:
                self._drag_current_src = p
            roi = self._normalized_roi_from_drag()
            if roi is not None:
                self._roi = roi
                self.roiChanged.emit(roi)
            self._drag_start_src = None
            self._drag_current_src = None
            self.update()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor(20, 24, 28))

        if self._pixmap is not None:
            scaled = self._pixmap.scaled(self.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
            x = (self.width() - scaled.width()) / 2.0
            y = (self.height() - scaled.height()) / 2.0
            painter.drawPixmap(int(x), int(y), scaled)
            self._image_rect = QtCore.QRectF(x, y, scaled.width(), scaled.height())
        else:
            self._image_rect = QtCore.QRectF()
            painter.setPen(QtGui.QPen(QtGui.QColor("#8b949e"), 1))
            painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "Aguardando vídeo...")

        roi_to_draw = self._roi
        if self._dragging:
            roi_to_draw = self._normalized_roi_from_drag()

        if roi_to_draw is not None and self._source_size[0] > 0:
            rect = self._source_to_widget_rect(roi_to_draw)
            if not rect.isNull():
                pen = QtGui.QPen(QtGui.QColor("#00d084"), 2)
                painter.setPen(pen)
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(rect, 4, 4)

                label = f"ROI: x={roi_to_draw[0]} y={roi_to_draw[1]} w={roi_to_draw[2]} h={roi_to_draw[3]}"
                fm = painter.fontMetrics()
                pad = 6
                text_w = fm.horizontalAdvance(label) + 2 * pad
                text_h = fm.height() + 2 * pad
                text_rect = QtCore.QRectF(rect.left(), max(0.0, rect.top() - text_h - 4), text_w, text_h)
                painter.fillRect(text_rect, QtGui.QColor(0, 208, 132, 160))
                painter.setPen(QtGui.QPen(QtGui.QColor("black"), 1))
                painter.drawText(text_rect.adjusted(pad, 0, -pad, 0), QtCore.Qt.AlignmentFlag.AlignVCenter, label)

        painter.end()


# ==============================
# Janela principal
# ==============================

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("rPPG ao vivo - POS + FFT em tempo real")
        self.resize(1500, 920)
        self._apply_dark_theme()

        self.worker_thread: Optional[QtCore.QThread] = None
        self.worker: Optional[CaptureWorker] = None

        self.timestamps = deque(maxlen=int(DEFAULT_PROCESS_BUFFER_S * 120))
        self.rgb_values = deque(maxlen=int(DEFAULT_PROCESS_BUFFER_S * 120))
        self.latest_frame_time = None
        self.last_fft_update_time = 0.0
        self.last_bpm = math.nan
        self.last_snr = math.nan

        self._build_ui()
        self._connect_ui()

        self.plot_timer = QtCore.QTimer(self)
        self.plot_timer.setInterval(120)
        self.plot_timer.timeout.connect(self.update_processing)

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background-color: #0f1720;
                color: #e6edf3;
                font-size: 13px;
            }
            QFrame#Card, QGroupBox {
                background-color: #161b22;
                border: 1px solid #2b3442;
                border-radius: 10px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #8bd5ff;
            }
            QPushButton {
                background-color: #238636;
                border: none;
                border-radius: 8px;
                padding: 9px 14px;
                color: white;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #2ea043;
            }
            QPushButton:disabled {
                background-color: #30363d;
                color: #8b949e;
            }
            QPushButton#StopButton {
                background-color: #da3633;
            }
            QPushButton#StopButton:hover {
                background-color: #f85149;
            }
            QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {
                background-color: #0d1117;
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 6px 8px;
            }
            QLabel[role="metric"] {
                font-size: 20px;
                font-weight: 700;
            }
            QLabel[role="subtle"] {
                color: #8b949e;
            }
            """
        )

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)

        left = QtWidgets.QVBoxLayout()
        right = QtWidgets.QVBoxLayout()
        root.addLayout(left, stretch=4)
        root.addLayout(right, stretch=1)

        self.video_widget = VideoWidget()
        self.video_widget.setObjectName("Card")
        left.addWidget(self.video_widget, stretch=6)

        plots_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        plots_splitter.addWidget(self._build_signal_plot_card())
        plots_splitter.addWidget(self._build_fft_plot_card())
        plots_splitter.setSizes([320, 320])
        left.addWidget(plots_splitter, stretch=4)

        right.addWidget(self._build_controls_card())
        right.addWidget(self._build_metrics_card())
        right.addStretch(1)

    def _build_signal_plot_card(self) -> QtWidgets.QWidget:
        card = QtWidgets.QGroupBox("Sinal rPPG (POS)")
        layout = QtWidgets.QVBoxLayout(card)

        self.signal_plot = pg.PlotWidget()
        self.signal_plot.setBackground("#161b22")
        self.signal_plot.showGrid(x=True, y=True, alpha=0.25)
        self.signal_plot.setLabel("left", "Amplitude")
        self.signal_plot.setLabel("bottom", "Tempo", units="s")
        self.signal_plot.addLegend(offset=(10, 10))
        self.signal_curve = self.signal_plot.plot(name="BVP filtrado", pen=pg.mkPen("#4cc9f0", width=2))
        layout.addWidget(self.signal_plot)
        return card

    def _build_fft_plot_card(self) -> QtWidgets.QWidget:
        card = QtWidgets.QGroupBox("FFT / Frequência cardíaca")
        layout = QtWidgets.QVBoxLayout(card)

        self.fft_plot = pg.PlotWidget()
        self.fft_plot.setBackground("#161b22")
        self.fft_plot.showGrid(x=True, y=True, alpha=0.25)
        self.fft_plot.setLabel("left", "|FFT|")
        self.fft_plot.setLabel("bottom", "Frequência", units="BPM")
        self.fft_plot.setXRange(DEFAULT_BPM_MIN, DEFAULT_BPM_MAX)
        self.fft_curve = self.fft_plot.plot(pen=pg.mkPen("#f59e0b", width=2))
        self.fft_peak_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#ff4d6d", width=2))
        self.fft_plot.addItem(self.fft_peak_line)
        layout.addWidget(self.fft_plot)
        return card

    def _build_controls_card(self) -> QtWidgets.QWidget:
        card = QtWidgets.QGroupBox("Controles")
        layout = QtWidgets.QFormLayout(card)
        layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        layout.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(10)

        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems(["Baumer (neoAPI)", "Webcam (OpenCV)"])
        self.source_combo.setCurrentText(DEFAULT_SOURCE)

        self.camera_index_spin = QtWidgets.QSpinBox()
        self.camera_index_spin.setRange(0, 20)
        self.camera_index_spin.setValue(DEFAULT_CAMERA_INDEX)

        self.target_fps_spin = QtWidgets.QDoubleSpinBox()
        self.target_fps_spin.setRange(5.0, 240.0)
        self.target_fps_spin.setDecimals(1)
        self.target_fps_spin.setSingleStep(1.0)
        self.target_fps_spin.setValue(DEFAULT_TARGET_FPS)

        self.signal_window_spin = QtWidgets.QDoubleSpinBox()
        self.signal_window_spin.setRange(5.0, 60.0)
        self.signal_window_spin.setDecimals(1)
        self.signal_window_spin.setValue(DEFAULT_SIGNAL_WINDOW_S)

        self.fft_window_spin = QtWidgets.QDoubleSpinBox()
        self.fft_window_spin.setRange(5.0, 30.0)
        self.fft_window_spin.setDecimals(1)
        self.fft_window_spin.setValue(DEFAULT_FFT_WINDOW_S)

        self.fft_update_spin = QtWidgets.QDoubleSpinBox()
        self.fft_update_spin.setRange(1.0, 30.0)
        self.fft_update_spin.setDecimals(1)
        self.fft_update_spin.setValue(DEFAULT_FFT_UPDATE_EVERY_S)

        self.low_cut_spin = QtWidgets.QDoubleSpinBox()
        self.low_cut_spin.setRange(0.1, 3.0)
        self.low_cut_spin.setDecimals(2)
        self.low_cut_spin.setValue(DEFAULT_BANDPASS_LOW_HZ)

        self.high_cut_spin = QtWidgets.QDoubleSpinBox()
        self.high_cut_spin.setRange(0.5, 8.0)
        self.high_cut_spin.setDecimals(2)
        self.high_cut_spin.setValue(DEFAULT_BANDPASS_HIGH_HZ)

        self.bpm_min_spin = QtWidgets.QDoubleSpinBox()
        self.bpm_min_spin.setRange(20.0, 120.0)
        self.bpm_min_spin.setDecimals(0)
        self.bpm_min_spin.setValue(DEFAULT_BPM_MIN)

        self.bpm_max_spin = QtWidgets.QDoubleSpinBox()
        self.bpm_max_spin.setRange(80.0, 240.0)
        self.bpm_max_spin.setDecimals(0)
        self.bpm_max_spin.setValue(DEFAULT_BPM_MAX)

        self.start_button = QtWidgets.QPushButton("Iniciar")
        self.stop_button = QtWidgets.QPushButton("Parar")
        self.stop_button.setObjectName("StopButton")
        self.stop_button.setEnabled(False)

        self.status_label = QtWidgets.QLabel("Pronto.")
        self.status_label.setProperty("role", "subtle")
        self.status_label.setWordWrap(True)

        layout.addRow("Fonte", self.source_combo)
        layout.addRow("Índice webcam", self.camera_index_spin)
        layout.addRow("FPS alvo", self.target_fps_spin)
        layout.addRow("Janela do sinal (s)", self.signal_window_spin)
        layout.addRow("Janela FFT (s)", self.fft_window_spin)
        layout.addRow("Atualizar FFT a cada (s)", self.fft_update_spin)
        layout.addRow("Passa-banda min (Hz)", self.low_cut_spin)
        layout.addRow("Passa-banda max (Hz)", self.high_cut_spin)
        layout.addRow("BPM mínimo", self.bpm_min_spin)
        layout.addRow("BPM máximo", self.bpm_max_spin)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.start_button)
        btn_row.addWidget(self.stop_button)
        layout.addRow(btn_row)
        layout.addRow("Status", self.status_label)

        help_label = QtWidgets.QLabel(
            "Arraste com o botão esquerdo para definir a ROI. Clique direito no vídeo para limpar a ROI."
        )
        help_label.setWordWrap(True)
        help_label.setProperty("role", "subtle")
        layout.addRow(help_label)
        return card

    def _build_metrics_card(self) -> QtWidgets.QWidget:
        card = QtWidgets.QGroupBox("Métricas")
        layout = QtWidgets.QVBoxLayout(card)

        self.bpm_value = QtWidgets.QLabel("--")
        self.bpm_value.setProperty("role", "metric")
        self.snr_value = QtWidgets.QLabel("--")
        self.snr_value.setProperty("role", "metric")
        self.fps_value = QtWidgets.QLabel("--")
        self.fps_value.setProperty("role", "metric")
        self.samples_value = QtWidgets.QLabel("0")
        self.samples_value.setProperty("role", "metric")

        for title, widget in [
            ("Frequência cardíaca (BPM)", self.bpm_value),
            ("SNR espectral (dB)", self.snr_value),
            ("FPS estimado", self.fps_value),
            ("Amostras no buffer", self.samples_value),
        ]:
            box = QtWidgets.QFrame()
            box.setObjectName("Card")
            v = QtWidgets.QVBoxLayout(box)
            lbl = QtWidgets.QLabel(title)
            lbl.setProperty("role", "subtle")
            v.addWidget(lbl)
            v.addWidget(widget)
            layout.addWidget(box)

        return card

    def _connect_ui(self) -> None:
        self.start_button.clicked.connect(self.start_capture)
        self.stop_button.clicked.connect(self.stop_capture)
        self.video_widget.roiChanged.connect(self.on_roi_changed)
        self.video_widget.roiCleared.connect(self.on_roi_cleared)

    @QtCore.Slot(tuple)
    def on_roi_changed(self, roi: Tuple[int, int, int, int]) -> None:
        if self.worker is not None:
            self.worker.set_roi(roi)
        self.status_label.setText(f"ROI ativa: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")

    @QtCore.Slot()
    def on_roi_cleared(self) -> None:
        if self.worker is not None:
            self.worker.clear_roi()
        self.status_label.setText("ROI removida. Desenhe uma nova ROI para calcular o rPPG.")

    def _reset_buffers(self) -> None:
        self.timestamps.clear()
        self.rgb_values.clear()
        self.last_fft_update_time = 0.0
        self.last_bpm = math.nan
        self.last_snr = math.nan
        self.signal_curve.setData([], [])
        self.fft_curve.setData([], [])
        self.fft_peak_line.setPos(0)
        self.bpm_value.setText("--")
        self.snr_value.setText("--")
        self.fps_value.setText("--")
        self.samples_value.setText("0")

    @QtCore.Slot()
    def start_capture(self) -> None:
        if self.worker_thread is not None:
            return

        self._reset_buffers()

        source = self.source_combo.currentText()
        camera_index = int(self.camera_index_spin.value())
        target_fps = float(self.target_fps_spin.value())

        self.worker_thread = QtCore.QThread(self)
        self.worker = CaptureWorker(
            source_name=source,
            camera_index=camera_index,
            target_fps=target_fps,
            max_display_height=DEFAULT_MAX_DISPLAY_HEIGHT,
        )
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.frameReady.connect(self.on_frame_ready)
        self.worker.statusMessage.connect(self.on_status_message)
        self.worker.errorMessage.connect(self.on_worker_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

        roi = self.video_widget.roi()
        if roi is not None:
            self.worker.set_roi(roi)

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.plot_timer.start()
        self.status_label.setText("Inicializando captura...")

    @QtCore.Slot()
    def stop_capture(self) -> None:
        if self.worker is not None:
            self.worker.stop()
        self.plot_timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Parando captura...")

    @QtCore.Slot(str)
    def on_status_message(self, msg: str) -> None:
        self.status_label.setText(msg)

    @QtCore.Slot(str)
    def on_worker_error(self, msg: str) -> None:
        self.status_label.setText(f"Erro: {msg}")
        QtWidgets.QMessageBox.critical(self, "Erro na captura", msg)
        self.stop_capture()

    @QtCore.Slot()
    def on_worker_finished(self) -> None:
        self.plot_timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.worker = None
        self.worker_thread = None
        if not self.status_label.text().startswith("Erro"):
            self.status_label.setText("Captura encerrada.")

    @QtCore.Slot(object, float, tuple, object)
    def on_frame_ready(self, display_bgr8: np.ndarray, timestamp_s: float, source_size: tuple, mean_rgb: object) -> None:
        self.video_widget.set_frame(display_bgr8, source_size)
        self.latest_frame_time = timestamp_s

        if mean_rgb is not None:
            self.timestamps.append(float(timestamp_s))
            self.rgb_values.append(np.asarray(mean_rgb, dtype=np.float64))

            max_age = max(float(self.signal_window_spin.value()), float(self.fft_window_spin.value()), DEFAULT_PROCESS_BUFFER_S)
            cutoff = float(timestamp_s) - max_age
            while self.timestamps and self.timestamps[0] < cutoff:
                self.timestamps.popleft()
                self.rgb_values.popleft()

            self.samples_value.setText(str(len(self.timestamps)))

    @QtCore.Slot()
    def update_processing(self) -> None:
        if len(self.timestamps) < 20 or len(self.rgb_values) < 20:
            return

        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        rgb = np.asarray(self.rgb_values, dtype=np.float64)
        if timestamps.size != rgb.shape[0]:
            return

        target_fps = float(self.target_fps_spin.value())
        fps_est = estimate_fps(timestamps, target_fps)
        self.fps_value.setText(f"{fps_est:.2f}")

        try:
            rgb_u, t_u = resample_rgb(rgb, timestamps, fps_est)
            pos = pos_algorithm(rgb_u, fps=fps_est, window_sec=DEFAULT_POS_WINDOW_S)
            filtered = bandpass_filter(
                pos,
                fps=fps_est,
                low_hz=float(self.low_cut_spin.value()),
                high_hz=float(self.high_cut_spin.value()),
            )
        except Exception as exc:
            self.status_label.setText(f"Falha no processamento: {exc}")
            return

        t_rel = t_u - t_u[-1]

        signal_window_s = float(self.signal_window_spin.value())
        signal_mask = t_rel >= -signal_window_s
        if np.any(signal_mask):
            self.signal_curve.setData(t_rel[signal_mask], filtered[signal_mask])
            self.signal_plot.setXRange(-signal_window_s, 0.0, padding=0.01)

        now = time.perf_counter()
        fft_update_interval = float(self.fft_update_spin.value())
        if now - self.last_fft_update_time < fft_update_interval:
            return

        fft_window_s = float(self.fft_window_spin.value())
        fft_mask = t_rel >= -fft_window_s
        if np.count_nonzero(fft_mask) < 32:
            return

        fft_signal = filtered[fft_mask]
        freqs_bpm, spec, bpm_peak, snr_db = compute_fft_bpm(
            fft_signal,
            fps=fps_est,
            bpm_min=float(self.bpm_min_spin.value()),
            bpm_max=float(self.bpm_max_spin.value()),
        )
        if freqs_bpm.size == 0:
            return

        bpm_min = float(self.bpm_min_spin.value())
        bpm_max = float(self.bpm_max_spin.value())
        band = (freqs_bpm >= bpm_min) & (freqs_bpm <= bpm_max)
        self.fft_curve.setData(freqs_bpm[band], spec[band])
        self.fft_plot.setXRange(bpm_min, bpm_max, padding=0.01)

        if np.isfinite(bpm_peak):
            self.fft_peak_line.setPos(bpm_peak)
            self.last_bpm = bpm_peak
            self.last_snr = snr_db
            self.bpm_value.setText(f"{bpm_peak:.1f}")
            self.snr_value.setText("∞" if not np.isfinite(snr_db) else f"{snr_db:.1f}")
            self.status_label.setText(
                f"Estimativa atual: {bpm_peak:.1f} BPM | SNR {('∞' if not np.isfinite(snr_db) else f'{snr_db:.1f} dB')}"
            )

        self.last_fft_update_time = now

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.stop_capture()
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait(2000)
        event.accept()


def main() -> int:
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
