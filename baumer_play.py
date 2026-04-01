from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import h5py
import numpy as np


ARQUIVO_LEITURA = "captura_baumer_12bit_lock.h5"
ALTURA_JANELA_DEFAULT = 720
WINDOW_NAME = "Baumer HDF5 Player"


@dataclass
class PlaybackState:
    fps: float
    paused: bool = False
    show_overlay: bool = True
    speed_multiplier: float = 1.0


def load_metadata(h5f: h5py.File) -> Dict[str, object]:
    attrs = h5f.attrs
    return {
        "camera_model": attrs.get("camera_model", "Baumer"),
        "camera_serial": attrs.get("camera_serial", "unknown"),
        "fps_nominal": float(attrs.get("fps_configured", attrs.get("fps", 30.0))),
        "fps_effective_mean": float(attrs.get("capture_fps_effective_mean", 0.0)),
        "fps_effective_median": float(attrs.get("capture_fps_effective_median", 0.0)),
        "jitter_std_ms": float(attrs.get("capture_jitter_std_ms", 0.0)),
        "exposure_time_us": float(attrs.get("exposure_time_us", attrs.get("exposure_us", 0.0))),
        "gain": float(attrs.get("gain", 0.0)),
        "storage_layout": attrs.get("storage_layout", "bgr_uint16"),
        "storage_conversion_name": attrs.get("storage_conversion_name", attrs.get("saved_format", "unknown")),
        "sensor_pixel_format": attrs.get("sensor_pixel_format", attrs.get("source_pixel_format", "BayerRG12p")),
        "sensor_bit_depth": int(attrs.get("sensor_bit_depth", attrs.get("valid_bits_per_channel", 12))),
        "storage_container_bits": int(attrs.get("storage_container_bits", 16)),
        "display_height_default": int(attrs.get("display_height_default", ALTURA_JANELA_DEFAULT)),
        "roi_offset_x": int(attrs.get("roi_offset_x", attrs.get("roi_x", 0))),
        "roi_offset_y": int(attrs.get("roi_offset_y", attrs.get("roi_y", 0))),
        "roi_width": int(attrs.get("roi_width", attrs.get("roi_w", 0))),
        "roi_height": int(attrs.get("roi_height", attrs.get("roi_h", 0))),
        "buffer_gap_loss": int(attrs.get("buffer_gap_loss", -1)),
        "observed_min": int(attrs.get("observed_min", 0)),
        "observed_max": int(attrs.get("observed_max", 0)),
        "conversion_candidates": attrs.get("conversion_candidates", ""),
    }


def dataset_video(h5f: h5py.File):
    if "video_bruto" in h5f:
        return h5f["video_bruto"]
    raise KeyError("O dataset 'video_bruto' não foi encontrado neste arquivo.")


def inferir_alinhamento(sample_values: np.ndarray, sensor_bits: int, container_bits: int) -> str:
    sample = sample_values.astype(np.uint16, copy=False)
    max_sensor = (1 << sensor_bits) - 1
    max_obs = int(sample.max())

    if max_obs <= max_sensor:
        return "right_aligned"

    shift = max(container_bits - sensor_bits, 0)
    if shift > 0:
        low_mask = (1 << shift) - 1
        frac_low_zero = float(np.mean((sample & low_mask) == 0))
        if frac_low_zero > 0.95:
            return "left_shifted"

    return "scaled_full"


def converter_para_12bit_logico(frame_u16: np.ndarray, sensor_bits: int, container_bits: int, alignment: str) -> np.ndarray:
    max_sensor = (1 << sensor_bits) - 1
    frame = frame_u16.astype(np.uint16, copy=False)

    if alignment == "right_aligned":
        frame12 = frame
    elif alignment == "left_shifted":
        shift = max(container_bits - sensor_bits, 0)
        frame12 = frame >> shift
    else:
        max_obs = int(frame.max())
        if max_obs <= 0:
            frame12 = np.zeros_like(frame, dtype=np.uint16)
        else:
            frame12 = np.round(frame.astype(np.float32) * max_sensor / max_obs).astype(np.uint16)

    return np.clip(frame12, 0, max_sensor).astype(np.uint16)


def frame12_para_display8(frame12: np.ndarray, sensor_bits: int) -> np.ndarray:
    max_sensor = float((1 << sensor_bits) - 1)
    frame8 = np.round(frame12.astype(np.float32) * 255.0 / max_sensor)
    return np.clip(frame8, 0, 255).astype(np.uint8)


def prepare_frame_for_display(frame: np.ndarray, metadata: Dict[str, object], alignment: str) -> np.ndarray:
    sensor_bits = int(metadata["sensor_bit_depth"])
    container_bits = int(metadata["storage_container_bits"])

    if frame.dtype == np.uint16:
        frame12 = converter_para_12bit_logico(frame, sensor_bits, container_bits, alignment)
        frame8 = frame12_para_display8(frame12, sensor_bits)
    elif frame.dtype == np.uint8:
        frame8 = frame
    else:
        frame8 = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    if frame8.ndim == 2:
        return cv2.cvtColor(frame8, cv2.COLOR_GRAY2BGR)
    return frame8


def resize_frame(frame_bgr8: np.ndarray, max_height: int) -> np.ndarray:
    h, w = frame_bgr8.shape[:2]
    scale = min(1.0, max_height / float(h))
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    return cv2.resize(frame_bgr8, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def choose_base_fps(metadata: Dict[str, object]) -> float:
    fps = float(metadata.get("fps_effective_median", 0.0))
    if fps <= 0:
        fps = float(metadata.get("fps_effective_mean", 0.0))
    if fps <= 0:
        fps = float(metadata.get("fps_nominal", 30.0))
    return fps


def draw_overlay(frame: np.ndarray, index: int, total: int, state: PlaybackState, metadata: Dict[str, object], alignment: str) -> np.ndarray:
    canvas = frame.copy()
    line1 = (
        f"Frame {index + 1}/{total} | FPS play {state.fps * state.speed_multiplier:.3f} | "
        f"Exp {metadata['exposure_time_us']:.0f} us | Gain {metadata['gain']:.2f}"
    )
    line2 = (
        f"{metadata['camera_model']} SN={metadata['camera_serial']} | "
        f"sensor={metadata['sensor_pixel_format']} ({metadata['sensor_bit_depth']}b) | "
        f"save={metadata['storage_conversion_name']}/uint{metadata['storage_container_bits']} | {alignment}"
    )
    line3 = (
        f"fps_nom={metadata['fps_nominal']:.5f} | fps_eff_med={metadata['fps_effective_median']:.5f} | "
        f"jitter={metadata['jitter_std_ms']:.3f} ms | gap={metadata['buffer_gap_loss']}"
    )

    cv2.putText(canvas, line1, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, line2, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, line3, (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    if state.paused:
        cv2.putText(canvas, "PAUSADO", (12, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def wait_time_ms(state: PlaybackState) -> int:
    effective_fps = max(0.01, state.fps * state.speed_multiplier)
    return max(1, int(round(1000.0 / effective_fps)))


def infer_stats_from_timestamps(h5f: h5py.File) -> Tuple[float, float, float]:
    if "timestamps_s" not in h5f:
        return 0.0, 0.0, 0.0
    ts = np.asarray(h5f["timestamps_s"][:], dtype=np.float64)
    if ts.size < 2:
        return 0.0, 0.0, 0.0
    dt = np.diff(ts)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return 0.0, 0.0, 0.0
    fps_mean = float((ts.size - 1) / (ts[-1] - ts[0])) if ts[-1] > ts[0] else 0.0
    fps_median = float(1.0 / np.median(dt))
    jitter_std_ms = float(1000.0 * np.std(dt))
    return fps_mean, fps_median, jitter_std_ms


def main() -> int:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(ARQUIVO_LEITURA)
    if not input_path.exists():
        print(f"Arquivo não encontrado: {input_path}")
        return 1

    try:
        with h5py.File(input_path, "r") as h5f:
            metadata = load_metadata(h5f)
            video = dataset_video(h5f)
            total_frames = video.shape[0]
            if total_frames <= 0:
                raise RuntimeError("O arquivo HDF5 não contém frames.")

            inferred_mean, inferred_median, inferred_jitter = infer_stats_from_timestamps(h5f)
            if metadata["fps_effective_mean"] <= 0:
                metadata["fps_effective_mean"] = inferred_mean
            if metadata["fps_effective_median"] <= 0:
                metadata["fps_effective_median"] = inferred_median
            if metadata["jitter_std_ms"] <= 0:
                metadata["jitter_std_ms"] = inferred_jitter

            preview_height = int(metadata.get("display_height_default", ALTURA_JANELA_DEFAULT))
            state = PlaybackState(fps=choose_base_fps(metadata))
            sample = video[: min(5, total_frames)]
            alignment = inferir_alinhamento(sample, int(metadata["sensor_bit_depth"]), int(metadata["storage_container_bits"]))

            print("=" * 72)
            print("ARQUIVO CARREGADO")
            print("=" * 72)
            print(f"Arquivo:                 {input_path}")
            print(f"Camera:                  {metadata['camera_model']}  SN={metadata['camera_serial']}")
            print(f"Frames:                  {total_frames}")
            print(f"FPS nominal:             {metadata['fps_nominal']:.5f}")
            print(f"FPS efetivo (médio):     {metadata['fps_effective_mean']:.5f}")
            print(f"FPS efetivo (mediana):   {metadata['fps_effective_median']:.5f}")
            print(f"Jitter std:              {metadata['jitter_std_ms']:.3f} ms")
            print(f"Sensor:                  {metadata['sensor_pixel_format']} ({metadata['sensor_bit_depth']} bits)")
            print(f"Layout salvo:            {metadata['storage_layout']} | {metadata['storage_conversion_name']}/uint{metadata['storage_container_bits']}")
            print(f"Range observado:         {metadata.get('observed_min', 0)} .. {metadata.get('observed_max', 0)}")
            print(f"Inferência de alinhamento: {alignment}")
            print("Controles: ESPAÇO pausa | +/- velocidade | I overlay | ESC/Q sair")

            frame_index = 0
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            while frame_index < total_frames:
                if not state.paused:
                    frame = video[frame_index]
                    display = prepare_frame_for_display(frame, metadata, alignment)
                    display = resize_frame(display, preview_height)
                    if state.show_overlay:
                        display = draw_overlay(display, frame_index, total_frames, state, metadata, alignment)
                    cv2.imshow(WINDOW_NAME, display)
                    frame_index += 1

                key = cv2.waitKey(0 if state.paused else wait_time_ms(state)) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key == ord(" "):
                    state.paused = not state.paused
                elif key in (ord("+"), ord("=")):
                    state.speed_multiplier = min(8.0, state.speed_multiplier * 1.25)
                elif key in (ord("-"), ord("_")):
                    state.speed_multiplier = max(0.125, state.speed_multiplier / 1.25)
                elif key in (ord("i"), ord("I")):
                    state.show_overlay = not state.show_overlay
        return 0
    except Exception as exc:
        print(f"Erro ao ler o arquivo: {exc}")
        return 1
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
