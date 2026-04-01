from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import h5py
import neoapi
import numpy as np


@dataclass(frozen=True)
class RecorderConfig:
    alvo_fps: float = 45.0
    tempo_captura_seg: float = 30.0
    arquivo_saida: str = "captura_baumer_12bit_lock.h5"
    altura_max_display: int = 720
    pixel_format_camera: object = neoapi.PixelFormat_BayerRG12p
    pixel_format_camera_nome: str = "BayerRG12p"
    sensor_bit_depth: int = 12
    margem_exposicao_us: float = 2000.0
    passo_gain: float = 0.5
    flush_frames_iniciais: int = 5
    hdf5_compression: Optional[str] = None
    dataset_video: str = "video_bruto"
    dataset_timestamps: str = "timestamps_camera"
    dataset_host_timestamps: str = "timestamps_host_ns"
    dataset_buffer_ids: str = "buffer_ids"
    dataset_timestamps_rel: str = "timestamps_s"
    storage_candidates: Tuple[Tuple[str, str, int], ...] = (
        ("BGR16", "bgr_uint16", 16),
        ("BGR12", "bgr_uint16", 16),
    )


CONFIG = RecorderConfig()
WINDOW_PREVIEW = "Baumer - Preview"
WINDOW_ROI = "Baumer - Selecione a ROI e pressione ENTER"


def get_feature_value(camera: neoapi.Cam, feature_name: str, default=None):
    try:
        if camera.HasFeature(feature_name):
            return getattr(camera.f, feature_name).value
    except Exception:
        pass
    return default


def get_increment(feature, default: int = 1) -> int:
    try:
        inc = int(feature.GetInc())
        return max(1, inc)
    except Exception:
        return default


def align_down(value: int, minimum: int, increment: int) -> int:
    value = max(value, minimum)
    return minimum + ((value - minimum) // increment) * increment


def compute_display_geometry(width: int, height: int, max_height: int) -> Tuple[int, int, float]:
    scale = min(1.0, max_height / float(height))
    display_w = max(1, int(round(width * scale)))
    display_h = max(1, int(round(height * scale)))
    return display_w, display_h, scale


def resize_for_display(frame_bgr8: np.ndarray, max_height: int) -> Tuple[np.ndarray, float]:
    h, w = frame_bgr8.shape[:2]
    display_w, display_h, scale = compute_display_geometry(w, h, max_height)
    resized = cv2.resize(frame_bgr8, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale


def reset_full_sensor(camera: neoapi.Cam) -> None:
    try:
        if camera.HasFeature("OffsetX"):
            camera.f.OffsetX.value = camera.f.OffsetX.GetMin()
        if camera.HasFeature("OffsetY"):
            camera.f.OffsetY.value = camera.f.OffsetY.GetMin()
        if camera.HasFeature("Width"):
            camera.f.Width.value = camera.f.Width.GetMax()
        if camera.HasFeature("Height"):
            camera.f.Height.value = camera.f.Height.GetMax()
    except Exception:
        pass


def convert_preview_to_bgr8(img) -> np.ndarray:
    converted = img.Convert("BGR8")
    if converted.IsEmpty():
        raise RuntimeError("Falha ao converter frame para preview BGR8.")
    arr = converted.GetNPArray()
    if arr is None or arr.size == 0:
        raise RuntimeError("Frame de preview vazio.")
    return arr


def configure_camera_base(camera: neoapi.Cam, config: RecorderConfig) -> Dict[str, float]:
    try:
        camera.f.TriggerMode.value = neoapi.TriggerMode_Off
    except Exception:
        pass

    try:
        camera.f.ExposureAuto.value = neoapi.ExposureAuto_Off
    except Exception:
        pass
    try:
        camera.f.GainAuto.value = neoapi.GainAuto_Off
    except Exception:
        pass

    camera.f.PixelFormat.value = config.pixel_format_camera

    try:
        camera.f.DeviceLinkThroughputLimit.value = camera.f.DeviceLinkThroughputLimit.GetMax()
    except Exception:
        pass

    frame_period_us = 1_000_000.0 / config.alvo_fps
    exposure_us = max(1.0, frame_period_us - config.margem_exposicao_us)
    try:
        exposure_us = min(exposure_us, float(camera.f.ExposureTime.GetMax()))
        exposure_us = max(exposure_us, float(camera.f.ExposureTime.GetMin()))
    except Exception:
        pass

    camera.f.ExposureTime.value = exposure_us

    try:
        camera.f.AcquisitionFrameRateEnable.value = True
        camera.f.AcquisitionFrameRate.value = config.alvo_fps
        fps_configurado = float(camera.f.AcquisitionFrameRate.value)
    except Exception:
        fps_configurado = config.alvo_fps

    return {
        "fps_configurado": float(fps_configurado),
        "exposure_us": float(camera.f.ExposureTime.value),
    }


def preview_and_select_roi(camera: neoapi.Cam, config: RecorderConfig) -> Tuple[int, int, int, int, float, str, int, int]:
    print("\n[LIVE PREVIEW] Ajuste a imagem.")
    print(" -> W/S: ajusta ganho | ENTER: congelar imagem e selecionar ROI")

    cv2.namedWindow(WINDOW_PREVIEW, cv2.WINDOW_AUTOSIZE)

    last_full_frame = None
    last_scale = 1.0
    sensor_w = 0
    sensor_h = 0

    while True:
        img = camera.GetImage()
        if not img.IsEmpty():
            full_frame = convert_preview_to_bgr8(img)
            sensor_h, sensor_w = full_frame.shape[:2]
            frame_display, last_scale = resize_for_display(full_frame, config.altura_max_display)
            gain_value = float(get_feature_value(camera, "Gain", 0.0) or 0.0)
            overlay = f"Gain: {gain_value:.2f} | {sensor_w}x{sensor_h} | sensor={config.pixel_format_camera_nome}"
            cv2.putText(frame_display, overlay, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW_PREVIEW, frame_display)
            last_full_frame = full_frame.copy()

        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            break
        if key in (ord("w"), ord("W")):
            try:
                camera.f.Gain.value = min(camera.f.Gain.GetMax(), camera.f.Gain.value + config.passo_gain)
            except Exception:
                pass
        elif key in (ord("s"), ord("S")):
            try:
                camera.f.Gain.value = max(camera.f.Gain.GetMin(), camera.f.Gain.value - config.passo_gain)
            except Exception:
                pass

    cv2.destroyWindow(WINDOW_PREVIEW)

    if last_full_frame is None:
        raise RuntimeError("Nenhum frame válido foi obtido durante o preview.")

    frozen_display, _ = resize_for_display(last_full_frame, config.altura_max_display)
    roi_small = cv2.selectROI(WINDOW_ROI, frozen_display, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(WINDOW_ROI)

    if roi_small[2] <= 0 or roi_small[3] <= 0:
        return 0, 0, sensor_w, sensor_h, float(get_feature_value(camera, "Gain", 0.0) or 0.0), "full", sensor_w, sensor_h

    x = int(round(roi_small[0] / last_scale))
    y = int(round(roi_small[1] / last_scale))
    w = int(round(roi_small[2] / last_scale))
    h = int(round(roi_small[3] / last_scale))

    x = max(0, min(x, sensor_w - 1))
    y = max(0, min(y, sensor_h - 1))
    w = max(1, min(w, sensor_w - x))
    h = max(1, min(h, sensor_h - y))

    return x, y, w, h, float(get_feature_value(camera, "Gain", 0.0) or 0.0), "user", sensor_w, sensor_h


def apply_roi_hardware(camera: neoapi.Cam, x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
    if not all(camera.HasFeature(name) for name in ("OffsetX", "OffsetY", "Width", "Height")):
        return x, y, w, h

    fx = camera.f.OffsetX
    fy = camera.f.OffsetY
    fw = camera.f.Width
    fh = camera.f.Height

    min_x = int(fx.GetMin())
    min_y = int(fy.GetMin())
    min_w = int(fw.GetMin())
    min_h = int(fh.GetMin())
    max_w = int(fw.GetMax())
    max_h = int(fh.GetMax())

    inc_x = get_increment(fx)
    inc_y = get_increment(fy)
    inc_w = get_increment(fw)
    inc_h = get_increment(fh)

    x_aligned = align_down(int(x), min_x, inc_x)
    y_aligned = align_down(int(y), min_y, inc_y)
    w_aligned = align_down(int(w), min_w, inc_w)
    h_aligned = align_down(int(h), min_h, inc_h)

    w_aligned = min(w_aligned, max_w - x_aligned)
    h_aligned = min(h_aligned, max_h - y_aligned)
    w_aligned = max(min_w, align_down(w_aligned, min_w, inc_w))
    h_aligned = max(min_h, align_down(h_aligned, min_h, inc_h))

    fx.value = min_x
    fy.value = min_y
    fw.value = w_aligned
    fh.value = h_aligned
    fx.value = x_aligned
    fy.value = y_aligned

    return int(fx.value), int(fy.value), int(fw.value), int(fh.value)


def flush_initial_frames(camera: neoapi.Cam, n_frames: int) -> None:
    for _ in range(max(0, n_frames)):
        img = camera.GetImage()
        del img


def probe_storage_conversion(camera: neoapi.Cam, config: RecorderConfig) -> Tuple[str, str, int, Tuple[int, ...]]:
    img = camera.GetImage()
    if img.IsEmpty():
        raise RuntimeError("Não foi possível obter frame para sondagem do formato de armazenamento.")

    erros = []
    for conversion_name, storage_layout, container_bits in config.storage_candidates:
        try:
            converted = img.Convert(conversion_name)
            if converted.IsEmpty():
                erros.append(f"{conversion_name}: conversão vazia")
                continue
            frame = converted.GetNPArray()
            if frame is None or frame.size == 0:
                erros.append(f"{conversion_name}: array vazio")
                continue
            if frame.dtype != np.uint16:
                frame = frame.astype(np.uint16)
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if not (frame.ndim == 3 and frame.shape[2] == 3):
                erros.append(f"{conversion_name}: shape inesperado {frame.shape}")
                continue
            return conversion_name, storage_layout, container_bits, frame.shape
        except Exception as exc:
            erros.append(f"{conversion_name}: {exc}")

    detalhe = " | ".join(erros) if erros else "nenhum detalhe"
    raise RuntimeError(
        "Nenhuma conversão RGB de 12 bits/uint16 foi aceita para armazenamento. "
        f"Tentativas: {detalhe}"
    )


def convert_storage_frame(img, conversion_name: str) -> np.ndarray:
    converted = img.Convert(conversion_name)
    if converted.IsEmpty():
        raise RuntimeError(f"Falha ao converter frame para {conversion_name}.")
    frame = converted.GetNPArray()
    if frame is None or frame.size == 0:
        raise RuntimeError("Frame convertido vazio.")
    if frame.dtype != np.uint16:
        frame = frame.astype(np.uint16)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if not (frame.ndim == 3 and frame.shape[2] == 3):
        raise RuntimeError(f"Formato inesperado na conversão: shape={frame.shape}")
    return frame


def get_camera_identity(camera: neoapi.Cam) -> Dict[str, object]:
    return {
        "camera_model": str(get_feature_value(camera, "DeviceModelName", "Baumer")),
        "camera_vendor": str(get_feature_value(camera, "DeviceVendorName", "Baumer")),
        "camera_serial": str(get_feature_value(camera, "DeviceSerialNumber", "unknown")),
        "device_version": str(get_feature_value(camera, "DeviceVersion", "unknown")),
        "transport_layer": str(get_feature_value(camera, "DeviceTLType", "unknown")),
    }


def create_hdf5_file(
    output_path: Path,
    config: RecorderConfig,
    n_frames: int,
    frame_shape: Tuple[int, ...],
    metadata: Dict[str, object],
):
    file_handle = h5py.File(output_path, "w")
    file_handle.attrs["file_format"] = "baumer_capture_hdf5"
    file_handle.attrs["file_format_version"] = "3.0"
    file_handle.attrs["software_name"] = "baumer_record_profissional_final"
    file_handle.attrs["software_version"] = "3.0"
    file_handle.attrs["created_unix_time"] = time.time()

    for key, value in metadata.items():
        file_handle.attrs[key] = value

    chunks = (1, *frame_shape)
    video_ds = file_handle.create_dataset(
        config.dataset_video,
        shape=(n_frames, *frame_shape),
        dtype=np.uint16,
        chunks=chunks,
        compression=config.hdf5_compression,
    )
    ts_cam_ds = file_handle.create_dataset(config.dataset_timestamps, shape=(n_frames,), dtype=np.int64)
    ts_host_ds = file_handle.create_dataset(config.dataset_host_timestamps, shape=(n_frames,), dtype=np.int64)
    buffer_ds = file_handle.create_dataset(config.dataset_buffer_ids, shape=(n_frames,), dtype=np.int64)
    ts_rel_ds = file_handle.create_dataset(config.dataset_timestamps_rel, shape=(n_frames,), dtype=np.float64)
    ts_rel_ds.attrs["description"] = "Tempo relativo de chegada do frame desde o início da captura"
    return file_handle, video_ds, ts_cam_ds, ts_host_ds, buffer_ds, ts_rel_ds


def print_capture_summary(metadata: Dict[str, object], target_frames: int, output_path: Path) -> None:
    print("=" * 72)
    print("PRONTO PARA CAPTURAR")
    print("=" * 72)
    print(f"Camera:                 {metadata.get('camera_model', 'Baumer')}  SN={metadata.get('camera_serial', 'unknown')}")
    print(f"ROI aplicada:           x={metadata['roi_offset_x']}, y={metadata['roi_offset_y']}, w={metadata['roi_width']}, h={metadata['roi_height']}")
    print(f"Formato na camera:      {metadata['source_pixel_format']}")
    print(f"Conversao usada:        {metadata['storage_conversion_name']}")
    print(f"Formato salvo no HDF5:  {metadata['storage_layout']} (uint{metadata['storage_container_bits']})")
    print(f"Bits válidos por canal: {metadata['valid_bits_per_channel']}")
    print(f"FPS pedido:             {metadata['fps_requested']:.3f}")
    print(f"FPS configurado:        {metadata['fps_configured']:.3f}")
    print(f"Exposição:              {metadata['exposure_time_us']:.1f} us")
    print(f"Duração alvo:           {metadata['capture_duration_s']:.3f} s")
    print(f"Frames alvo:            {target_frames}")
    print(f"Arquivo:                {output_path}")


def summarize_timestamps(ts_rel: np.ndarray) -> Tuple[float, float, float]:
    if ts_rel.size < 2:
        return 0.0, 0.0, 0.0
    dt = np.diff(ts_rel)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return 0.0, 0.0, 0.0
    fps_mean = float((ts_rel.size - 1) / (ts_rel[-1] - ts_rel[0])) if ts_rel[-1] > ts_rel[0] else 0.0
    fps_median = float(1.0 / np.median(dt))
    jitter_std_ms = float(1000.0 * np.std(dt))
    return fps_mean, fps_median, jitter_std_ms


def main() -> int:
    camera = None
    output_path = Path(CONFIG.arquivo_saida)

    try:
        camera = neoapi.Cam()
        camera.Connect()
        identity = get_camera_identity(camera)
        print(f"Câmera conectada: {identity['camera_model']} (SN={identity['camera_serial']})")

        reset_full_sensor(camera)
        base_cfg = configure_camera_base(camera, CONFIG)

        roi_x, roi_y, roi_w, roi_h, gain_final, roi_mode, full_w, full_h = preview_and_select_roi(camera, CONFIG)
        roi_x, roi_y, roi_w, roi_h = apply_roi_hardware(camera, roi_x, roi_y, roi_w, roi_h)

        flush_initial_frames(camera, CONFIG.flush_frames_iniciais)
        storage_conversion_name, storage_layout, container_bits, frame_shape = probe_storage_conversion(camera, CONFIG)
        print(f"Formato de gravação travado para esta captura: {storage_conversion_name} ({storage_layout}, uint{container_bits})")

        target_frames = int(round(base_cfg["fps_configurado"] * CONFIG.tempo_captura_seg))
        if target_frames <= 0:
            raise RuntimeError("Configuração inválida: o número de frames alvo ficou <= 0.")

        metadata: Dict[str, object] = {
            **identity,
            "fps": float(base_cfg["fps_configurado"]),
            "exposure_us": float(base_cfg["exposure_us"]),
            "gain": float(gain_final),
            "sensor_pixel_format": CONFIG.pixel_format_camera_nome,
            "sensor_bit_depth": int(CONFIG.sensor_bit_depth),
            "saved_format": storage_conversion_name,
            "storage_container_bits": int(container_bits),
            "roi_x": int(roi_x),
            "roi_y": int(roi_y),
            "roi_w": int(roi_w),
            "roi_h": int(roi_h),
            "full_width": int(full_w),
            "full_height": int(full_h),
            "fps_requested": float(CONFIG.alvo_fps),
            "fps_configured": float(base_cfg["fps_configurado"]),
            "frame_interval_target_s": float(1.0 / base_cfg["fps_configurado"]) if base_cfg["fps_configurado"] > 0 else np.nan,
            "capture_duration_s": float(CONFIG.tempo_captura_seg),
            "exposure_time_us": float(base_cfg["exposure_us"]),
            "source_pixel_format": CONFIG.pixel_format_camera_nome,
            "storage_layout": storage_layout,
            "storage_conversion_name": storage_conversion_name,
            "valid_bits_per_channel": int(CONFIG.sensor_bit_depth),
            "roi_offset_x": int(roi_x),
            "roi_offset_y": int(roi_y),
            "roi_width": int(roi_w),
            "roi_height": int(roi_h),
            "roi_mode": roi_mode,
            "frame_shape": str(frame_shape),
            "display_height_default": int(CONFIG.altura_max_display),
            "dataset_video_name": CONFIG.dataset_video,
            "dataset_timestamps_name": CONFIG.dataset_timestamps,
            "dataset_host_timestamps_name": CONFIG.dataset_host_timestamps,
            "dataset_buffer_ids_name": CONFIG.dataset_buffer_ids,
            "conversion_candidates": ",".join(name for name, _, _ in CONFIG.storage_candidates),
        }

        print_capture_summary(metadata, target_frames, output_path)
        input("Pressione ENTER para iniciar a gravação... ")

        flush_initial_frames(camera, CONFIG.flush_frames_iniciais)
        h5f, video_ds, ts_cam_ds, ts_host_ds, buffer_ds, ts_rel_ds = create_hdf5_file(
            output_path, CONFIG, target_frames, frame_shape, metadata
        )

        try:
            t0 = time.perf_counter()
            observed_min = None
            observed_max = None
            empty_frames = 0
            last_buffer_id = None
            buffer_gap_loss = 0

            ts_rel_cache = np.zeros(target_frames, dtype=np.float64)

            for captured in range(target_frames):
                while True:
                    img = camera.GetImage()
                    if img.IsEmpty():
                        empty_frames += 1
                        continue
                    break

                frame = convert_storage_frame(img, storage_conversion_name)
                video_ds[captured] = frame

                frame_min = int(frame.min())
                frame_max = int(frame.max())
                observed_min = frame_min if observed_min is None else min(observed_min, frame_min)
                observed_max = frame_max if observed_max is None else max(observed_max, frame_max)

                try:
                    cam_ts = int(img.GetTimestamp())
                except Exception:
                    cam_ts = -1
                try:
                    buffer_id = int(img.GetBufferID())
                except Exception:
                    buffer_id = -1

                if last_buffer_id is not None and buffer_id >= 0 and buffer_id > last_buffer_id + 1:
                    buffer_gap_loss += (buffer_id - last_buffer_id - 1)
                if buffer_id >= 0:
                    last_buffer_id = buffer_id

                t_rel = time.perf_counter() - t0
                ts_rel_cache[captured] = t_rel
                ts_cam_ds[captured] = cam_ts
                ts_host_ds[captured] = time.perf_counter_ns()
                buffer_ds[captured] = buffer_id
                ts_rel_ds[captured] = t_rel

                if (captured + 1) % 50 == 0 or (captured + 1) == target_frames:
                    print(f" -> {captured + 1}/{target_frames} frames gravados")

            elapsed = float(ts_rel_cache[-1]) if target_frames > 0 else 0.0
            fps_effective_mean, fps_effective_median, jitter_std_ms = summarize_timestamps(ts_rel_cache)

            h5f.attrs["captured_frames"] = int(target_frames)
            h5f.attrs["empty_frames"] = int(empty_frames)
            h5f.attrs["buffer_gap_loss"] = int(buffer_gap_loss)
            h5f.attrs["capture_elapsed_wall_s"] = elapsed
            h5f.attrs["capture_fps_effective_mean"] = fps_effective_mean
            h5f.attrs["capture_fps_effective_median"] = fps_effective_median
            h5f.attrs["capture_jitter_std_ms"] = jitter_std_ms
            h5f.attrs["observed_min"] = int(observed_min if observed_min is not None else 0)
            h5f.attrs["observed_max"] = int(observed_max if observed_max is not None else 0)

        finally:
            h5f.flush()
            h5f.close()

        print("=" * 72)
        print("CAPTURA FINALIZADA")
        print("=" * 72)
        print(f"Frames gravados:         {target_frames}")
        print(f"FPS efetivo (médio):     {fps_effective_mean:.5f}")
        print(f"FPS efetivo (mediana):   {fps_effective_median:.5f}")
        print(f"Jitter std:              {jitter_std_ms:.3f} ms")
        print(f"Perda por gap buffer:    {buffer_gap_loss}")
        print(f"Range observado:         min={observed_min} | max={observed_max}")
        print(f"Conversão travada:       {storage_conversion_name}")
        print(f"Arquivo HDF5:            {output_path.resolve()}")
        print("Programa encerrado.")
        return 0

    except KeyboardInterrupt:
        print("\nCaptura interrompida pelo usuário.")
        return 130
    except Exception as exc:
        print(f"\nErro inesperado: {exc}")
        return 1
    finally:
        cv2.destroyAllWindows()
        if camera is not None:
            try:
                if camera.IsConnected():
                    camera.Disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
