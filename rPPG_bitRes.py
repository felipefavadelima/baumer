import os
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, filtfilt
from scipy.fft import fft, fftfreq

# ====================================================================
# CONFIGURAÇÃO
# ====================================================================
ARQUIVO_ENTRADA = "captura_baumer_12bit_lock.h5"
FS_CORTAR_INICIO = 30
BPM_MIN = 40
BPM_MAX = 180
# ====================================================================


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


def aplicar_pos(rgb_signal: np.ndarray) -> np.ndarray:
    mean_rgb = np.mean(rgb_signal, axis=1, keepdims=True)
    mean_rgb[mean_rgb == 0] = 1.0
    cnorm = rgb_signal / mean_rgb - 1.0

    proj = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]], dtype=np.float64) @ cnorm
    std2 = np.std(proj[1, :])
    alpha = 0.0 if std2 == 0 else np.std(proj[0, :]) / std2
    return proj[0, :] + alpha * proj[1, :]


def filtro_passa_banda(sinal: np.ndarray, fps: float) -> np.ndarray:
    nyq = 0.5 * fps
    b, a = butter(4, [0.75 / nyq, 3.0 / nyq], btype="band")
    return filtfilt(b, a, sinal)


def estimar_bpm_e_snr(yf: np.ndarray, xf_hz: np.ndarray) -> tuple[float, float]:
    xf_bpm = xf_hz * 60.0
    banda = (xf_bpm >= BPM_MIN) & (xf_bpm <= BPM_MAX)
    if not np.any(banda):
        return np.nan, np.nan

    yf_band = yf[banda]
    xf_band = xf_bpm[banda]
    idx_peak = int(np.argmax(yf_band))
    bpm_peak = float(xf_band[idx_peak])

    janela = np.abs(xf_band - bpm_peak) <= 3.0
    pot_sinal = float(np.sum(yf_band[janela] ** 2))
    pot_ruido = float(np.sum(yf_band[~janela] ** 2))
    snr_db = np.inf if pot_ruido <= 0 else 10.0 * np.log10((pot_sinal + 1e-12) / (pot_ruido + 1e-12))
    return bpm_peak, snr_db


def analisar_timestamps(h5f: h5py.File, total_frames: int):
    if "timestamps_s" not in h5f:
        fps = float(h5f.attrs.get("capture_fps_effective_median", h5f.attrs.get("fps", 30.0)))
        return None, fps, fps, 0.0

    ts = np.asarray(h5f["timestamps_s"][:], dtype=np.float64)
    if ts.size != total_frames:
        ts = ts[:total_frames]
    mask = np.isfinite(ts)
    ts = ts[mask]
    if ts.size < 2:
        fps = float(h5f.attrs.get("capture_fps_effective_median", h5f.attrs.get("fps", 30.0)))
        return None, fps, fps, 0.0

    ts = ts - ts[0]
    uniq_mask = np.concatenate(([True], np.diff(ts) > 0))
    ts = ts[uniq_mask]
    if ts.size < 2:
        fps = float(h5f.attrs.get("capture_fps_effective_median", h5f.attrs.get("fps", 30.0)))
        return None, fps, fps, 0.0

    dt = np.diff(ts)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        fps = float(h5f.attrs.get("capture_fps_effective_median", h5f.attrs.get("fps", 30.0)))
        return None, fps, fps, 0.0

    fps_mean = float((ts.size - 1) / (ts[-1] - ts[0])) if ts[-1] > ts[0] else float(h5f.attrs.get("fps", 30.0))
    fps_median = float(1.0 / np.median(dt))
    jitter_std_ms = float(1000.0 * np.std(dt))
    return ts, fps_mean, fps_median, jitter_std_ms


def reamostrar_rgb(rgb: np.ndarray, ts: np.ndarray, fps_uniforme: float) -> tuple[np.ndarray, np.ndarray]:
    if ts is None or ts.size != rgb.shape[1]:
        return rgb, np.arange(rgb.shape[1], dtype=np.float64) / fps_uniforme

    if ts[-1] <= ts[0]:
        return rgb, np.arange(rgb.shape[1], dtype=np.float64) / fps_uniforme

    t_uniforme = np.arange(ts[0], ts[-1] + (0.5 / fps_uniforme), 1.0 / fps_uniforme, dtype=np.float64)
    if t_uniforme.size < 2:
        return rgb, ts

    rgb_uniforme = np.empty((rgb.shape[0], t_uniforme.size), dtype=np.float64)
    for ch in range(rgb.shape[0]):
        rgb_uniforme[ch, :] = np.interp(t_uniforme, ts, rgb[ch, :])
    return rgb_uniforme, t_uniforme


def main():
    if not os.path.exists(ARQUIVO_ENTRADA):
        print("Arquivo não encontrado!")
        return

    with h5py.File(ARQUIVO_ENTRADA, "r") as h5f:
        video = h5f["video_bruto"]
        total_frames, _, _, _ = video.shape

        ts_all, fps_mean, fps_median, jitter_std_ms = analisar_timestamps(h5f, total_frames)
        fps_nominal = float(h5f.attrs.get("fps", 30.0))
        sensor_bits = int(h5f.attrs.get("sensor_bit_depth", 12))
        container_bits = int(h5f.attrs.get("storage_container_bits", 16))

        sample = video[: min(5, total_frames)]
        alignment = inferir_alinhamento(sample, sensor_bits, container_bits)
        sample_min = int(sample.min())
        sample_max = int(sample.max())

        print(f"FPS nominal: {fps_nominal:.5f}")
        print(f"FPS efetivo (médio): {fps_mean:.5f}")
        print(f"FPS efetivo (mediana): {fps_median:.5f}")
        print(f"Jitter std: {jitter_std_ms:.3f} ms")
        print(f"Frames: {total_frames}")
        print(f"Faixa observada no arquivo: min={sample_min}, max={sample_max}")
        print(f"Inferência de armazenamento: {alignment}")

        frame0_12 = converter_para_12bit_logico(video[0], sensor_bits, container_bits, alignment)
        frame_ref = frame12_para_display8(frame0_12, sensor_bits)

        roi = cv2.selectROI("Selecione a Pele e aperte ENTER", frame_ref, fromCenter=False)
        cv2.destroyAllWindows()
        x, y, w, h = roi
        if w <= 0 or h <= 0:
            raise RuntimeError("ROI inválida.")

        print(f"Processando {total_frames} frames...")
        rgb_12 = np.zeros((3, total_frames), dtype=np.float64)
        rgb_8 = np.zeros((3, total_frames), dtype=np.float64)

        for i in range(total_frames):
            frame_roi_u16 = video[i][y:y + h, x:x + w]
            frame12 = converter_para_12bit_logico(frame_roi_u16, sensor_bits, container_bits, alignment)
            frame8 = (frame12 >> max(sensor_bits - 8, 0)).astype(np.uint8)

            mean_bgr_12 = np.mean(frame12, axis=(0, 1))
            mean_bgr_8 = np.mean(frame8, axis=(0, 1))

            rgb_12[:, i] = np.array([mean_bgr_12[2], mean_bgr_12[1], mean_bgr_12[0]], dtype=np.float64)
            rgb_8[:, i] = np.array([mean_bgr_8[2], mean_bgr_8[1], mean_bgr_8[0]], dtype=np.float64)

        if total_frames <= FS_CORTAR_INICIO + 10:
            raise RuntimeError("Vídeo muito curto após corte inicial.")

        rgb_12_cut = rgb_12[:, FS_CORTAR_INICIO:]
        rgb_8_cut = rgb_8[:, FS_CORTAR_INICIO:]
        ts_cut = None if ts_all is None else ts_all[FS_CORTAR_INICIO:]

        if ts_cut is not None:
            uniq_mask = np.concatenate(([True], np.diff(ts_cut) > 0))
            ts_cut = ts_cut[uniq_mask]
            rgb_12_cut = rgb_12_cut[:, uniq_mask]
            rgb_8_cut = rgb_8_cut[:, uniq_mask]

        fps_analise = fps_mean if fps_mean > 0 else fps_nominal
        rgb_12_uniforme, t_uniforme = reamostrar_rgb(rgb_12_cut, ts_cut, fps_analise)
        rgb_8_uniforme, _ = reamostrar_rgb(rgb_8_cut, ts_cut, fps_analise)

        sig_12 = aplicar_pos(rgb_12_uniforme)
        sig_8 = aplicar_pos(rgb_8_uniforme)

        bvp_12 = filtro_passa_banda(detrend(sig_12), fps_analise)
        bvp_8 = filtro_passa_banda(detrend(sig_8), fps_analise)

        n = len(bvp_12)
        xf = fftfreq(n, 1.0 / fps_analise)[: n // 2]
        yf_12 = np.abs(fft(bvp_12))[: n // 2]
        yf_8 = np.abs(fft(bvp_8))[: n // 2]

        bpm_12, snr_12 = estimar_bpm_e_snr(yf_12, xf)
        bpm_8, snr_8 = estimar_bpm_e_snr(yf_8, xf)

        print("\nResumo da comparação:")
        print(f"12-bit preservado -> pico: {bpm_12:.2f} bpm | SNR espectral: {snr_12:.2f} dB")
        print(f"8-bit quantizado -> pico: {bpm_8:.2f} bpm | SNR espectral: {snr_8:.2f} dB")
        print(f"FFT calculada com FPS de análise: {fps_analise:.5f} Hz")

        xf_bpm = xf * 60.0
        banda = (xf_bpm >= BPM_MIN) & (xf_bpm <= BPM_MAX)

        tempo_plot = t_uniforme - t_uniforme[0] if t_uniforme.size == len(bvp_12) else np.arange(len(bvp_12)) / fps_analise

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(tempo_plot, bvp_12, label="12-bit preservado")
        plt.plot(tempo_plot, bvp_8, label="8-bit quantizado", alpha=0.7)
        plt.title("Comparação rPPG: 12-bit vs 8-bit")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(xf_bpm[banda], yf_12[banda], label=f"12-bit | pico={bpm_12:.1f} bpm | SNR={snr_12:.1f} dB")
        plt.plot(xf_bpm[banda], yf_8[banda], label=f"8-bit | pico={bpm_8:.1f} bpm | SNR={snr_8:.1f} dB", alpha=0.7)
        plt.xlim(BPM_MIN, BPM_MAX)
        plt.xlabel("BPM")
        plt.ylabel("|FFT|")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        print("Iniciando validação visual...")
        for i in range(0, total_frames, 2):
            frame12 = converter_para_12bit_logico(video[i], sensor_bits, container_bits, alignment)
            frame_val = frame12_para_display8(frame12, sensor_bits)
            cv2.rectangle(frame_val, (x, y), (x + w, y + h), (0, 255, 0), 2)

            fator = 480 / frame_val.shape[0]
            display = cv2.resize(frame_val, (int(frame_val.shape[1] * fator), 480))
            cv2.imshow("Validacao ROI (12-bit logico -> 8-bit display)", display)
            if cv2.waitKey(max(1, int(1000 / max(fps_analise, 1e-3)))) & 0xFF == 27:
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
