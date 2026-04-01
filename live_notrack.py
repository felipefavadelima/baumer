import cv2
import neoapi
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, detrend
from collections import deque
import time

# =========================================================
# CONFIGURAÇÃO
# =========================================================
ALVO_FPS = 30.0
ALTURA_MAX_DISPLAY = 600

# Buffer total
JANELA_SEG = 20.0

# Atualizações
FC_UPDATE_SEG = 0.75
RPPG_PLOT_UPDATE_SEG = 0.20
FFT_UPDATE_SEG = 10.0

# Faixa cardíaca
FC_MIN_BPM = 45.0
FC_MAX_BPM = 180.0

# Janela mostrada no gráfico temporal
RPPG_VIEW_SEG = 10.0

# Duração mínima para começar estimativa
MIN_SEG_PARA_ESTIMAR = 5.0


# =========================================================
# UTILITÁRIOS
# =========================================================
def resize_for_display(frame, altura_max=ALTURA_MAX_DISPLAY):
    h, w = frame.shape[:2]
    fator = altura_max / h
    largura_v = int(w * fator)
    altura_v = int(h * fator)
    frame_display = cv2.resize(frame, (largura_v, altura_v), interpolation=cv2.INTER_LINEAR)
    return frame_display, fator


def desenhar_texto(frame, linhas, x=10, y0=25, dy=22, cor=(0, 255, 0)):
    y = y0
    for linha in linhas:
        cv2.putText(
            frame, linha, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor, 1, cv2.LINE_AA
        )
        y += dy


def aplicar_pos(rgb_signal):
    """
    rgb_signal shape = (3, N), ordem B,G,R
    """
    mean_rgb = np.mean(rgb_signal, axis=1, keepdims=True)
    if np.any(mean_rgb == 0):
        return None

    cnorm = rgb_signal / mean_rgb - 1.0
    s = np.dot(np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float64), cnorm)

    std2 = np.std(s[1, :])
    if std2 < 1e-12:
        return None

    alpha = np.std(s[0, :]) / std2
    bvp = s[0, :] + alpha * s[1, :]
    return bvp


def filtro_passa_banda(sinal, fps, fc_min_bpm=FC_MIN_BPM, fc_max_bpm=FC_MAX_BPM):
    nyq = 0.5 * fps
    f1 = (fc_min_bpm / 60.0) / nyq
    f2 = (fc_max_bpm / 60.0) / nyq

    if f1 <= 0 or f2 >= 1 or f1 >= f2:
        return None

    b, a = butter(3, [f1, f2], btype='band')
    return filtfilt(b, a, sinal)


def processar_rppg(rgb_buffer, fps):
    rgb_buffer = np.asarray(rgb_buffer, dtype=np.float64)
    if rgb_buffer.ndim != 2 or rgb_buffer.shape[0] < int(MIN_SEG_PARA_ESTIMAR * fps):
        return None, None

    rgb_signal = rgb_buffer.T  # (3, N)
    bvp = aplicar_pos(rgb_signal)
    if bvp is None:
        return None, None

    bvp = detrend(bvp)
    bvp_f = filtro_passa_banda(bvp, fps)
    if bvp_f is None:
        return None, None

    return bvp, bvp_f


def estimar_fc_fft(bvp_f, fps):
    if bvp_f is None or len(bvp_f) < 16:
        return None, None, None

    n = len(bvp_f)
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(bvp_f * win))
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    bpm = freqs * 60.0

    mask = (bpm >= FC_MIN_BPM) & (bpm <= FC_MAX_BPM)
    if not np.any(mask):
        return None, bpm, spec

    spec_mask = spec[mask]
    bpm_mask = bpm[mask]
    idx = np.argmax(spec_mask)
    fc_estim = bpm_mask[idx]

    return fc_estim, bpm, spec


def preparar_graficos():
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    fig.suptitle("Monitor rPPG ao vivo")

    # rPPG temporal
    ax1.set_title("rPPG filtrado ao vivo")
    ax1.set_xlabel("Tempo relativo (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)
    line_rppg, = ax1.plot([], [], lw=1.2)

    # FFT
    ax2.set_title("FFT")
    ax2.set_xlabel("BPM")
    ax2.set_ylabel("Magnitude")
    ax2.set_xlim(FC_MIN_BPM - 10, FC_MAX_BPM + 10)
    ax2.grid(True)
    line_fft, = ax2.plot([], [], lw=1.2)
    line_fc = ax2.axvline(0, linestyle='--', visible=False)

    fig.tight_layout()
    fig.canvas.draw_idle()
    plt.pause(0.001)

    return fig, ax1, ax2, line_rppg, line_fft, line_fc


def atualizar_plot_rppg(ax1, line_rppg, bvp_f, fps):
    if bvp_f is None or len(bvp_f) < 2:
        return

    n_view = min(len(bvp_f), int(RPPG_VIEW_SEG * fps))
    y = bvp_f[-n_view:]
    t = np.arange(n_view) / fps
    t = t - t[-1]

    line_rppg.set_data(t, y)

    ymin = float(np.min(y))
    ymax = float(np.max(y))
    if np.isclose(ymin, ymax):
        ymin -= 1.0
        ymax += 1.0

    ax1.set_xlim(t[0], t[-1] if len(t) > 1 else 0.0)
    ax1.set_ylim(ymin, ymax)


def atualizar_plot_fft(ax2, line_fft, line_fc, bpm, spec, fc):
    if bpm is None or spec is None or len(bpm) == 0:
        return

    mask = (bpm >= FC_MIN_BPM - 10) & (bpm <= FC_MAX_BPM + 10)
    x = bpm[mask]
    y = spec[mask]

    if len(x) == 0:
        return

    line_fft.set_data(x, y)

    ymax = float(np.max(y))
    if ymax <= 0:
        ymax = 1.0

    ax2.set_xlim(x[0], x[-1])
    ax2.set_ylim(0, ymax * 1.05)

    if fc is not None:
        line_fc.set_xdata([fc, fc])
        line_fc.set_visible(True)
    else:
        line_fc.set_visible(False)


# =========================================================
# FLUXO PRINCIPAL
# =========================================================
def main():
    camera = None
    try:
        camera = neoapi.Cam()
        camera.Connect()
        modelo = camera.f.DeviceModelName.value
        print(f"Câmera conectada: {modelo}")

        try:
            camera.f.OffsetX.value = camera.f.OffsetX.GetMin()
            camera.f.OffsetY.value = camera.f.OffsetY.GetMin()
            camera.f.Width.value = camera.f.Width.GetMax()
            camera.f.Height.value = camera.f.Height.GetMax()
        except Exception:
            pass

        camera.f.ExposureAuto.value = neoapi.ExposureAuto_Off
        camera.f.GainAuto.value = neoapi.GainAuto_Off
        camera.f.DeviceLinkThroughputLimit.value = camera.f.DeviceLinkThroughputLimit.GetMax()

        exposicao_us = (1000000.0 / ALVO_FPS) - 2000.0
        if exposicao_us < 100:
            exposicao_us = 100

        camera.f.ExposureTime.value = exposicao_us
        camera.f.AcquisitionFrameRateEnable.value = True
        camera.f.AcquisitionFrameRate.value = ALVO_FPS

        try:
            camera.f.PixelFormat.value = neoapi.PixelFormat_BayerRG8
        except Exception:
            pass

        print("\n[LIVE PREVIEW]")
        print("W/S = ganho | ENTER = selecionar ROI e iniciar | Q = sair")

        cv2.namedWindow("rPPG Live", cv2.WINDOW_AUTOSIZE)

        ultima_img_full = None
        fator_escala = 1.0

        # -----------------------------------------------------
        # PREVIEW + AJUSTE
        # -----------------------------------------------------
        while True:
            img = camera.GetImage()
            if not img.IsEmpty():
                frame_full = img.Convert("BGR8").GetNPArray()
                ultima_img_full = frame_full.copy()

                h_orig, w_orig = frame_full.shape[:2]
                frame_display, fator_escala = resize_for_display(frame_full)

                ganho_atual = camera.f.Gain.value
                linhas = [
                    f"Modelo: {modelo}",
                    f"Resolucao: {w_orig}x{h_orig}",
                    f"FPS alvo: {ALVO_FPS:.1f}",
                    f"Exposicao: {camera.f.ExposureTime.value:.0f} us",
                    f"Ganho: {ganho_atual:.2f}",
                    "W/S ganho | ENTER ROI | Q sair"
                ]
                desenhar_texto(frame_display, linhas)
                cv2.imshow("rPPG Live", frame_display)

            key = cv2.waitKey(1) & 0xFF
            if key == 13:
                break
            elif key in (ord('q'), ord('Q')):
                cv2.destroyAllWindows()
                camera.Disconnect()
                return
            elif key in (ord('w'), ord('W')):
                try:
                    camera.f.Gain.value = min(camera.f.Gain.GetMax(), camera.f.Gain.value + 0.5)
                except Exception:
                    pass
            elif key in (ord('s'), ord('S')):
                try:
                    camera.f.Gain.value = max(camera.f.Gain.GetMin(), camera.f.Gain.value - 0.5)
                except Exception:
                    pass

        if ultima_img_full is None:
            raise RuntimeError("Não foi possível obter frame da câmera.")

        # -----------------------------------------------------
        # SELEÇÃO ROI
        # -----------------------------------------------------
        frame_sel_display, fator_escala = resize_for_display(ultima_img_full)
        roi_small = cv2.selectROI(
            "Selecione a ROI e aperte ENTER",
            frame_sel_display,
            fromCenter=False,
            showCrosshair=True
        )
        cv2.destroyWindow("Selecione a ROI e aperte ENTER")

        x = int(roi_small[0] / fator_escala)
        y = int(roi_small[1] / fator_escala)
        w = int(roi_small[2] / fator_escala)
        h = int(roi_small[3] / fator_escala)

        if w <= 0 or h <= 0:
            raise RuntimeError("ROI inválida.")

        # -----------------------------------------------------
        # BUFFERS / ESTADO
        # -----------------------------------------------------
        n_janela = int(JANELA_SEG * ALVO_FPS)
        rgb_buffer = deque(maxlen=n_janela)

        ultima_fc = None
        qualidade = "Aguardando..."
        fps_medido = ALVO_FPS

        t0 = time.perf_counter()
        t_last_frame = t0
        t_last_fc = t0
        t_last_rppg_plot = t0
        t_last_fft = t0

        frame_counter_fps = 0
        t_ref_fps = t0

        bvp_f_cache = None
        bpm_cache = None
        spec_cache = None

        fig, ax1, ax2, line_rppg, line_fft, line_fc = preparar_graficos()

        print("\n[ESTIMATIVA AO VIVO OTIMIZADA]")
        print("R = reselecionar ROI | W/S = ganho | Q = sair")

        # -----------------------------------------------------
        # LOOP LIVE
        # -----------------------------------------------------
        while True:
            img = camera.GetImage()
            if img.IsEmpty():
                continue

            agora = time.perf_counter()
            frame_full = img.Convert("BGR8").GetNPArray()

            # medição simples de FPS real
            frame_counter_fps += 1
            if (agora - t_ref_fps) >= 1.0:
                fps_medido = frame_counter_fps / (agora - t_ref_fps)
                frame_counter_fps = 0
                t_ref_fps = agora

            H, W = frame_full.shape[:2]
            x = max(0, min(x, W - 1))
            y = max(0, min(y, H - 1))
            w = max(1, min(w, W - x))
            h = max(1, min(h, H - y))

            roi_frame = frame_full[y:y+h, x:x+w]
            if roi_frame.size > 0:
                media_bgr = np.mean(roi_frame, axis=(0, 1))
                rgb_buffer.append(media_bgr)

            # -------- processamento FC --------
            if len(rgb_buffer) >= int(MIN_SEG_PARA_ESTIMAR * ALVO_FPS) and (agora - t_last_fc) >= FC_UPDATE_SEG:
                _, bvp_f = processar_rppg(rgb_buffer, ALVO_FPS)
                bvp_f_cache = bvp_f

                if bvp_f_cache is not None:
                    fc_tmp, _, _ = estimar_fc_fft(bvp_f_cache, ALVO_FPS)
                    if fc_tmp is not None:
                        ultima_fc = fc_tmp
                        qualidade = "OK"
                    else:
                        qualidade = "Sinal fraco"
                else:
                    qualidade = "Sinal fraco"

                t_last_fc = agora

            # -------- atualização do gráfico rPPG --------
            if bvp_f_cache is not None and (agora - t_last_rppg_plot) >= RPPG_PLOT_UPDATE_SEG:
                atualizar_plot_rppg(ax1, line_rppg, bvp_f_cache, ALVO_FPS)
                fig.canvas.draw_idle()
                plt.pause(0.001)
                t_last_rppg_plot = agora

            # -------- FFT a cada 10 s --------
            if bvp_f_cache is not None and (agora - t_last_fft) >= FFT_UPDATE_SEG:
                fc_fft, bpm_cache, spec_cache = estimar_fc_fft(bvp_f_cache, ALVO_FPS)
                if fc_fft is not None:
                    ultima_fc = fc_fft
                atualizar_plot_fft(ax2, line_fft, line_fc, bpm_cache, spec_cache, ultima_fc)
                fig.canvas.draw_idle()
                plt.pause(0.001)
                t_last_fft = agora

            # -------- display câmera --------
            frame_display, fator_escala = resize_for_display(frame_full)
            xd = int(round(x * fator_escala))
            yd = int(round(y * fator_escala))
            wd = int(round(w * fator_escala))
            hd = int(round(h * fator_escala))

            cv2.rectangle(frame_display, (xd, yd), (xd + wd, yd + hd), (0, 255, 0), 2)

            linhas = [
                f"ROI: x={x} y={y} w={w} h={h}",
                f"Ganho: {camera.f.Gain.value:.2f}",
                f"FPS real: {fps_medido:.1f}",
                f"Buffer: {len(rgb_buffer)}/{n_janela} frames",
                f"Qualidade: {qualidade}",
                f"FC: {'--' if ultima_fc is None else f'{ultima_fc:.1f} BPM'}",
                "R reselecionar ROI | W/S ganho | Q sair"
            ]
            desenhar_texto(frame_display, linhas)
            cv2.imshow("rPPG Live", frame_display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            elif key in (ord('w'), ord('W')):
                try:
                    camera.f.Gain.value = min(camera.f.Gain.GetMax(), camera.f.Gain.value + 0.5)
                except Exception:
                    pass
            elif key in (ord('s'), ord('S')):
                try:
                    camera.f.Gain.value = max(camera.f.Gain.GetMin(), camera.f.Gain.value - 0.5)
                except Exception:
                    pass
            elif key in (ord('r'), ord('R')):
                pausa = frame_full.copy()
                pausa_display, fator_tmp = resize_for_display(pausa)
                roi_small = cv2.selectROI(
                    "Reselecione a ROI e aperte ENTER",
                    pausa_display,
                    fromCenter=False,
                    showCrosshair=True
                )
                cv2.destroyWindow("Reselecione a ROI e aperte ENTER")

                x_new = int(roi_small[0] / fator_tmp)
                y_new = int(roi_small[1] / fator_tmp)
                w_new = int(roi_small[2] / fator_tmp)
                h_new = int(roi_small[3] / fator_tmp)

                if w_new > 0 and h_new > 0:
                    x, y, w, h = x_new, y_new, w_new, h_new
                    rgb_buffer.clear()
                    ultima_fc = None
                    qualidade = "ROI resetada"
                    bvp_f_cache = None
                    bpm_cache = None
                    spec_cache = None
                    t_last_fc = agora
                    t_last_rppg_plot = agora
                    t_last_fft = agora

                    line_rppg.set_data([], [])
                    line_fft.set_data([], [])
                    line_fc.set_visible(False)
                    fig.canvas.draw_idle()
                    plt.pause(0.001)

    except Exception as exc:
        print(f"\nErro crítico: {exc}")

    finally:
        cv2.destroyAllWindows()
        plt.close('all')
        try:
            if camera is not None and camera.IsConnected():
                camera.Disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()