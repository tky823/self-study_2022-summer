import time

import numpy as np
import scipy.signal as ss
from mir_eval.separation import bss_eval_sources
from ssspy.algorithm import projection_back


class SDRiReporter:
    def __init__(
        self,
        waveform_src_img,
        n_fft=4096,
        hop_length=2048,
        window="hann",
        reference_id=0,
        save_freq=10,
        offset_time=0,
    ):
        self.waveform_src_img = waveform_src_img

        self.n_fft, self.hop_length = n_fft, hop_length
        self.window = window
        self.reference_id = reference_id
        self.n_samples = self.waveform_src_img.shape[-1]

        self.save_freq = save_freq
        self.offset_time = offset_time
        self.iter_idx = 0

    def __call__(self, method):
        n_fft, hop_length = self.n_fft, self.hop_length
        window = self.window
        reference_id = self.reference_id
        n_samples = self.n_samples

        if not hasattr(self, "start"):
            self.start = time.perf_counter()
            self.loss_time = 0

        if self.iter_idx % self.save_freq == 0:
            loss_time_start = time.perf_counter()

            if method.spatial_algorithm in ["IP", "IP1", "IP2"]:
                spectrogram_mix, demix_filter = method.input, method.demix_filter
                spectrogram_est = method.separate(
                    spectrogram_mix, demix_filter=demix_filter
                )
            else:
                spectrogram_mix, spectrogram_est = method.input, method.output

            if not hasattr(method, "sdr_mix"):
                _, waveform_mix = ss.istft(
                    spectrogram_mix,
                    window=window,
                    nperseg=n_fft,
                    noverlap=n_fft - hop_length,
                )
                waveform_mix = waveform_mix[..., :n_samples]

                method.sdr_mix, _, _, _ = bss_eval_sources(
                    self.waveform_src_img[reference_id], waveform_mix
                )
                sdr_est = method.sdr_mix
            else:
                spectrogram_est = projection_back(
                    spectrogram_est,
                    reference=spectrogram_mix,
                    reference_id=reference_id,
                )

                _, waveform_est = ss.istft(
                    spectrogram_est,
                    window=window,
                    nperseg=n_fft,
                    noverlap=n_fft - hop_length,
                )
                waveform_est = waveform_est[..., :n_samples]

                sdr_est, _, _, _ = bss_eval_sources(
                    self.waveform_src_img[reference_id], waveform_est
                )

            sdri = np.mean(sdr_est - method.sdr_mix)

            method.sdri.append(sdri)

            loss_time_end = time.perf_counter()
            self.loss_time += loss_time_end - loss_time_start

            method.times.append(
                loss_time_end - self.start - self.loss_time + self.offset_time
            )

        self.iter_idx += 1
