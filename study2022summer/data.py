import os
import shutil
import urllib.request

import numpy as np
import scipy.signal as ss
from scipy.io import wavfile, loadmat

sisec2010_tags = ["dev1_female3", "dev1_female4"]


def download_data(
    sisec2010_root=".data/SiSEC2010",
    mird_root=".data/MIRD",
    n_sources=3,
    sisec2010_tag="dev1_female3",
    degrees=[0, 15, 345, 30, 330, 45, 315, 60, 300, 75, 285, 90, 270],
    channels=[3, 4, 2, 5, 1, 6, 0, 7],
    max_samples=160000,
):
    assert sisec2010_tag in sisec2010_tags, "Choose sisec2010_tag from {}".format(
        sisec2010_tags
    )

    sisec2010_npz_path = download_sisec2010(
        root=sisec2010_root, n_sources=n_sources, tag=sisec2010_tag
    )
    sisec2010_npz = np.load(sisec2010_npz_path)

    mird_npz_path = download_mird(
        root=mird_root, n_sources=n_sources, degrees=degrees, channels=channels
    )
    mird_npz = np.load(mird_npz_path)

    waveform_src_img = []

    for src_idx in range(n_sources):
        key = "src_{}".format(src_idx + 1)
        waveform_src = sisec2010_npz[key][:max_samples]
        n_samples = len(waveform_src)
        _waveform_src_img = []

        for waveform_rir in mird_npz[key]:
            waveform_conv = np.convolve(waveform_src, waveform_rir)[:n_samples]
            _waveform_src_img.append(waveform_conv)

        _waveform_src_img = np.stack(_waveform_src_img, axis=0)
        waveform_src_img.append(_waveform_src_img)

    waveform_src_img = np.stack(waveform_src_img, axis=1)

    return waveform_src_img


def download_sisec2010(root=".data/SiSEC2010", n_sources=3, tag="dev1_female3"):
    sample_rate = 16000
    filename = "dev1.zip"
    url = "http://www.irisa.fr/metiss/SiSEC10/underdetermined/{}".format(filename)
    zip_path = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)

    if not os.path.exists(os.path.join(root, "{}_inst_matrix.mat".format(tag))):
        shutil.unpack_archive(zip_path, root)

    source_paths = []

    for src_idx in range(n_sources):
        source_path = os.path.join(root, "{}_src_{}.wav".format(tag, src_idx + 1))
        source_paths.append(source_path)

    source_paths = source_paths[:n_sources]

    n_channels = n_sources
    npz_path = os.path.join(root, "SiSEC2010-{}ch.npz".format(n_channels))

    assert n_channels == n_sources, "Mixing system should be determined."

    if not os.path.exists(npz_path):
        dry_sources = {}

        for src_idx, source_path in enumerate(source_paths):
            _, data = wavfile.read(source_path)  # 16 bits
            dry_sources["src_{}".format(src_idx + 1)] = data / 2**15

        np.savez(
            npz_path,
            sample_rate=sample_rate,
            n_sources=n_sources,
            n_channels=n_channels,
            **dry_sources
        )

    return npz_path


def download_cmu_arctic(root=".data/cmu_arctic", tags=["awb", "bdl", "clb"]):
    sample_rate = 16000
    n_channels = n_sources = len(tags)

    os.makedirs(root, exist_ok=True)

    for tag in tags:
        filename = "cmu_us_{}_arctic-0.95-release.tar.bz2".format(tag)
        url = "http://festvox.org/cmu_arctic/cmu_arctic/packed/{}".format(filename)
        zip_path = os.path.join(root, filename)

        if not os.path.exists(zip_path):
            urllib.request.urlretrieve(url, zip_path)

        if not os.path.exists(
            os.path.join(root, "cmu_us_{}_arctic".format(tag), "wav")
        ):
            shutil.unpack_archive(zip_path, root)

    source_paths = []

    for tag_idx, tag in enumerate(tags):
        source_path = os.path.join(
            root,
            "cmu_us_{}_arctic".format(tag),
            "wav",
            "arctic_a{:04d}.wav".format(tag_idx + 1),
        )
        source_paths.append(source_path)

    npz_path = os.path.join(root, "cmu_arctic_{}.npz".format("-".join(tags)))

    if not os.path.exists(npz_path):
        dry_sources = {}

        for src_idx, source_path in enumerate(source_paths):
            _, data = wavfile.read(source_path)  # 16 bits
            dry_sources["src_{}".format(src_idx + 1)] = data / 2**15

        np.savez(
            npz_path,
            sample_rate=sample_rate,
            n_sources=n_sources,
            n_channels=n_channels,
            **dry_sources
        )

    return npz_path


def download_mird(root=".data/MIRD", n_sources=3, degrees=None, channels=None):
    filename = (
        "Impulse_response_Acoustic_Lab_Bar-Ilan_University__"
        "Reverberation_0.160s__3-3-3-8-3-3-3.zip"
    )
    url = (
        "https://www.iks.rwth-aachen.de/fileadmin/user_upload/downloads/"
        "forschung/tools-downloads/{filename}"
    )
    url = url.format(filename=filename)
    zip_path = os.path.join(root, filename)

    if degrees is None:
        degrees = [30, 345, 0, 60, 315]

    if channels is None:
        channels = [3, 4, 2, 5, 1, 6, 0, 7]

    sample_rate = 16000
    duration = 0.160

    degrees = degrees[:n_sources]
    channels = channels[:n_sources]

    n_channels = len(channels)
    n_samples = int(sample_rate * duration)

    template_rir_name = (
        "Impulse_response_Acoustic_Lab_Bar-Ilan_University_"
        "(Reverberation_{:.3f}s)_3-3-3-8-3-3-3_1m_{:03d}.mat"
    )

    os.makedirs(root, exist_ok=True)

    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)

    if not os.path.exists(os.path.join(root, template_rir_name.format(0.16, 0))):
        shutil.unpack_archive(zip_path, root)

    degrees_name = "-".join([str(degree) for degree in degrees])
    channels_name = "-".join([str(channel) for channel in channels])
    npz_path = os.path.join(root, "MIRD_{}_{}.npz".format(degrees_name, channels_name))

    assert n_channels == n_sources, "Mixing system should be determined."

    if not os.path.exists(npz_path):
        rirs = {}

        for src_idx, degree in enumerate(degrees):
            rir_path = os.path.join(root, template_rir_name.format(duration, degree))
            rir = resample_mird_rir(rir_path, sample_rate_out=sample_rate)
            rirs["src_{}".format(src_idx + 1)] = rir[channels, :n_samples]

        np.savez(
            npz_path,
            sample_rate=sample_rate,
            n_sources=n_sources,
            n_channels=n_channels,
            **rirs
        )

    return npz_path


def resample_mird_rir(rir_path: str, sample_rate_out: int) -> np.ndarray:
    sample_rate_in = 48000
    rir_mat = loadmat(rir_path)
    rir = rir_mat["impulse_response"]

    rir_resampled = ss.resample_poly(rir, sample_rate_out, sample_rate_in, axis=0)

    return rir_resampled.T
