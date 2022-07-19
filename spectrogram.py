import librosa
from scipy.fft import fft
import numpy as np
import matplotlib.pyplot as plt
import time
import plotly.graph_objects as go
import math
from scipy.signal import savgol_filter


def DEBUG_plot(xf, yf):
    plt.plot(xf, yf)
    plt.show()


def show_mesh(x, y, z):
    fig = go.Figure(
        data=[go.Mesh3d(x=x, y=y, z=z, color='lightpink', opacity=0.50)])
    fig.show()


def build_frame_verts(samples, sampling_rate, timestamp, frame_idx):
    n = len(samples)
    T = 1 / sampling_rate
    fft_points = fft(samples)
    yf = 2.0/n * np.log10(np.abs(fft_points[:n//2]))
    xf = np.linspace(0.0, 1.0 / (2.0 * T), n//2)

    # DEBUG_plot(xf, yf)

    return list(map(lambda coords: (coords[0], coords[1], frame_idx), (zip(xf, yf))))


def build_mesh_from_frames(frames):
    """
    - sliding window of two frames at a time
        - sliding window of two points at a time
    - mesh algo (build a square plane from two triangle polys):
        f = frame index
        i = point index
        TRIANGLE(frames[f][i],   frames[f+1][i],   frames[f][i+1])
        TRIANGLE(frames[f+1][i], frames[f+1][i+1], frames[f][i+1])
    """


def get_timestamp(playhead_t):
    seconds = playhead_t // 1
    minutes = playhead_t // 60
    print(playhead_t, seconds, minutes)
    timestamp = '{}:{}'.format(minutes, seconds)
    return timestamp


def getFFT(data, rate, chunk_size, log_scale=False):
    data = data * np.hamming(len(data))
    try:
        FFT = np.abs(np.fft.rfft(data)[1:])
    except:
        FFT = np.fft.fft(data)
        left, right = np.split(np.abs(FFT), 2)
        FFT = np.add(left, right[::-1])

    # fftx = np.fft.fftfreq(chunk_size, d=1.0/rate)
    # fftx = np.split(np.abs(fftx), 2)[0]

    if log_scale:
        try:
            FFT = np.multiply(20, np.log10(FFT))
        except Exception as e:
            print('Log(FFT) failed: %s' % str(e))

    return FFT


def round_up_to_even(f):
    return int(math.ceil(f / 2.) * 2)


def spectrogram():
    # frames = []
    fft_window_size_ms = 100
    duration = 0.1
    for frame_idx in range(5):
        samples, sampling_rate = librosa.load(
            'feel.wav', sr=None, mono=True, offset=frame_idx, duration=duration)
        timestamp = get_timestamp(frame_idx * duration)
        frame = build_frame_verts(samples, sampling_rate, timestamp, frame_idx)

        n_frequency_bins = 1400
        # filter_width = 10
        filter_width = round_up_to_even(0.03*n_frequency_bins) - 1

        fft = getFFT(samples, sampling_rate, 10, log_scale=False)
        fft_window_size = round_up_to_even(
            sampling_rate * fft_window_size_ms / 1000)
        fftx = np.arange(int(fft_window_size/2), dtype=float) * \
            sampling_rate / fft_window_size
        power_normalization_coefficients = np.logspace(np.log2(1), np.log2(
            np.log2(sampling_rate/2)), len(fftx), endpoint=True, base=2, dtype=None)
        fft = fft * power_normalization_coefficients
        fftx_bin_indices = np.logspace(np.log2(len(fftx)), 0, len(
            fftx), endpoint=True, base=2, dtype=None) - 1
        fftx_bin_indices = np.round(
            ((fftx_bin_indices - np.max(fftx_bin_indices))*-1) / (len(fftx) / n_frequency_bins), 0).astype(int)
        fftx_bin_indices = np.minimum(
            np.arange(len(fftx_bin_indices)), fftx_bin_indices - np.min(fftx_bin_indices))

        frequency_bin_energies = np.zeros(n_frequency_bins)
        frequency_bin_centres = np.zeros(n_frequency_bins)
        fftx_indices_per_bin = []
        for bin_index in range(n_frequency_bins):
            bin_frequency_indices = np.where(fftx_bin_indices == bin_index)
            fftx_indices_per_bin.append(bin_frequency_indices)
            fftx_frequencies_this_bin = fftx[bin_frequency_indices]
            frequency_bin_centres[bin_index] = np.mean(
                fftx_frequencies_this_bin)

        for bin_index in range(n_frequency_bins):
            frequency_bin_energies[bin_index] = np.mean(
                fft[fftx_indices_per_bin[bin_index]])

        frequency_bin_energies = np.nan_to_num(
            frequency_bin_energies, copy=True)
        if filter_width > 3:
            frequency_bin_energies = savgol_filter(
                frequency_bin_energies, filter_width, 3)
        frequency_bin_energies[frequency_bin_energies < 0] = 0

        print(len(frequency_bin_energies))

        DEBUG_plot(np.arange(0, n_frequency_bins, 1), frequency_bin_energies)

        # frames.append(frame)
        # flat_verts_list = [x for xs in frames for x in xs]
        # xs, ys, zs = [], [], []
        # for (x, y, z) in flat_verts_list:
        #     xs.append(x)
        #     ys.append(y)
        #     zs.append(z)
        # show_mesh(xs, zs, ys)


if __name__ == '__main__':
    spectrogram()
