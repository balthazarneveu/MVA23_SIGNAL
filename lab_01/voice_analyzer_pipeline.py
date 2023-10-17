import numpy as np 
import matplotlib.pylab as plt 

from interactive_pipe.data_objects.curves import Curve
from interactive_pipe import interactive, interactive_pipeline
import scipy
from numpy.fft import fft,fftfreq,fftshift


def load_signal(path: str, global_params={}) -> np.array:
    rate, signal = scipy.io.wavfile.read(path)
    global_params["sampling_rate"] = rate
    return signal


@interactive(
    time_selection=(0., [0., 100.], "selected time [%]"), 
    window=(0.3, [0.01, 1.], "window size [s]")
)
def trim_signal(
    signal,
    time_selection: float=0.,
    window: float = 0.3,
    global_params={}
):
    rate = global_params["sampling_rate"]
    full_time = np.arange(len(signal))/rate
    full_signal = signal/(2.**14-1)
    temporal_signal_length = len(signal)/rate
    time_center  = time_selection/100.*temporal_signal_length
    time_start = max(0,time_center-window/2.)
    time_end = min(temporal_signal_length, time_center+window/2.)
    index_start, index_end = int(time_start*rate), min(int(time_end*rate), len(signal))
    time = full_time[index_start:index_end] 
    current_signal = full_signal[index_start:index_end]
    global_params["time_start"] = time_start
    global_params["time_end"] = time_end 
    return time, current_signal



def visualize_signal(timeline, signal, global_params={}):
    curve = Curve(
        [
            (timeline, signal, "k-")
        ],
        grid=True,
        title=f"signal {global_params['time_start']:.2f} {global_params['time_end']:.2f}",
        xlim=(timeline[0], timeline[-1]),
        ylim=(-1., 1.),
        xlabel="t[s]",
        ylabel="Audio amplitude"
    )
    return curve


def visualize_fourier(frequencies, fourier_coefficients, global_params={}):
    freq_axis = frequencies[len(frequencies)//2:]
    amplitude_axis = np.abs(fourier_coefficients[len(frequencies)//2:])
    curve = Curve(
        [
            (freq_axis, amplitude_axis, "b-")
        ],
        grid=True,
        xlabel="f[Hz]",
    )
    return curve

def compute_spectrum(x, global_params={}):
    fs = global_params["sampling_rate"]
    X=fft(x)/fs
    f= fftfreq(len(x),d=1/fs)
    X_ordered = fftshift(X)
    f_ordered=fftshift(f)
    return f_ordered, X_ordered

@interactive_pipeline(gui="mpl")
def audio_pipeline(audio_path):
    full_audio_signal = load_signal(audio_path)
    trimmed_timeline, trimmed_audio = trim_signal(full_audio_signal)
    sig_viz = visualize_signal(trimmed_timeline, trimmed_audio)
    freq, ampl = compute_spectrum(trimmed_audio)
    ft_viz = visualize_fourier(freq, ampl)
    return sig_viz, ft_viz

if __name__ == '__main__':
    path="Voice.wav"
    audio_pipeline(path)
    