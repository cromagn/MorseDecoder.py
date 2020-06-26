import logging
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sg
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy.fft import fftshift
from scipy.signal import butter, sosfilt, sosfreqz,lfilter

logging.basicConfig(level=logging.INFO)
def find_clock_frequency(spectrum):
    maxima = sp.signal.argrelextrema(spectrum, np.greater_equal)[0]
    while maxima[0] < 2:
        maxima = maxima[1:]
    if maxima.any():
        threshold = max(spectrum[2:-1])*0.8
        indices_above_threshold = np.argwhere(spectrum[maxima] > threshold)
        return maxima[indices_above_threshold[0]]
    else:
        return 0

def midpoint(a):
    mean_a = np.mean(a)
    mean_a_greater = np.ma.masked_greater(a, mean_a)
    high = np.ma.median(mean_a_greater)
    mean_a_less_or_equal = np.ma.masked_array(a, ~mean_a_greater.mask)
    low = np.ma.median(mean_a_less_or_equal)
    return (high + low) / 2

# whole packet clock recovery
# input: real valued NRZ-like waveform (array, tuple, or list)
#        must have at least 2 samples per symbol
#        must have at least 2 symbol transitions
# output: list of symbols
def wpcr(a):
    if len(a) < 4:
        return []
    b = (a > midpoint(a)) * 1.0
    d = np.diff(b)**2
    if len(np.argwhere(d > 0)) < 2:
        return []
    f = sp.fft.fft(d, len(a))
    p = find_clock_frequency(abs(f))
    if p == 0:
        return []
    cycles_per_sample = (p*1.0)/len(f)
    clock_phase = 0.5 + np.angle(f[p])/(tau)
    if clock_phase <= 0.5:
        clock_phase += 1
    symbols = []
    for i in range(len(a)):
        if clock_phase >= 1:
            clock_phase -= 1
            symbols.append(a[i])
        clock_phase += cycles_per_sample
    if debug:
        print("peak frequency index: %d / %d" % (p, len(f)))
        print("samples per symbol: %f" % (1.0/cycles_per_sample))
        print("clock cycles per sample: %f" % (cycles_per_sample))
        print("clock phase in cycles between 1st and 2nd samples: %f" % (clock_phase))
        print("clock phase in cycles at 1st sample: %f" % (clock_phase - cycles_per_sample/2))
        print("symbol count: %d" % (len(symbols)))
    return symbols

# convert soft symbols into bits (assuming binary symbols)
def slice_bits(symbols):
    symbols_average = np.average(symbols)
    bits = (symbols >= symbols_average)
    return np.array(bits, dtype=np.uint8)

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

tau = np.pi * 2
max_samples = 1000000
debug = True

#fs, data = wavfile.read('C:/tmp/satnogs_5_lp20024.wav')
#fs, data = wavfile.read('C:/tmp/satnogs_5_lphp.wav')
#fs, data = wavfile.read('C:/tmp/satnogs_sample.wav')
#fs, data = wavfile.read('C:/tmp/morse.wav')
fs, data = wavfile.read('C:/tmp/satnogs_6.wav')
#fs=44100
decimateFactor=4
ddec=sg.decimate(data,decimateFactor)
T = 1.0 / fs * decimateFactor
N=len(ddec)
x = np.linspace(0.0, N*T, N)
tSlice=0.1
nslice=tSlice/fs * decimateFactor

for x in range(0, N-nslice-1, nslice):
    yf = fft(ddec[x:x+nslice])
    xf = np.linspace(0.0, 1.0/(2.0*T), nslice//2)
    # plt.subplot(614)
    plt.plot(xf, 2.0/nslice * np.abs(yf[0:nslice//2]))
    plt.grid()
    plt.show()




yf = fft(ddec)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
# plt.subplot(614)
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.show()

f, Pxx_den = signal.welch(ddec, fs/decimateFactor, nperseg=1024)
mean_p = np.mean(Pxx_den)
peaks, _ = signal.find_peaks(Pxx_den, height=1.2*mean_p,distance=5)
#plt.plot(f[peaks], Pxx_den[peaks], "x")
plt.semilogy(f, Pxx_den)
plt.plot(mean_p, "x")
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()
#
lowcut=412-120
highcut=412+120
cutoff=5
yfiltered = butter_bandpass_filter(ddec, lowcut, highcut, fs/decimateFactor, order=6)
ysquared=yfiltered*yfiltered
ylp = butter_lowpass_filter(ysquared, cutoff, fs/decimateFactor, order=4)
plt.plot(x,ylp)
plt.show()
symbols=wpcr(ylp)
bits=slice_bits(symbols)
print(list(bits))
plt.plot(bits,'ro-')
plt.show()

morseChar={'A':[0,0,1,0,1,1,1,0,0], 'B': [0,0,1,1,1,0,1,0,1,0,1,0,0],'C':[0,0,1,1,1,0,1,0,1,1,1,0,1,0,0], 'D': [0,0,1,1,1,0,1,0,1,0,0],
           'E':[0,0,1,0,0], 'F': [0,0,1,0,1,0,1,1,1,0,1,0,0],'G':[0,0,1,1,1,0,1,1,1,0,1,0,0], 'H': [0,0,1,0,1,0,1,0,1,0,0],
           'I':[0,0,1,0,1,0,0], 'J': [0,0,1,0,1,1,1,0,1,1,1,0,1,1,1,0,0],'K':[0,0,1,1,1,0,1,0,1,1,1,0,0], 'L': [0,0,1,0,1,1,1,0,1,0,1,0,0],
           'M':[0,0,1,1,1,0,1,1,1,0,0], 'N': [0,0,1,1,1,0,1,0,0],'O':[0,0,1,1,1,0,1,1,1,0,1,1,1,0,0], 'P': [0,0,1,0,1,1,1,0,1,1,1,0,1,0,0],
           'Q':[0,0,1,1,1,0,1,1,1,0,1,0,1,1,1,0,0], 'R': [0,0,1,0,1,1,1,0,1,0,0],'S':[0,0,1,0,1,0,1,0,0], 'T': [0,0,1,1,1,0,0],
           'U':[0,0,1,0,1,0,1,1,1,0,0], 'V': [0,0,1,0,1,0,1,0,1,1,1,0,0],'W':[0,0,1,0,1,1,1,0,1,1,1,0,0], 'X': [0,0,1,1,1,0,1,0,1,0,1,1,1,0,0],
           'Y':[0,0,1,1,1,0,1,0,1,1,1,0,1,1,1,0,0], '1': [0,0,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,0],'2':[0,0,1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,0],
           '3': [0,0,1,0,1,0,1,0,1,1,1,0,1,1,1,0,0],'4':[0,0,1,0,1,0,1,0,1,0,1,1,1,0,0], '5': [0,0,1,0,1,0,1,0,1,0,1,0,0],'6':[0,0,1,1,1,0,1,0,1,0,1,0,1,0,0],
           '7': [0,0,1,1,1,0,1,1,1,0,1,0,1,0,1,0,0],'8': [0,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,0,0],'9': [0,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,0],'0': [0,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,0],
           'SP': [0,0,0,0,0],'SPW': [0,0,0,0,0,0,0]}

lb = len(bits)
decoded = [0] * lb
for chartofind, representation in morseChar.items():

    le=len(representation)
    res=0
    buckets = [0] * lb

    for b in range(lb-le):
        for i in range(le):
            if (bits[b+i]==representation[i]):
                res=res+1
        buckets[b]=res
        res=0
    maxElement = np.amax(buckets)
    print(maxElement, le)
    if(maxElement==le):
        result = np.where(buckets == np.amax(buckets))
        for x in result:
            decoded[x[0]]=chartofind
print(decoded)
X =  [x for x in decoded if not x==0]
print(X)