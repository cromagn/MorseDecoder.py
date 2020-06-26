from biquad_module import Biquad
from scipy.io import wavfile
from pylab import *
from math import *
import random, re


def ntrp(x, xa, xb, ya, yb):
    return (x - xa) * (yb - ya) / (xb - xa) + ya

sample_rate, data = wavfile.read('C:/tmp/morse.wav')
#sample_rate = 40000.0  # sampling frequency

cf = 1000

pll_integral = 0
old_ref = 0
pll_cf = 1000
pll_loop_gain = 0.00003
ref_sig = 0

invsqr2 = 1.0 / sqrt(2.0)

cutoff = .06  # Units Hz

loop_lowpass = Biquad(Biquad.LOWPASS, cutoff, sample_rate, invsqr2)

lock_lowpass = Biquad(Biquad.LOWPASS, cutoff, sample_rate, invsqr2)

ta = []
da = []
db = []

noise_level = 100  # +40 db

#dur = 300  # very long run time
dur= len(data)/sample_rate
print (len(data))
#for n in range(int(sample_rate) * dur):
for n in range(len(data)):
    t = n / sample_rate

    # BEGIN test signal block
    window = (0, 1)[t > dur * .25 and t < dur * .75]
    print (n)
    print (t)
    print (window)
    test_sig = data[n] * window
    print (test_sig)


