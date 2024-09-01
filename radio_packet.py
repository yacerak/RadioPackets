import numpy as np
from scipy.signal import lfilter, butter, freqz
import math
from math import pi
import commpy as cp
import matplotlib.pyplot as plt


# Define modulation parameters
f_carrier = 100000  # carrier frequency in Hz
fs = 5000  # sample rate in Hz
Ns = 125  # number of symbols
N= 500  # number of bits
qam_order = 16  # QAM order
alpha = 0.5  # RRC filter alpha value
Fif = 10000  # Intermediate frequency
# Define the frequency and phase offsets
f_offset = -6.127999999999999e+04  # in Hz
phi_offset = 0.539  # in radians
# Define the standard deviation of the phase noise
phase_noise_std = 0.1  # in radians
# Define the attenuation factor and the delay
attenuation_factor = 158.5
delay = 1.4e-6  # in seconds
# Define the maximum one-way propagation distance and the carrier and Doppler frequencies
max_propagation_distance = 6.654155100000000e+05  # in meters
f_doppler = 62200  # in Hz
# Define the QAM16 array with amplitudes and phases
QAM16 = np.array([1+1j, 1+3j, 3+1j, 3+3j, 1-1j, 1-3j, 3-1j, 3-3j,
                  -1+1j, -1+3j, -3+1j, -3+3j, -1-1j, -1-3j, -3-1j, -3-3j])


def data_gen(N, data_sync=0):
   if data_sync == 0:
      data_sync_osc = []
      for i in range(176):
            data_sync_osc.append(1)
      data_sync_symb = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
      data_sync = np.concatenate((data_sync_osc, data_sync_symb), axis=None)
   data_r = np.random.rand(N - len(data_sync))
   data_r[np.where(data_r >= 0.5)] = 1
   data_r[np.where(data_r < 0.5)] = 0
   data = np.concatenate((data_sync, data_r), axis=None)
   return(data)

def slicer(data):
   dataI = data[slice(0, len(data), 2)]
   dataQ = data[slice(1, len(data), 2)]
   return(dataI, dataQ)

def mapper_16QAM(QAM16, data):
   map0 = 2*data[slice(0, len(data), 2)] + data[slice(1, len(data), 2)]
   map0 = list(map(int, map0))
   dataMapped = []
   for i in range(len(map0)):
      dataMapped.append(QAM16[map0[i]])
   return(dataMapped)

def upsampler(Ns, K, symbols):
   up = np.zeros(Ns*K)
   up[slice(0, len(up), K)] = symbols
   return(up)

def shaping_filter(upsampler, Ns, alpha, Fif, Fs):
   [x_axis, y_response] = cp.rrcosfilter(Ns, alpha, 2/Fif, Fs)
   shaped_signal = np.convolve(upsampler, y_response, 'full')
   return(shaped_signal, x_axis, y_response)

def oscillator(start, stop, step, frequency, phase):
   t = np.arange(start, stop, step)
   print("t:",t)
   Osc = np.sin(2*pi*frequency*t + phase)
   return(Osc, t)

def mixer(signal, carrier):
   mix = []
   for i in range(len(signal)):
      mix.append(signal[i]*carrier[i])
   return(mix)

def combiner(signal_I, signal_Q):
   combined_sig = []
   for i in range(len(signal_I)):
      combined_sig.append(signal_I[i] + signal_Q[i])
   return(combined_sig)

def channel(modulated_signal, SNR, f_offset, phi_offset, phase_noise_std, attenuation_factor, delay, f_doppler,mpd):
    # Calculate the signal power
    signal_power = np.mean(np.abs(modulated_signal) ** 2)

    # Calculate the noise power from the SNR
    noise_power = signal_power / (10 ** (SNR / 10))

    # Generate the noise
    noise = np.sqrt(noise_power / 2) * np.random.randn(len(modulated_signal)) + 1j * np.sqrt(
        noise_power / 2) * np.random.randn(len(modulated_signal))

    # Add the noise to the modulated signal
    noisy_signal = modulated_signal + noise

    # Define the sampling frequency and the time vector

    t = np.arange(len(noisy_signal)) / fs

    # Generate the frequency and phase offset signals
    freq_offset_signal = np.exp(1j * 2 * np.pi * f_offset * t)
    phase_offset_signal = np.exp(1j * phi_offset)

    # Apply the frequency and phase offsets to the noisy signal
    offset_signal = noisy_signal * freq_offset_signal * phase_offset_signal

    # Generate the phase noise signal
    phase_noise = phase_noise_std * np.random.randn(len(offset_signal))

    # Apply the phase noise to the offset signal
    noisy_offset_signal = offset_signal * np.exp(1j * phase_noise)

    # Compute the filter coefficients for the delay and attenuation
    taps = int(delay * fs)
    attenuated_signal = noisy_offset_signal * attenuation_factor
    delay_filter = np.concatenate((np.zeros(taps), [1]))

    # apply delay filter to signal
    delayed_signal = lfilter(delay_filter, 1, attenuated_signal)

    # Calculate the Doppler shift factor
    c = 299792458  # speed of light in m/s
    doppler_factor = np.exp(-1j * 2 * np.pi * f_doppler * mpd / c)

    # Apply the Doppler shift to the attenuated and delayed signal
    doppler_signal = delayed_signal * doppler_factor

    # Generate the carrier signal
    t = np.arange(len(doppler_signal)) / fs
    carrier_signal = np.exp(1j * 2 * np.pi * f_carrier * t)

    # Multiply the Doppler signal with the carrier signal
    doppler_modulated_signal = doppler_signal * carrier_signal

    # Define the fading channel coefficients and delays
    channel_coeffs = np.array([0.7, 0.5j, -0.3, -0.2j])
    channel_delays = np.array([0, 5, 10, 15]) / 1e6  # in seconds

    # Generate the multipath fading channel

    t = np.arange(len(doppler_modulated_signal)) / fs
    fading_channel = np.zeros_like(t, dtype=np.complex128)
    for coeff, delay in zip(channel_coeffs, channel_delays):
        fading_channel += coeff * np.exp(1j * 2 * np.pi * f_carrier * (t - delay))

    # Apply the multipath fading effect to the Doppler modulated signal
    signal_over_channel = doppler_modulated_signal * fading_channel
    return signal_over_channel

def PLL(input_signal, Fs, lenght, N):
   zeta = .707  # damping factor
   k = 1
   Bn = 0.01*Fs  #Noise Bandwidth
   K_0 = 1  # NCO gain
   K_d = 1/2  # Phase Detector gain
   K_p = (1/(K_d*K_0))*((4*zeta)/(zeta+(1/(4*zeta)))) * \
      (Bn/Fs)  # Proporcional gain
   K_i = (1/(K_d*K_0))*(4/(zeta+(1/(4*zeta)**2))) * \
      (Bn/Fs)**2  # Integrator gain
   integrator_out = 0
   phase_estimate = np.zeros(lenght)
   e_D = []  # phase-error output
   e_F = []  # loop filter output
   sin_out_n = np.zeros(lenght)
   cos_out_n = np.ones(lenght)
   for n in range(lenght-1):
      # phase detector
      try:
            e_D.append(
               math.atan(input_signal[n] * (cos_out_n[n] + sin_out_n[n])))
      except IndexError:
            e_D.append(0)
      # loop filter
      integrator_out += K_i * e_D[n]
      e_F.append(K_p * e_D[n] + integrator_out)
      # NCO
      try:
            phase_estimate[n+1] = phase_estimate[n] + K_0 * e_F[n]
      except IndexError:
            phase_estimate[n+1] = K_0 * e_F[n]
      sin_out_n[n+1] = -np.sin(2*np.pi*(k/N)*(n+1) + phase_estimate[n])
      cos_out_n[n+1] = np.cos(2*np.pi*(k/N)*(n+1) + phase_estimate[n])

   sin_out_n = -sin_out_n
   cos_out = cos_out_n[280:400]
   sin_out = sin_out_n[280:400]

   for i in range(18):
      cos_out = np.concatenate(
            (cos_out, cos_out_n[280:400], cos_out_n[280:400]), axis=None)
      sin_out = np.concatenate(
            (sin_out, sin_out_n[280:400], sin_out_n[280:400]), axis=None)
   return(cos_out, sin_out)

def LPF(signal, fc, Fs):
   o = 5  # order of the filter
   fc = np.array([fc])
   wn = 2*fc/Fs

   [b, a] = butter(o, wn, btype='lowpass')
   [W, h] = freqz(b, a, worN=1024)

   W = Fs*W/(2*pi)

   signal_filt = lfilter(b, a, signal)
   return(signal_filt, W, h)

def matched_filter(signal, template):
   signal_filt = np.convolve(signal, template, 'full')
   return(signal_filt)


def downsampler(signal, packet_s, upsampler_f):
   e = 0
   gardner_e = []
   peak_sample = 0
   peak_sample_acc = []
   low_point = 0
   threshold = 4
   for i in range(len(signal)):
      if signal[low_point] < -threshold:
            if signal[i] > threshold:
               e = (abs(signal[(i+1)]) -
                     abs(signal[i-1])) * abs(signal[i])
               gardner_e.append(e)
               if e > 0.8:
                  peak_sample = peak_sample + 1
                  peak_sample_acc.append(peak_sample)
               elif e < -0.8:
                  peak_sample = peak_sample - 1
                  peak_sample_acc.append(peak_sample)
               else:
                  break
            else:
               peak_sample = peak_sample + 1
               peak_sample_acc.append(peak_sample)
      else:
            low_point = low_point + 1
            peak_sample = peak_sample + 1
            peak_sample_acc.append(peak_sample)

   # 450 is the number of samples before the convergence symbol of the algorithm.
   cut_i = peak_sample - 450
   cut_f = cut_i + int((packet_s/4)*upsampler_f)
   print("Cut_i = ", cut_i)
   print("Cut_f = ", cut_f)

   # For the code to still work, even when there is a big BER, this secction is required.
   if cut_i > 730:
      signal = signal[261:2306+510]
   elif cut_i < 690:
      signal = signal[261:2306+510]
   else:
      signal = signal[cut_i:cut_f]

   symbols = signal[slice(0, len(signal), upsampler_f)]
   return(symbols)

def demapper(symbols_I, symbols_Q, packetSize, threshold = 3.0):
   Ns = int(packetSize/4)
   bits_I = []
   bits_Q = []
   for i in range(Ns):
      if symbols_I[i] >= 0 and symbols_I[i] <= threshold:
            bits_I.append(1)
            bits_I.append(0)

      if symbols_I[i] > threshold:
            bits_I.append(1)
            bits_I.append(1)

      if symbols_I[i] < 0 and symbols_I[i] >= -threshold:
            bits_I.append(0)
            bits_I.append(1)

      if symbols_I[i] < -threshold:
            bits_I.append(0)
            bits_I.append(0)

      if symbols_Q[i] >= 0 and symbols_Q[i] <= threshold:
            bits_Q.append(1)
            bits_Q.append(0)

      if symbols_Q[i] > threshold:
            bits_Q.append(1)
            bits_Q.append(1)

      if symbols_Q[i] < 0 and symbols_Q[i] >= -threshold:
            bits_Q.append(0)
            bits_Q.append(1)

      if symbols_Q[i] < -threshold:
            bits_Q.append(0)
            bits_Q.append(0)

   bits_I = list(map(int, bits_I))
   bits_Q = list(map(int, bits_Q))

   bitStream = np.zeros(packetSize)

   for i in range(len(bits_I)):
      bitStream[2*i] = bits_I[i]
      bitStream[2*i-1] = bits_Q[i-1]
   return(bitStream)

def snr():
    snr_list=[]
    snr = 30
    while snr >= -30:
        snr_list.append(snr)
        snr-=5
    return snr_list


# Generate packet of data
packet = data_gen(N)

# Separate real and imaginary components
data_I, data_Q = slicer(packet)

# Map binary data onto 16-QAM constellation
data_I_mapped = mapper_16QAM(QAM16, data_I)
data_Q_mapped = mapper_16QAM(QAM16, data_Q)

upI = upsampler(Ns, 4, data_I_mapped)
upQ= upsampler(Ns, 4, data_Q_mapped)

filtredI, x, template= shaping_filter(upI,Ns, alpha, Fif, fs)
filtredQ, _, _ = shaping_filter(upQ,Ns, alpha, Fif, fs)
cos_carrier, t=oscillator(0,len(filtredI)/fs,1/fs,f_carrier, 0)
sin_carrier, _=oscillator(0,len(filtredQ)/fs,1/fs,f_carrier,-pi/2)

mixedI = mixer(filtredI, cos_carrier)
mixedQ = mixer(filtredQ, sin_carrier)

modulated_signal = combiner(mixedI, mixedQ)
for SNR in snr():
    signal_channel = channel(modulated_signal, SNR, f_offset, phi_offset, phase_noise_std, attenuation_factor, delay,
                             f_doppler, max_propagation_distance)

    cos_out, sin_out = PLL(signal_channel, fs, len(mixedI), 4)
    mixedI_out = mixer(signal_channel, cos_out)
    mixedQ_out = mixer(signal_channel, sin_out)

    fc = (fs / 2) - 100
    LPFI_out, _, Yout = LPF(mixedI_out, fc, fs)
    LPFQ_out, _, _ = LPF(mixedQ_out, fc, fs)

    filtered_I = matched_filter(LPFI_out, template)
    filtered_Q = matched_filter(LPFQ_out, template)

    downI = downsampler(filtered_I, N, 4)
    downQ = downsampler(filtered_Q, N, 4)

    demapped = demapper(filtered_I, filtered_Q, N)

    # Define the transmitted and received signals
    tx_signal = packet
    rx_signal = demapped

    # Convert the transmitted and received signals to bits
    tx_bits = np.where(tx_signal.real > 0, 1, 0)
    tx_bits = np.concatenate((tx_bits, np.where(tx_signal.imag > 0, 1, 0)))
    rx_bits = np.where(rx_signal.real > 0, 1, 0)
    rx_bits = np.concatenate((rx_bits, np.where(rx_signal.imag > 0, 1, 0)))

    # Compute the number of bit errors
    num_errors = np.sum(tx_bits != rx_bits)

    # Calculate the bit error rate (BER)
    ber = num_errors / len(tx_bits)
    print(f"the value of BER for snr={SNR}: {ber}")

