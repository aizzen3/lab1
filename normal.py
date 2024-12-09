import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Load data from the pickle file
with open('Flux_quantized_resonator_freq_5.806026', 'rb') as file:
    data = pickle.load(file)

# Extract data
current_bias = data['Bias_array']
frequency = data['Frequency']
S21 = data['S21']


current_bias_microA = current_bias * 1e6

S21_dB = 20 * np.log10(np.abs(S21))

resonance_freqs = []
for i in range(len(current_bias)):

    S21_mag = np.abs(S21[:, i])

    S21_smooth = np.convolve(S21_mag, np.ones(5)/5, mode='same')
    resonance_idx = np.argmin(S21_smooth)
    resonance_freqs.append(frequency[resonance_idx])

resonance_freqs = np.array(resonance_freqs)


def cosine_func(x, amplitude, period, phase, offset):
    return amplitude * np.cos(2 * np.pi * x / period + phase) + offset

data_range = np.ptp(current_bias_microA)
amplitude_guess = np.ptp(resonance_freqs) / 2
period_guess = data_range / 2
offset_guess = np.mean(resonance_freqs)
phase_guess = 0

p0 = [amplitude_guess, period_guess, phase_guess, offset_guess]

bounds = ([0, 0, -2*np.pi, min(resonance_freqs)],
          [np.inf, np.inf, 2*np.pi, max(resonance_freqs)])

popt, pcov = curve_fit(cosine_func, current_bias_microA, resonance_freqs,
                      p0=p0, bounds=bounds, maxfev=100)

current_fit = np.linspace(min(current_bias_microA), max(current_bias_microA), 1000)
resonance_fit = cosine_func(current_fit, *popt)

X, Y = np.meshgrid(current_bias_microA, frequency)
contour = plt.contourf(X, Y, S21_dB, levels=50, cmap="viridis")
plt.colorbar(contour, label="Magnitude dB")
plt.plot(current_fit, resonance_fit, 'black', label='Cosine fit', linewidth=1.0)
plt.xlabel("Flux bias Current (μA)")
plt.ylabel("Frequency (GHz)")
plt.title("Frequency vs Current with Cosine Fit (Resonance Frequency 5.975GHz)")
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()

print(p0)