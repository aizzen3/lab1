import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Load data
with open('FullSpectra_Group0_1.pkl', 'rb') as file:
    data = pickle.load(file)

Magnitude = data["Magnitude"]
Frequency = data["Frequency"]

# Define a Skewed Lorentzian function
def gaussian(f, a, f0, gamma, skew, offset):
    return a * gamma**2 / ((f - f0)**2 + gamma**2) * (1 + skew * (f - f0)) + offset

# Define the frequency range for filtering
freq_min = 5845009980.0  # Lower bound frequency
freq_max = 5870019990.0   # Upper bound frequency

# Create a mask for the frequencies within the specified range
mask = (Frequency >= freq_min) & (Frequency <= freq_max)

# Apply the mask to the Frequency and Magnitude data
filtered_Frequency = Frequency[mask]
filtered_Magnitude = Magnitude[mask]

# Initial guesses for fitting parameters
initial_guesses = [100, np.mean(filtered_Frequency), 0.5e6, 0, np.mean(filtered_Magnitude)]

# Perform curve fitting
popt, _ = curve_fit(gaussian, filtered_Frequency, filtered_Magnitude, p0=initial_guesses)

# Generate fitted curve
fitted_curve = gaussian(filtered_Frequency, *popt)

# Plot original data and fitted curve within the desired frequency range
plt.plot(filtered_Frequency, filtered_Magnitude, label="Original Data", color="#6CB4EE", linewidth=3.5)
plt.plot(filtered_Frequency, fitted_curve, label="Fit", linestyle="--", color="#000000")
plt.xlabel("Frequency [GHz]")
plt.xlim(freq_min, freq_max)
plt.ylabel("Magnitude [dB]")
plt.legend()
plt.title("Magnitude vs Frequency for Resonance Frequency 5.857 GHz")
plt.grid(True)
plt.show()

