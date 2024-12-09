import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Load data
with open('FullSpectra_Group0_1.pkl', 'rb') as file:
    data = pickle.load(file)

Phase = data["Phase"]
Frequency = data["Frequency"]

# Define a Skewed Lorentzian function
def skewed_lorentzian(f, a, f0, gamma, skew, offset):
    return a * gamma**2 / ((f - f0)**2 + gamma**2) * (1 + skew * (f - f0)) + offset

# Define the frequency range for filtering
freq_min = 5845009980.0  # Lower bound frequency
freq_max = 5870019990.0   # Upper bound frequency

# Create a mask for the frequencies within the specified range
mask = (Frequency >= freq_min) & (Frequency <= freq_max)

# Apply the mask to the Frequency and Magnitude data
filtered_Frequency = Frequency[mask]
filtered_Phase = Phase[mask]

# Initial guesses for fitting parameters
initial_guesses = [160, np.mean(filtered_Frequency), 1.5e6, 110, np.mean(filtered_Phase)]
print(initial_guesses)

# Perform curve fitting
popt, _ = curve_fit(skewed_lorentzian, filtered_Frequency, filtered_Phase, p0=initial_guesses)

# Generate fitted curve
fitted_curve = skewed_lorentzian(Frequency, *popt)

# Plot original data and fitted curve
plt.plot(Frequency, Phase, label="Original Data", color="#6CB4EE", linewidth = 3.5)
plt.plot(Frequency, fitted_curve, label="Fit", linestyle="--", color="black")
plt.xlabel("Frequency [GHz]")
plt.ylabel("Phase [rad]")
plt.xlim([freq_min, freq_max])
plt.ylim(0,210)
plt.title("Phase vs Frequency for Resonance Frequency 5.857 GHz")
plt.legend()
plt.grid(True)
plt.show()
