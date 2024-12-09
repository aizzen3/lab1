import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

with open('Flux_quantized_resonator_freq_5.806026.pkl', 'rb') as file:
    data = pickle.load(file)

# Load your data (already loaded as 'data' in your case)
bias_array = data['Bias_array']
frequency = data['Frequency']
s21 = np.abs(data['S21']).T  # Transpose to align with frequency


# Function to fit S21 using the provided equation
def s21_model(f, f_res, Q_load, Q_c, alpha, tau):

    delta_f = (f - f_res) / f_res
    Q_int = 1 / (1 / Q_load - 1 / Q_c)
    term = (Q_load / Q_c) / (1 + 2j * Q_load * delta_f)
    return np.abs(alpha * (1 - term) * np.exp(1j * 2 * np.pi * f * tau))


# Analyze each bias point
results = []
for i, bias in enumerate(bias_array):
    # Extract S21 magnitude for current bias
    s21_magnitude = s21[i, :]

    # Identify resonance frequency (min of S21 magnitude)
    min_index = np.argmin(s21_magnitude)
    f_res = frequency[min_index]

    # Initial guess for fit parameters
    Q_load_guess = 1e4  # Adjust based on typical values
    Q_c_guess = 1e5  # Adjust based on typical values
    alpha_guess = np.mean(s21_magnitude)  # Mean amplitude
    tau_guess = 0  # Approximation of cable delay

    # Fit the S21 curve
    try:
        popt, pcov = curve_fit(
            s21_model,
            frequency,
            s21_magnitude,
            p0=[f_res, Q_load_guess, Q_c_guess, alpha_guess, tau_guess],
        )
        f_res_fit, Q_load, Q_c, alpha, tau = popt
        Q_int = 1 / (1 / Q_load - 1 / Q_c)
        results.append((bias, f_res_fit, Q_load, Q_c, Q_int))
    except RuntimeError as e:
        print(f"Fit failed for bias {bias}: {e}")
        continue

    # Plot the fitted curve for validation
    fitted_curve = s21_model(frequency, *popt)
    plt.figure(figsize=(10, 6))
    plt.plot(frequency, 20 * np.log10(s21_magnitude), label="Measured Data")
    plt.plot(frequency, 20 * np.log10(fitted_curve), label="Fitted Curve")
    plt.title("Magnitude vs Frequency")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Magnitude (dB)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Display results
print("Bias (A), Resonance Frequency (Hz), Q_load, Q_c, Q_int")
for res in results:
    print(f"{res[0]:.2e}, {res[1]:.2e}, {res[2]:.2e}, {res[3]:.2e}, {res[4]:.2e}")

