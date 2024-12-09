import pickle
import matplotlib.pyplot as plt

with open('FullSpectra_Group0_1.pkl','rb') as file:
    data = pickle.load(file)


Magnitude = data["Magnitude"]
Frequency = data["Frequency"]

plt.plot(Frequency, Magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()