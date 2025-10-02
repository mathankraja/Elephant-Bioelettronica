# Elephant-Bioelettronica
# 🧠 Real-Time EEG Viewer for OT Bioelettronica E1

This project provides a **stand-alone Python application** to acquire and visualize EEG data from the **E1 amplifier** over a COM port.  
It replicates the functionality of the MATLAB demo script and displays EEG signals in real time using a Dash web app.

---

## ✨ Features
- Opens the E1 amplifier via **serial COM port** and starts streaming EEG.
- Decodes **24-bit signed EEG samples** into microvolts.
- **Dash web interface** with:
  - Live EEG waveform (single channel, e.g., AFz).
  - Theta/Beta neurofeedback gauge with baseline toggle.
  - Band power bar chart (Delta, Theta, Alpha, Beta, Gamma).
  - PSD (power spectral density) plot.
  - Status badges for amplifier connection and data sync.
- Automatic browser launch on start.

---

## 📦 Requirements

Python 3.9+ recommended.  
Install dependencies with:

```bash
pip install pyserial numpy plotly dash scipy
