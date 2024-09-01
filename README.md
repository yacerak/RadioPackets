# 16-QAM Communication System Simulation in Python

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Parameters](#parameters)
- [Output](#output)
- [License](#license)
- [Contact](#contact)

## Overview

This project implements a comprehensive simulation of a 16-Quadrature Amplitude Modulation (16-QAM) communication system using Python. The simulation encompasses the entire communication chain, including data generation, modulation, channel effects, demodulation, and Bit Error Rate (BER) analysis. It is designed to help users understand the intricacies of digital communication systems and evaluate system performance under various conditions.

## Features

- **Data Generation**: Creates random binary data with synchronization symbols.
- **16-QAM Mapping**: Maps binary data to 16-QAM constellation points.
- **Upsampling and Shaping**: Applies upsampling and Root Raised Cosine (RRC) filtering for signal shaping.
- **Carrier Generation and Mixing**: Generates carrier signals and mixes them with the baseband signal to produce a modulated signal.
- **Channel Simulation**: Models various channel impairments, including noise, frequency and phase offsets, phase noise, attenuation, delay, Doppler shifts, and multipath fading.
- **Phase-Locked Loop (PLL)**: Implements a PLL for carrier recovery.
- **Filtering and Matched Filtering**: Applies low-pass filtering and matched filtering to the received signal.
- **Downsampling and Demapping**: Downsamples the filtered signal and demaps the symbols back to binary data.
- **BER Calculation**: Computes the Bit Error Rate (BER) for different Signal-to-Noise Ratio (SNR) values.
- **Visualization**: Utilizes matplotlib for plotting signals and filter responses (though not explicitly shown in the provided code).

## Requirements

Ensure that the following Python packages are installed:

- Python 3.6 or higher
- [NumPy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)
- [Commpy](https://github.com/veeresht/commpy)
- [Matplotlib](https://matplotlib.org/)

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/16QAM-Simulation.git
   cd 16QAM-Simulation
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install numpy scipy commpy matplotlib
   ```

   *Note: `commpy` may require additional steps for installation. Refer to the [Commpy GitHub repository](https://github.com/veeresht/commpy) for detailed instructions.*

## Usage

1. **Run the Simulation**

   Execute the Python script containing the simulation code:

   ```bash
   python simulation.py
   ```

   *Ensure that the script is named appropriately (e.g., `simulation.py`) and located in the current directory.*

2. **Review Output**

   The script will print the Bit Error Rate (BER) for various SNR values ranging from 30 dB to -30 dB in decrements of 5 dB. Example output:

   ```
   the value of BER for snr=30: 0.0
   the value of BER for snr=25: 0.0
   ...
   the value of BER for snr=-30: 1.0
   ```

3. **Visualization (Optional)**

   The script imports `matplotlib.pyplot` as `plt`, allowing you to add plots for signal constellations, filter responses, or BER vs. SNR curves. You can modify the script to include visualizations as needed.

## Code Structure

The simulation script comprises the following key components and functions:

1. **Imports**

   ```python
   import numpy as np
   from scipy.signal import lfilter, butter, freqz
   import math
   from math import pi
   import commpy as cp
   import matplotlib.pyplot as plt
   ```

2. **Parameter Definitions**

   Defines various modulation and channel parameters, such as carrier frequency, sample rate, QAM order, filter parameters, frequency and phase offsets, phase noise, attenuation, delay, propagation distance, and Doppler frequency.

3. **16-QAM Constellation**

   ```python
   QAM16 = np.array([1+1j, 1+3j, 3+1j, 3+3j, 1-1j, 1-3j, 3-1j, 3-3j,
                     -1+1j, -1+3j, -3+1j, -3+3j, -1-1j, -1-3j, -3-1j, -3-3j])
   ```

4. **Function Definitions**

   - `data_gen`: Generates a packet of binary data with optional synchronization symbols.
   - `slicer`: Splits the data into In-phase (I) and Quadrature (Q) components.
   - `mapper_16QAM`: Maps binary data to 16-QAM symbols.
   - `upsampler`: Upsamples the symbol sequence.
   - `shaping_filter`: Applies an RRC filter to shape the signal.
   - `oscillator`: Generates carrier signals.
   - `mixer`: Multiplies the signal with the carrier.
   - `combiner`: Combines I and Q signals into a complex modulated signal.
   - `channel`: Simulates channel effects, including noise, offsets, phase noise, attenuation, delay, Doppler shifts, and multipath fading.
   - `PLL`: Implements a Phase-Locked Loop for carrier recovery.
   - `LPF`: Applies a low-pass filter to the signal.
   - `matched_filter`: Applies a matched filter to the signal.
   - `downsampler`: Downsamples the signal post-filtering.
   - `demapper`: Converts symbols back to binary data.
   - `snr`: Generates a list of SNR values for BER computation.

5. **Main Simulation Flow**

   - Generates and maps data.
   - Upsamples and filters the signal.
   - Generates carrier signals and performs mixing.
   - Combines I and Q to form the modulated signal.
   - Iterates over various SNR values to simulate channel conditions.
   - Applies channel effects and performs carrier recovery using PLL.
   - Filters and matches the received signal.
   - Downsamples and demaps the signal to recover binary data.
   - Calculates and prints the BER for each SNR value.

## Parameters

The simulation uses the following key parameters, which can be adjusted to study their impact on system performance:

- **Modulation Parameters**
  - `f_carrier`: Carrier frequency in Hz (default: 100,000 Hz)
  - `fs`: Sample rate in Hz (default: 5,000 Hz)
  - `Ns`: Number of symbols (default: 125)
  - `N`: Number of bits (default: 500)
  - `qam_order`: QAM order (default: 16)
  - `alpha`: RRC filter roll-off factor (default: 0.5)
  - `Fif`: Intermediate frequency (default: 10,000 Hz)

- **Channel Parameters**
  - `f_offset`: Frequency offset in Hz (default: -61,279.99999999999 Hz)
  - `phi_offset`: Phase offset in radians (default: 0.539 rad)
  - `phase_noise_std`: Standard deviation of phase noise in radians (default: 0.1 rad)
  - `attenuation_factor`: Attenuation factor (default: 158.5)
  - `delay`: Signal delay in seconds (default: 1.4e-6 s)
  - `max_propagation_distance`: Maximum propagation distance in meters (default: 665,415.51 m)
  - `f_doppler`: Doppler frequency in Hz (default: 62,200 Hz)

- **PLL Parameters**
  - `zeta`: Damping factor (default: 0.707)
  - `Bn`: Noise bandwidth (default: 1% of sample rate)
  - Gains: `K_0`, `K_d`, `K_p`, `K_i` based on PLL design

- **Filter Parameters**
  - `o`: Order of the low-pass filter (default: 5)
  - `fc`: Cutoff frequency for LPF (defined as `(fs / 2) - 100` Hz)

- **Demapping Parameters**
  - `threshold`: Threshold for symbol decision (default: 3.0)

## Output

The simulation prints the BER for each SNR value in the range of 30 dB to -30 dB in steps of -5 dB. Example:

```
the value of BER for snr=30: 0.0
the value of BER for snr=25: 0.0
the value of BER for snr=20: 0.0
...
the value of BER for snr=-25: 1.0
the value of BER for snr=-30: 1.0
```

A BER curve can be plotted by capturing these outputs and visualizing BER against SNR, providing insights into system performance under varying noise conditions.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions or suggestions, please contact [yacermeftah@gmail.com](mailto:yacermeftah@gmail.com).

# Additional Notes

Given that the code does not have a main function or any separation into modules, the README should be clear that it's a script to be run directly.

Potential Improvements:

- Adding visualization of constellation diagrams before and after the channel.
- Plotting BER vs. SNR curves.
- Adding command-line arguments to adjust parameters dynamically.
- Structuring the code into modules or classes for better maintainability.

However, as per the user request, only a README explaining the provided code is needed.
