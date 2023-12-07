"""
[GitHub repository](https://github.com/vyhyb/imptube)

This library provides an implementation of transfer function impedance tube measurements according to [ISO 10534-2:1998](https://www.iso.org/standard/22851.html).


It requires several Python packages to be installed:

- sounddevice
- soundfile
- scipy
- numpy
- pandas

## Installation

It is currently not possible to install this library using `pip` or `conda`, please use the latest [released package](https://github.com/vyhyb/imptube/releases) instead and install using [`pip` locally](https://packaging.python.org/en/latest/tutorials/installing-packages/).

## Usage

The library currently provides two ways to perform a measurement.

### Method 1: Direct measurement without saving any data

This method shows the inner logic of the whole measurement. In contrast to the second method, this method does not work with any folder/file structure.

```python
import imptube as imp

temp = 29  # Temperature in degrees Celsius
humidity = 30  # Relative humidity in percent
atm_pressure = 101300  # Atmospheric pressure in Pascal

tube = imp.Tube(
    further_mic_dist=0.4,
    closer_mic_dist=0.1,
    freq_limit=1000
)

measurement = imp.Measurement(device=15)  # Create an instance of the Measurement class

# Perform the configuration 1 measurement
data, fs = measurement.measure(export=False, thd_filter=True)  
p11, p12 = imp.stereo_to_spectra(data.T)
freqs = imp.frequencies(p11, fs)

input("Ready to measure in the second configuration?") 

# Perform the configuration 2 measurement
data, fs = measurement.measure(export=False, thd_filter=True)
p21, p22 = imp.stereo_to_spectra(data.T)

# Calculate the calibration factor based on the spectra
cf = imp.calibration_factor(p11, p12, p21, p22)
# Calculate the transfer function based on the spectra
# of the first configuration
tf = imp.transfer_function(p11, p12)  

# Correct the transfer function using the calibration factor
tf_corrected = tf / cf  
tf_I, tf_R = imp.tf_i_r(temp, freqs, tube.mic_spacing)

# Calculate the reflection factor, absorption coefficient and 
# surface impedance based on the transfer function and other parameters
refl_factor = imp.reflection_factor(tf_I, tf_R, tf_corrected, temp, freqs, tube.closer_mic_dist)  

absorption_coeff = imp.absorption_coefficient(refl_factor)

surf_impedance = imp.surface_impedance(refl_factor, temp, atm_pressure)
```

### Method 2: Using the `Sample` class and autosaving the captured and processed data to a specific folder structure

```python
import imptube as imp

tube = imp.Tube(
    further_mic_dist=0.3,
    closer_mic_dist=0.1,
    freq_limit=1000)

measurement = imp.Measurement(device=15)

sample = imp.Sample("test1",
    temperature=29,
    rel_humidity=30,
    tube=tube)

imp.calibration(
    sample=sample,
    measurement=measurement
)

imp.single_measurement(
    sample=sample,
    measurement=measurement,
    depth=160
)
```
## Acknowledgments

This library was created thanks to the [FAST-J-22-7880](https://www.vut.cz/vav/projekty/detail/33840) project.

Special thanks to prof. Christ Glorieux for his help with the theory.

Github Copilot was used to generate parts of the documentation and code.

## Author

- [David Jun](https://www.fce.vutbr.cz/o-fakulte/lide/david-jun-12801/)
  
  PhD student at [Brno University of Technology](https://www.vutbr.cz/en/)

## Contributing

Pull requests are welcome. For any changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)

"""
from .tube import (
    Tube, 
    Measurement, 
    Sample, 
    single_measurement, 
    calculate_alpha, 
    calibration
)

from .processing.signal_proc import (
    read_audio, 
    separate_mono, 
    stereo_to_spectra, 
    calibration_factor, 
    transfer_function, 
    tf_i_r, 
    frequencies, 
    reflection_factor, 
    absorption_coefficient, 
    surface_impedance
)

from .processing.files import read_file