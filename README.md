# imptube 
This library provides an implementation of transfer function impedance tube measurements according to [ISO 10534-2:1998](https://www.iso.org/standard/22851.html).

It uses several python packages:
- sounddevice
- soundfile
- scipy
- numpy
- pandas

## Installation
It is currently not possible to install this library using pip or conda.

## Usage
The library currently provides two way to perform a measurement.

### Method 1 - direct measurement without saving any data
This method shows the inner logic of the whole measurement. In contrast to the second method, this method does not work with any folder/file structure.

```python
import imptube as imp

temp = 29  # Temperature in degrees Celsius
humidity = 30  # Humidity level in percentage
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

### Method 2 - using the `Sample` class and autosaving the captured and processed data to a specific folder structure

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

## Contributing

Pull requests are welcome. For any changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)

