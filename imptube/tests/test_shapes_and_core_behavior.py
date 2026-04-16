import numpy as np
import pytest

from imptube.processing.filters import (
    extend_with_zeros,
    tf_filtering,
    ir_filtering,
    harmonic_distortion_filter,
)
from imptube.processing.signal_proc import (
    separate_mono,
    stereo_to_spectra,
    frequencies,
    auto_spectrum,
    cross_spectrum,
    transfer_function,
    calibration_factor,
    tf_i_r,
    reflection_factor,
    absorption_coefficient,
    surface_impedance,
)
from imptube.tube import Tube, Sample, read_env_bc


class DummyMeasurement:
    def __init__(self):
        self.sub_measurements = 1
        self.fs = 48000
        self.f_limits = [10, 1000]
        self.sweep = np.ones(256)
        self.data = np.zeros((256, 2), dtype=float)
        self.fs_to_spl = 130


def test_separate_mono_and_stereo_to_spectra_shapes():
    n = 128
    stereo = np.vstack([
        np.sin(np.linspace(0, 2 * np.pi, n, endpoint=False)),
        np.cos(np.linspace(0, 2 * np.pi, n, endpoint=False)),
    ])

    p1_time, p2_time = separate_mono(stereo)
    assert p1_time.shape == (n,)
    assert p2_time.shape == (n,)

    p1, p2 = stereo_to_spectra(stereo)
    assert p1.shape == (n,)
    assert p2.shape == (n,)
    assert np.iscomplexobj(p1)
    assert np.iscomplexobj(p2)


def test_frequency_and_transfer_related_shapes():
    n = 256
    fs = 48000
    t = np.arange(n) / fs
    s1 = np.sin(2 * np.pi * 500 * t)
    s2 = 0.7 * np.sin(2 * np.pi * 500 * t + 0.1)

    p1 = np.fft.fft(s1)
    p2 = np.fft.fft(s2)

    freqs = frequencies(p1, fs)
    assert freqs.shape == (n,)

    a11 = auto_spectrum(p1)
    c21 = cross_spectrum(p1, p2)
    h12 = transfer_function(p1, p2)
    assert a11.shape == (n,)
    assert c21.shape == (n,)
    assert h12.shape == (n,)


def test_calibration_factor_shape():
    n = 128
    rng = np.random.default_rng(123)
    p11 = rng.normal(size=n) + 1j * rng.normal(size=n)
    p12 = rng.normal(size=n) + 1j * rng.normal(size=n)
    p21 = rng.normal(size=n) + 1j * rng.normal(size=n)
    p22 = rng.normal(size=n) + 1j * rng.normal(size=n)

    cf = calibration_factor(p11, p12, p21, p22)
    assert cf.shape == (n,)
    assert np.iscomplexobj(cf)


def test_wave_propagation_related_shapes():
    freqs = np.linspace(10, 3000, 200)
    tf_i, tf_r = tf_i_r(temperature=23.0, freqs=freqs, s=0.03)
    assert tf_i.shape == freqs.shape
    assert tf_r.shape == freqs.shape

    tf12 = 0.3 * tf_i + 0.2 * tf_r
    rf = reflection_factor(tf_i, tf_r, tf12, 23.0, freqs, x1=0.08)
    alpha = absorption_coefficient(rf)
    z = surface_impedance(rf, 23.0, 101325.0)

    assert rf.shape == freqs.shape
    assert alpha.shape == freqs.shape
    assert z.shape == freqs.shape


def test_extend_with_zeros_shapes_for_1d_and_2d():
    arr_1d = np.arange(10)
    ext_1d = extend_with_zeros(arr_1d)
    assert ext_1d.shape == (20,)
    np.testing.assert_array_equal(ext_1d[:10], arr_1d)
    np.testing.assert_array_equal(ext_1d[10:], np.zeros(10, dtype=arr_1d.dtype))

    arr_2d = np.arange(12).reshape(2, 6)
    ext_2d = extend_with_zeros(arr_2d)
    assert ext_2d.shape == (2, 12)
    np.testing.assert_array_equal(ext_2d[:, :6], arr_2d)
    np.testing.assert_array_equal(ext_2d[:, 6:], np.zeros_like(arr_2d))


def test_tf_filtering_and_ir_filtering_shapes():
    n = 1024
    tf = np.ones(n, dtype=complex)
    tf_filt, win = tf_filtering(tf, lowcut=50, highcut=5000, fs=48000)

    assert tf_filt.shape == (n,)
    assert win.shape == (n,)
    assert np.all(win >= 0)

    ir = np.ones(n, dtype=complex)
    ir_filtered = ir_filtering(ir.copy())
    assert ir_filtered.shape == (n,)
    np.testing.assert_array_equal(ir_filtered[n // 2 :], np.zeros(n // 2, dtype=complex))


def test_harmonic_distortion_filter_output_shape_is_stereo_time():
    fs = 48000
    n = 2048
    t = np.arange(n) / fs

    ref = np.sin(2 * np.pi * 200 * t)
    stereo = np.vstack([
        0.5 * np.sin(2 * np.pi * 200 * t),
        0.25 * np.sin(2 * np.pi * 200 * t + 0.2),
    ])

    out = harmonic_distortion_filter(
        p_time=stereo,
        p_ref=ref,
        f_low=50,
        f_high=2000,
        fs=fs,
    )

    assert out.shape == stereo.shape
    assert np.isrealobj(out)


def test_sample_calculate_alpha_returns_expected_tuple_shapes():
    n = 300
    freqs = np.linspace(1.0, 5000.0, n)

    tube = Tube(further_mic_dist=0.10, closer_mic_dist=0.06, freq_limit=4500)
    sample = Sample(
        name="dummy",
        tube=tube,
        measurement=DummyMeasurement(),
        temperature=24.0,
        rel_humidity=50.0,
        atm_pressure=101300.0,
    )

    sample.freqs = freqs
    sample.tf = np.ones(n, dtype=complex)
    sample.tf_corrected = 0.2 * np.ones(n, dtype=complex)

    alpha, f = sample.calculate_alpha(f_limits=[100, 2000])
    assert alpha.shape == f.shape
    assert alpha.ndim == 1

    alpha2, f2, r2 = sample.calculate_alpha(return_r=True, f_limits=[100, 2000])
    assert alpha2.shape == f2.shape
    assert r2.shape == f2.shape

    alpha3, f3, r3, z3 = sample.calculate_alpha(return_r=True, return_z=True, f_limits=[100, 2000])
    assert alpha3.shape == f3.shape
    assert r3.shape == f3.shape
    assert z3.shape == f3.shape


def test_sample_calculate_alpha_raises_when_tf_missing():
    tube = Tube(further_mic_dist=0.10, closer_mic_dist=0.06, freq_limit=4500)
    sample = Sample(
        name="dummy",
        tube=tube,
        measurement=DummyMeasurement(),
        temperature=24.0,
        rel_humidity=50.0,
        atm_pressure=101300.0,
    )

    sample.tf = None
    sample.freqs = np.linspace(1.0, 5000.0, 128)
    sample.tf_corrected = np.ones(128, dtype=complex)

    with pytest.raises(ValueError, match="No transfer function found"):
        sample.calculate_alpha()


class _GoodSensor:
    def read_temperature(self):
        return 25.5

    def read_humidity(self):
        return 44.0

    def read_pressure(self):
        return 100900.0


class _FailingSensor:
    def read_temperature(self):
        raise RuntimeError("fail")

    def read_humidity(self):
        raise RuntimeError("fail")

    def read_pressure(self):
        raise RuntimeError("fail")


def test_read_env_bc_success_path():
    temp, rh, p = read_env_bc(_GoodSensor())
    assert temp == pytest.approx(25.5)
    assert rh == pytest.approx(44.0)
    assert p == pytest.approx(100900.0)


def test_read_env_bc_failure_exits():
    with pytest.raises(SystemExit):
        read_env_bc(_FailingSensor())
