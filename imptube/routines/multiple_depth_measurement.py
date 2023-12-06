"""
!!! warning. Needs to be reimplemented !!!
"""
from imptube import Measurement
from imptube.pistepper import PiStepper
import os
import numpy as np
from time import sleep

def multiple_depth_measurement(
        self,
        measurement : Measurement,
        depth_init : float,
        depth_end : float,
        step : int=10,
        resolution : str="1/4",
        thd_filter : bool = True):
    """Perform multiple depth measurements.

    This function performs multiple depth measurements within a specified range.
    It uses the provided `Measurement` object to measure the depth at each step.

    Parameters
    ----------
    measurement : Measurement
        The `Measurement` object used for impedance tube measurement.
    depth_init : float
        The initial depth value in mm for the measurement range.
    depth_end : float
        The final depth value in mm for the measurement range.
    step : int, optional
        The step size between each depth measurement in mm, by default 10.
    resolution : str, optional
        The resolution of the stepper, by default "1/4".
    thd_filter : bool, optional
        Flag indicating whether to apply THD (Total Harmonic Distortion) filtering, by default True.
    """

    m = measurement

    if depth_end is depth_init:
        print("You have to choose two different limits for measured depth.")
        return

    p = PiStepper(res=resolution)
    if depth_init is not p.position:
        pre_cycles = (p.position - depth_init) / p.wall_mm_per_cycle
        if pre_cycles < 0:
            pre_direction = False
        else:
            pre_direction = True
        p.on()
        p.enable()
        p.turn(abs(pre_cycles), pre_direction)
        p.disable()
        p.position = depth_init

    cycles = abs(step / p.wall_mm_per_cycle)
    if depth_init > depth_end:
        step = -step

    if step < 0:
        direction = True
    else:
        direction = False
    
    depth = np.arange(
        depth_init, 
        depth_end+step, 
        step).astype(int)
    

    for d in depth:
        running = True
        while running:
            print(d, depth[-1], d == depth[-1])
            for s in range(m.sub_measurements):
                f = os.path.join(m.trees[4][0], m.trees[1]+f"_wav_d{d}_{s}.wav")
                m.measure(f, thd_filter=thd_filter)
                sleep(1)
            if input(f"Repeat measurement for depth {d} mm? [y/N]").lower() == "y":
                continue
            else:
                running = False    
        if d is not depth[-1]:
            p.on()
            p.enable()
            p.turn(abs(cycles), direction)
            p.disable()