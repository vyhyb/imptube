""" An example implementation of the Stepper class for movable termination.
In this case, implemented for the original impedance tube operated solely
by Raspberry Pi 4 Model B. Because of some sound stability issues,
newer implementation relies on FT232H breakout board from Adafruit.

This module provides a PiStepper class that contains methods for controlling
a stepper motor using a Raspberry Pi and a DRV8825 driver. The class allows
for setting the microstepping resolution, defining the ramping of the motor
speed, initiating GPIO settings, enabling/disabling the motor, and performing
the actual spinning of the motor.

Constants:
- DIR: Direction GPIO Pin
- STEP: Step GPIO Pin
- MODE: Microstep Resolution GPIO Pins
- ENABLE: Enable GPIO Pin
- CW: Clockwise Rotation
- CCW: Counterclockwise Rotation
- SPR: Steps per Revolution
- FREQ: Frequency
- RESOLUTION: Dictionary mapping microstepping resolution names to GPIO pin values
- RESOLUTION_M: Dictionary mapping microstepping resolution names to step multipliers

"""
from time import sleep
# import RPi.GPIO as GPIO
import numpy as np

DIR = 20   # Direction GPIO Pin
STEP = 21 # Step GPIO Pin
MODE = (14, 15, 18)   # Microstep Resolution GPIO Pins
ENABLE = 16

CW = 1     # Clockwise Rotation
CCW = 0    # Counterclockwise Rotation
SPR = 200   # Steps per Revolution
FREQ = 2
RESOLUTION = {'Full': (0, 0, 0),
              'Half': (1, 0, 0),
              '1/4': (0, 1, 0),
              '1/8': (1, 1, 0),
              '1/16': (0, 0, 1),
              '1/32': (1, 0, 1)}
RESOLUTION_M = {'Full': 1,
                'Half': 2,
                '1/4': 4,
                '1/8': 8,
                '1/16': 16,
                '1/32': 32}

class PiStepper:
    """
    Contains methods for stepper motor operation using Pi and DRV8825 driver
    
    Attributes:
    ----------
    res : str
        The microstepping resolution of the stepper motor.
    delay : numpy.ndarray
        The delay values for each step in the stepper motor movement.
    
    Methods:
    --------
    __init__(res="Half"):
        Initializes a PiStepper object with the specified microstepping resolution.
        
        Parameters:
        ----------
        res : str, optional
            The microstepping resolution of the stepper motor. Default is "Half".
    
    set_delay(step_count, fade=1, sin_begin=4):
        Sets the delay values for each step in the stepper motor movement.
        
        Parameters:
        ----------
        step_count : int
            The total number of steps in the stepper motor movement.
        fade : float, optional
            The duration of the ramping effect at the beginning and end of the movement. Default is 1.
        sin_begin : int, optional
            The number of steps over which the ramping effect is applied. Default is 4.
    
    on():
        Initializes the GPIO settings for the stepper motor operation.
    
    off():
        Cleans up the GPIO settings after the stepper motor operation.
    
    enable():
        Enables the stepper motor by operating the ENABLE pin on the driver.
    
    disable():
        Disables the stepper motor by operating the ENABLE pin on the driver.
    
    turn(revolutions=1, clockwise=True):
        Performs the actual spinning of the stepper motor.
        
        Parameters:
        ----------
        revolutions : float, optional
            The number of full revolutions to be performed by the stepper motor. Default is 1.
        clockwise : bool, optional
            Specifies the direction of rotation. True for clockwise, False for counterclockwise. Default is True.
    """

    def __init__(self, res="Half"):
        """
        Initializes a PiStepper object with the specified microstepping resolution.
        
        Parameters:
        ----------
        res : str, optional
            The microstepping resolution of the stepper motor. Default is "Half".
        """
        self.res = res

    def set_delay(self, step_count, fade=1, sin_begin=4):
        """
        Sets the delay values for each step in the stepper motor movement.
        
        Parameters:
        ----------
        step_count : int
            The total number of steps in the stepper motor movement.
        fade : float, optional
            The duration of the ramping effect at the beginning and end of the movement. Default is 1.
        sin_begin : int, optional
            The number of steps over which the ramping effect is applied. Default is 4.
        """
        self.delay = np.full(
            step_count,
            1 / FREQ / SPR / RESOLUTION_M[self.res]
            )
        len_fade = int(fade * SPR)
        ramp_in = 1/np.sin(np.linspace(
            np.pi/sin_begin,
            np.pi/2,
            len_fade
            ))
        ramp_out = 1/np.sin(np.linspace(
            np.pi/2,
            np.pi - np.pi/sin_begin,
            len_fade
            ))
        ramp_shape = ramp_in.shape[0]
        self.delay[0:ramp_shape] = self.delay[0:ramp_shape] * ramp_in
        self.delay[-ramp_shape:] = self.delay[-ramp_shape:] * ramp_out

    def on(self):
        """
        Initializes the GPIO settings for the stepper motor operation.
        """
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(DIR, GPIO.OUT)
        GPIO.setup(STEP, GPIO.OUT)
        GPIO.setup(ENABLE, GPIO.OUT)
        GPIO.output(DIR, CW)
        GPIO.setup(MODE, GPIO.OUT)

        GPIO.output(MODE, RESOLUTION[self.res])

    def off(self):
        """
        Cleans up the GPIO settings after the stepper motor operation.
        """
        GPIO.cleanup()

    def enable(self):
        """
        Enables the stepper motor by operating the ENABLE pin on the driver.
        """
        GPIO.output(ENABLE, 0)

    def disable(self):
        """
        Disables the stepper motor by operating the ENABLE pin on the driver.
        """
        GPIO.output(ENABLE, 1)


    def turn(self, revolutions=1, clockwise=True):
        """
        Performs the actual spinning of the stepper motor.
        
        Parameters:
        ----------
        revolutions : float, optional
            The number of full revolutions to be performed by the stepper motor. Default is 1.
        clockwise : bool, optional
            Specifies the direction of rotation. True for clockwise, False for counterclockwise. Default is True.
        """
        if clockwise:
            GPIO.output(DIR, CW)
        else:
            GPIO.output(DIR, CCW)

        step_count = int(revolutions * SPR * RESOLUTION_M[self.res])
        
        self.set_delay(step_count)

        for d in self.delay:
            GPIO.output(STEP, GPIO.HIGH)
            sleep(d)
            GPIO.output(STEP, GPIO.LOW)
            sleep(d)



if __name__ == "__main__":
    p = PiStepper()
    p.on()
    p.turn(0.75)
    p.off()


    
