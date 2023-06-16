from time import sleep
import RPi.GPIO as GPIO
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
    
    methods:
    __init__(res) - makes object with the definition of microstepping
    set_delay(step_count, fade, sin_begin) - defines ramping of the stepper motor speed
    on() - initiates GPIO settings
    off() - cleanes up the GPIO settings
    enable()/disable() - operates the ENABLE pin on driver to avoid the noise induces by holding motor in a position.
    turn(revolutions, clockwise) - performing actual spinning
    """
    def __init__(self, res="Half"):
        self.res = res

    def set_delay(self, step_count, fade=1, sin_begin=4):
        """
        docstring
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
        docstring
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
        docstring
        """
        GPIO.cleanup()

    def enable(self):
        GPIO.output(ENABLE, 0)

    def disable(self):
        GPIO.output(ENABLE, 1)


    def turn(self, revolutions=1, clockwise=True):
        """
        docstring
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


    
