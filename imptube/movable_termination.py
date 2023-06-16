from typing import Protocol

class Stepper(Protocol):
    def on():
        ...

    def enable():
        ...

    def disable():
        ...
    
    def turn(self, revolutions : float, clockwise : bool) -> None:
        ...

class TerminationWall:
    """
    A class representing the movable termination wall
    operated by a stepper motor.

    Attributes
    ----------
    position : float
        wall position
    stepper : Stepper
        stepper object enabling for wall movements
    mm_per_cycle : float
        rotation to translation ratio
    """
    def __init__(self,
            position : float,
            stepper : Stepper,
            mm_per_cycle : float=0.7832
            ):
        self.position = position
        self.mm_per_cycle = mm_per_cycle
        self.stepper = stepper

    def adjust_wall(self,
            position_final,
            ) -> None:
        delta = self.position - position_final
        cycles = delta / self.mm_per_cycle
        self.stepper.on()
        self.stepper.enable()
        if cycles > 0:
            self.stepper.turn(abs(cycles), True)
        else:
            self.stepper.turn(abs(cycles), False)
        self.stepper.disable()
        self.position = position_final