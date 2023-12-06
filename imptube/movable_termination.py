"""
A module containing the class representing the movable termination wall
operated by a stepper motor.
"""

from typing import Protocol

class Stepper(Protocol):
    """A protocol representing a stepper motor.

    This protocol defines the methods required to control a stepper motor.

    Parameters
    ----------
    Protocol : type
        The base protocol type.

    Methods
    -------
    on()
        Turns on the stepper motor.
    enable()
        Enables the stepper motor.
    disable()
        Disables the stepper motor.
    turn(revolutions: float, clockwise: bool) -> None
        Turns the stepper motor by the specified number of revolutions in the specified direction.

    """
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
        The current position of the wall.
    stepper : Stepper
        The stepper object enabling wall movements.
    mm_per_cycle : float, optional
        The rotation to translation ratio. Default is 0.7832.
    """

    def __init__(self, position: float, stepper: Stepper, mm_per_cycle: float = 0.7832):
        self.position = position
        self.mm_per_cycle = mm_per_cycle
        self.stepper = stepper

    def adjust_wall(self, position_final) -> None:
        """
        Adjusts the wall position to the specified final position.

        Parameters
        ----------
        position_final : float
            The desired final position of the wall.

        Returns
        -------
        None
        """
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
