from PySpice.Spice.Netlist import SubCircuit
import PySpice.Unit as U


# sub circuit of the pixel. often change
class Pixel(SubCircuit):
    """
    This class implement the pixel circuit model.
    """
    __nodes__ = ('t_in', 't_out')

    def __init__(self, name, shunt=None):
        SubCircuit.__init__(self, name, *self.__nodes__)
        #a dopde in parallel with a shunt resistor. the number of diodes is configured equivalently by scaling the ideality factor, not here.
        self.Diode(1, 't_out', 't_in', model='Diode_B6')
        if shunt is not None:
            self.R(1, 't_out', 't_in', shunt @U.u_Ohm)
        return
