from PySpice.Spice.Netlist import SubCircuit
import PySpice.Unit as U


# sub circuit of the interface. often change
class SIROF(SubCircuit):
    """
    This class is the electrode-electrolyte sub-circuit.
    """
    __nodes__ = ('t_in', 't_out')

    def __init__(self, name, scaling=1, Vini=0, c0=20, Rdc=100):
        SubCircuit.__init__(self, name, *self.__nodes__, "PARAMS: Vini=0")
        #scaling accounts for the effects of area scaling on both R and C
        #Vini is the initial voltage across the caps
        self.C('f', 't_in', 't_out', (c0 * scaling) @ U.u_nF, raw_spice="IC={Vini}")
        #very large faradaic resistance by default
        self.R('dc', 't_in', 't_out', Rdc/scaling @ U.u_GOhm)
        return
