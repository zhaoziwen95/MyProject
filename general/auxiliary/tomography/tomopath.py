
class tomopath():
    def __init__(self, tx_pos, rx_pos, integral_tof, intermediate_pos = None):
        self.tx_pos = tx_pos
        self.rx_pos = rx_pos
        self.intermediate_pos = intermediate_pos
        self.integral_tof = integral_tof

    def add_intermediate_position(self,intermediate_pos):
        self.intermediate_pos = intermediate_pos

