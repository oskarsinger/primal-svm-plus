class FixedScheduler:

    def __init__(self, stepsize):

        self.stepsize = stepsize

    def get_stepsize(self):

        return self.stepsize

class InversePowerScheduler:

    def __init__(self, initial=0.1, power=0.5):

        self.initial = initial
        self.num_rounds = 0
        self.power = power

    def get_stepsize(self):

        self.num_rounds += 1
        denom = float(self.num_rounds)**(-self.power)

        return denom * self.initial

    def refresh(self):

        self.num_rounds = 0

    def get_status(self):

        return {
            'initial': self.initial,
            'num_rounds': self.num_rounds}
