

class Schedule(object):
    def name(self):
        """return string description of schedule"""
        raise NotImplementedError()

    def next_value(self):
        """moves schedule one ahead, returns Value of the schedule right now"""
        raise NotImplementedError()

    def get_value(self):
        """Value of the schedule right now"""
        raise NotImplementedError()


class ConstantSchedule(Schedule):
    def __init__(self, value):
        """
        Value remains constant over time.
        Parameters
        ----------
        value: float
            Constant value of the schedule
        """
        self.is_epsilon_greedy = True
        self._v = value

    def name(self):
        return "Constant_val={}".format(self._v)

    def next_value(self):
        """See Schedule.value"""
        return self._v

    def get_value(self):
        """See Schedule.value"""
        return self._v


class LinearSchedule(Schedule):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.is_epsilon_greedy = True
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p
        self.t = 0

    def name(self):
        s = "LinearSchedule_"
        return s + "timesteps={}_final={}_starting={}".format(self.schedule_timesteps, self.final_p, self.initial_p)

    def next_value(self):
        """See Schedule.value"""
        self.t += 1
        fraction = min(float(self.t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

    def get_value(self):
        """See Schedule.value"""
        fraction = min(float(self.t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class SoftMaxSchedule(Schedule):
    def __init__(self, schedule_time_steps, final_t, initial_t):
        """
        Value is not relevant to policy.
        """
        self.is_epsilon_greedy = False
        self.schedule_time_steps = schedule_time_steps
        self.final_t = final_t
        self.initial_t = initial_t
        self.step = 0

    def name(self):
        return "Softmax Schedule"

    def next_value(self):
        self.step += 1
        fraction = min(float(self.step) / self.schedule_time_steps, 1.0)
        return self.initial_t + fraction * (self.final_t - self.initial_t)

    def get_value(self):
        fraction = min(float(self.step) / self.schedule_time_steps, 1.0)
        return self.initial_t + fraction * (self.final_t - self.initial_t)
