from simpy.core import Environment
import pandas as pd

from obflowsim.config import Config


def to_sim_datetime(sim_time, start_date, base_time_unit):
    elapsed_timedelta = pd.to_timedelta(sim_time, unit=base_time_unit)
    return start_date + elapsed_timedelta


class SimCalendar:
    """

    """

    def __init__(self, env: Environment, config: Config):
        self.env = env
        self.start_date = config.start_date
        self.base_time_unit = config.base_time_unit
        self.use_calendar_time = config.use_calendar_time

    def now(self):
        return to_sim_datetime(self.env.now, self.start_date, self.base_time_unit)

    def datetime(self, sim_time: float):
        return to_sim_datetime(sim_time, self.start_date, self.base_time_unit)

    def date(self, sim_time):
        return to_sim_datetime(sim_time, self.start_date, self.base_time_unit).date



    def to_sim_time(self, sim_calendar_time):
        elapsed_timedelta = sim_calendar_time - self.start_date
        return elapsed_timedelta / pd.to_timedelta(1, unit=self.base_time_unit)