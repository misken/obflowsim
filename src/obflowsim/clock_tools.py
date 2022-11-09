from simpy.core import Environment
import pandas as pd

from obflowsim.config import Config


class SimCalendar:
    """

    """

    def __init__(self, env: Environment, config: Config):
        self.env = env
        self.start_date = config.start_date
        self.base_time_unit = config.base_time_unit
        self.use_calendar_time = config.use_calendar_time

    def now(self):
        return self.to_sim_calendar_time(self.env.now)

    def to_sim_calendar_time(self, sim_time):
        elapsed_timedelta = pd.to_timedelta(sim_time, unit=self.base_time_unit)
        return self.start_date + elapsed_timedelta

    def to_sim_time(self, sim_calendar_time):
        elapsed_timedelta = sim_calendar_time - self.start_date
        return elapsed_timedelta / pd.to_timedelta(1, unit=self.base_time_unit)