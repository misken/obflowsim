import pytest
import simpy
from numpy.random import default_rng

import obflowsim.simulate as obf
from obflowsim.obconstants import ArrivalType, PatientType, UnitName


class TestStopConditions:

    def test_stop_max_arrivals(self):
        """
        Ensure arrivals stop after max_arrivals

        Returns
        -------

        """
        env = simpy.Environment()
        arrival_type = ArrivalType.SPONT_LABOR
        arrival_rate = 1.0
        max_arrivals = 25
        seed = 271
        rg = default_rng(seed)

        arrival_generator = obf.PatientPoissonArrivals(env, arrival_type, arrival_rate, rg,
                                                       max_arrivals=max_arrivals)
        env.run()
        assert arrival_generator.num_patients_created == max_arrivals

    def test_stop_time(self):
        """
        Ensure arrivals stop after stop_time

        Returns
        -------

        """
        env = simpy.Environment()
        arrival_type = ArrivalType.SPONT_LABOR
        arrival_rate = 1.0
        stop_time = 20.0
        seed = 271
        rg = default_rng(seed)

        arrival_generator = obf.PatientPoissonArrivals(env, arrival_type, arrival_rate, rg,
                                                       stop_time=stop_time)
        env.run()
        assert arrival_generator.env.now >= stop_time

