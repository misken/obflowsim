import simpy

DEFAULT_BASE_TIME_UNIT = 'h'
DEFAULT_START_DATE = '2024-01-01'
DEFAULT_USE_CALENDAR_TIME = 1
DEFAULT_RUN_TIME = simpy.core.Infinity
DEFAULT_MAX_ARRIVALS = simpy.core.Infinity
DEFAULT_WARMUP_TIME = 0
DEFAULT_NUM_REPLICATIONS = 1

ALLOWED_LOS_DIST_LIST = ['exponential', 'gamma', 'normal', 'triangular', 'uniform', 'choice']

BASE_TIME_UNITS_PER_YEAR = {'h': 24 * 365, 'm': 1440 * 365}