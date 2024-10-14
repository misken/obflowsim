from enum import Enum, StrEnum
import simpy

# Debugging
MARKED_PATIENT = 'spont_labor_19'

# Defaults for key parameters
DEFAULT_BASE_TIME_UNIT = 'h'
DEFAULT_START_DATE = '2024-01-01'
DEFAULT_USE_CALENDAR_TIME = 1
DEFAULT_RUN_TIME = simpy.core.Infinity
DEFAULT_MAX_ARRIVALS = simpy.core.Infinity
DEFAULT_WARMUP_TIME = 0
DEFAULT_NUM_REPLICATIONS = 1
DEFAULT_GET_BED = 1
DEFAULT_RELEASE_BED = 1

ATT_GET_BED = 'get_bed'
ATT_RELEASE_BED = 'release_bed'

ALLOWED_LOS_DIST_LIST = ['exponential', 'gamma', 'normal', 'triangular', 'uniform', 'choice']

BASE_TIME_UNITS_PER_YEAR = {'h': 24 * 365, 'm': 1440 * 365}
BASE_TIME_UNITS_PER_DAY = {'h': 24, 'm': 1440}


class PatientType(StrEnum):
    """
    # Patient Type and Patient Flow Definitions

    # Type 1: random arrival spont labor, natural delivery, route = 1-2-4
    # Type 2: random arrival spont labor, C-section delivery, route = 1-3-2-4
    # Type 3: random arrival augmented labor, natural delivery, route = 1-2-4
    # Type 4: random arrival augmented labor, C-section delivery, route = 1-3-2-4
    # Type 5: sched arrival induced labor, natural delivery, route = 1-2-4
    # Type 6: sched arrival induced labor, C-section delivery, route = 1-3-2-4
    # Type 7: sched arrival, C-section delivery, route = 1-3-2-4

    # Type 8: urgent induced arrival, natural delivery, route = 1-2-4
    # Type 9: urgent induced arrival, C-section delivery, route = 1-3-2-4

    # Type 10: random arrival, non-delivered LD, route = 1
    # Type 11: random arrival, non-delivered PP route = 4
    """
    RAND_SPONT_NAT = 'RAND_SPONT_NAT'
    RAND_SPONT_CSECT = 'RAND_SPONT_CSECT'
    RAND_AUG_NAT = 'RAND_AUG_NAT'
    RAND_AUG_CSECT = 'RAND_AUG_CSECT'
    SCHED_IND_NAT = 'SCHED_IND_NAT'
    SCHED_IND_CSECT = 'SCHED_IND_CSECT'
    SCHED_CSECT = 'SCHED_CSECT'
    URGENT_IND_NAT = 'URGENT_IND_NAT'
    URGENT_IND_CSECT = 'URGENT_IND_CSECT'
    RAND_NONDELIV_LDR = 'RAND_NONDELIV_LDR'
    RAND_NONDELIV_PP = 'RAND_NONDELIV_PP'


class ArrivalType(StrEnum):
    """
    There are six distinct arrival streams of patients.
    """
    SPONT_LABOR = 'spont_labor'
    URGENT_INDUCED_LABOR = 'urgent_induced_labor'
    NON_DELIVERY_LDR = 'non_delivery_ldr'
    NON_DELIVERY_PP = 'non_delivery_pp'
    SCHED_CSECT = 'sched_csect'
    SCHED_INDUCED_LABOR = 'sched_induced_labor'


class PatientTypeArrivalType:
    """
    Mapping of patient types to arrival streams
    """
    pat_type_to_arrival_type = {
        PatientType.RAND_SPONT_NAT.value: ArrivalType.SPONT_LABOR,
        PatientType.RAND_SPONT_CSECT.value: ArrivalType.SPONT_LABOR,
        PatientType.RAND_AUG_NAT.value: ArrivalType.SPONT_LABOR,
        PatientType.RAND_AUG_CSECT.value: ArrivalType.SPONT_LABOR,
        PatientType.SCHED_IND_NAT.value: ArrivalType.SCHED_INDUCED_LABOR,
        PatientType.SCHED_IND_CSECT.value: ArrivalType.SCHED_INDUCED_LABOR,
        PatientType.SCHED_CSECT.value: ArrivalType.SCHED_CSECT,
        PatientType.URGENT_IND_NAT.value: ArrivalType.URGENT_INDUCED_LABOR,
        PatientType.URGENT_IND_CSECT.value: ArrivalType.URGENT_INDUCED_LABOR,
        PatientType.RAND_NONDELIV_LDR.value: ArrivalType.NON_DELIVERY_LDR,
        PatientType.RAND_NONDELIV_PP.value: ArrivalType.NON_DELIVERY_PP,
    }


class UnitName(StrEnum):
    """
    Each `PatientCareUnit` object should have one of these for its `name` property.
    """
    ENTRY = 'ENTRY'
    EXIT = 'EXIT'
    OBS = 'OBS'
    LDR = 'LDR'
    CSECT = 'CSECT'
    PP = 'PP'
    LDRP = 'LDRP'
    LD = 'LD'
    RECOVERY = 'R'
