#from enum import StrEnum
from enum import Enum, StrEnum

class PatientType(StrEnum):
    """
    # Patient Type and Patient Flow Definitions

    # Type 1: random arrival spont labor, regular delivery, route = 1-2-4
    # Type 2: random arrival spont labor, C-section delivery, route = 1-3-2-4
    # Type 3: random arrival augmented labor, regular delivery, route = 1-2-4
    # Type 4: random arrival augmented labor, C-section delivery, route = 1-3-2-4
    # Type 5: sched arrival induced labor, regular delivery, route = 1-2-4
    # Type 6: sched arrival induced labor, C-section delivery, route = 1-3-2-4
    # Type 7: sched arrival, C-section delivery, route = 1-3-2-4

    # Type 8: urgent induced arrival, regular delivery, route = 1-2-4
    # Type 9: urgent induced arrival, C-section delivery, route = 1-3-2-4

    # Type 10: random arrival, non-delivered LD, route = 1
    # Type 11: random arrival, non-delivered PP route = 4

    TODO - Python 3.11 has strEnum
    """
    RAND_SPONT_REG = 'RAND_SPONT_REG'
    RAND_SPONT_CSECT = 'RAND_SPONT_CSECT'
    RAND_AUG_REG = 'RAND_AUG_REG'
    RAND_AUG_CSECT = 'RAND_AUG_CSECT'
    SCHED_IND_REG = 'SCHED_IND_REG'
    SCHED_IND_CSECT = 'SCHED_IND_CSECT'
    SCHED_CSECT = 'SCHED_CSECT'
    URGENT_IND_REG = 'URGENT_IND_REG'
    URGENT_IND_CSECT = 'URGENT_IND_CSECT'
    RAND_NONDELIV_LDR = 'RAND_NONDELIV_LDR'
    RAND_NONDELIV_PP = 'RAND_NONDELIV_PP'


class ArrivalType(StrEnum):
    """

    """
    SPONT_LABOR = 'spont_labor'
    URGENT_INDUCED_LABOR = 'urgent_induced_labor'
    NON_DELIVERY_LDR = 'non_delivery_ldr'
    NON_DELIVERY_PP = 'non_delivery_pp'
    SCHED_CSECT = 'sched_csect'
    SCHED_INDUCED_LABOR = 'sched_induced_labor'


class PatientTypeArrivalType:
    pat_type_to_arrival_type = {
        PatientType.RAND_SPONT_REG.value: ArrivalType.SPONT_LABOR,
        PatientType.RAND_SPONT_CSECT.value: ArrivalType.SPONT_LABOR,
        PatientType.RAND_AUG_REG.value: ArrivalType.SPONT_LABOR,
        PatientType.RAND_SPONT_CSECT.value: ArrivalType.SPONT_LABOR,
        PatientType.SCHED_IND_REG.value: ArrivalType.SCHED_INDUCED_LABOR,
        PatientType.SCHED_IND_CSECT.value: ArrivalType.SCHED_INDUCED_LABOR,
        PatientType.SCHED_CSECT.value: ArrivalType.SCHED_CSECT,
        PatientType.URGENT_IND_REG.value: ArrivalType.URGENT_INDUCED_LABOR,
        PatientType.URGENT_IND_CSECT.value: ArrivalType.URGENT_INDUCED_LABOR,
        PatientType.RAND_NONDELIV_LDR.value: ArrivalType.NON_DELIVERY_LDR,
        PatientType.RAND_NONDELIV_PP.value: ArrivalType.NON_DELIVERY_PP,
    }

class UnitName(StrEnum):
    """

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

# TODO: Why is this commented out?
# class OBunitId(IntEnum):
#     ENTRY = 0
#     OBS = 1
#     LDR = 2
#     CSECT = 3
#     PP = 4
#     LDRP = 5
#     LD = 6
#     RECOVERY = 8
#     EXIT = 8