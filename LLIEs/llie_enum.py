from enum import Enum

class LLIEMethod(Enum):
    NONE = "none"
    HE = "he"
    CLAHE = "clahe"
    SSR = "ssr"
    MSR = "msr"
    ZERODCE = "zerodce"
    ENLIGHTENGAN = "enlightengan"