from LLIEs.llie_enum import LLIEMethod

LIGHTING_FOLDERS = {
    "normal": "MPIIGaze_Normalized_Images",
    "low_i": "low_light_simulated/low_i",
    "low_g": "low_light_simulated/low_g",
    "low_left": "low_light_simulated/low_left",
    "low_right": "low_light_simulated/low_right",
    "low_i_g_left": "low_light_simulated/low_i_g_left",
    "low_i_g_right": "low_light_simulated/low_i_g_right",
}

LLIE_METHODS = [
    LLIEMethod.NONE,
    LLIEMethod.HE,
    LLIEMethod.CLAHE,
    LLIEMethod.SSR,
    LLIEMethod.MSR,
    LLIEMethod.MSRCR,
    LLIEMethod.ZERODCE,
    LLIEMethod.ENLIGHTENGAN,
    LLIEMethod.MIRNET,
]

PARAMS = {
    LLIEMethod.SSR: [{"sigma": 15}, {"sigma": 30}, {"sigma": 60}],
    LLIEMethod.CLAHE: [{"clip_limit": 2.0}, {"clip_limit": 4.0}],
    LLIEMethod.MSR: [{"sigmas": [15,80,250]}, {"sigmas": [30,100,300]}],
    "default": [{}]
}