"""Approximate hull lengths of the named level 3 classes, in metres.

Rounded published figures, good to a few metres. Only classes that name one
ship class are listed; catch-all classes such as "Other Warship" have no
single length.
"""
from __future__ import annotations

CLASS_LENGTH_M = {
    "Enterprise": 342,
    "Nimitz": 333,
    "Midway": 296,
    "Wasp LL": 257,
    "LHA LL": 254,
    "Masyuu AS": 221,
    "YuZhao LL": 210,
    "Sanantonio AS": 208,
    "Hyuga DD": 197,
    "Commander": 194,
    "LSD 41 LL": 186,
    "Osumi LL": 178,
    "Austin LL": 173,
    "Ticonderoga": 173,
    "Atago DD": 165,
    "Arleigh Burke DD": 155,
    "Asagiri DD": 137,
    "Perry FF": 136,
    "Hatsuyuki DD": 130,
    "YuTing LL": 120,
    "EPF": 103,
}


def contradicts(label: str, measured_length_m: float, tolerance: float = 0.5) -> bool | None:
    """Whether a vessel is too short to be of the named class.

    True if the measured length is below `tolerance` times the class length,
    None for classes without a known length. Only "too short" is tested: a box
    that includes the wake overstates the length.
    """
    if label not in CLASS_LENGTH_M:
        return None
    return measured_length_m < tolerance * CLASS_LENGTH_M[label]
