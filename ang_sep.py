#--------------------------- Functions --------------------------------------------

import numpy as np

def ang_sep(lon1, lat1, lon2, lat2):
    """
    Angular separation between two points on a sphere

    Parameters
    ----------
    lon1, lat1, lon2, lat2 : Angle, Quantity or float
        Longitude and latitude of the two points.  Quantities should be in
        angular units; DEG

    Returns
    -------
    angular separation : Quantity or float
        Type depends on input; Quantity in angular units, or float in radians

    Notes
    -----
    The angular separation is calculated using the Vincenty formula [1]_,
    which is slightly more complex and computationally expensive than
    some alternatives, but is stable at at all distances, including the
    poles and antipodes.

    .. [1] http://en.wikipedia.org/wiki/Great-circle_distance
    """
    
    lon1, lat1, lon2, lat2 = np.deg2rad(lon1), np.deg2rad(lat1),np.deg2rad(lon2),np.deg2rad(lat2),
    
    sdlon = np.sin(lon2 - lon1)
    cdlon = np.cos(lon2 - lon1)
    slat1 = np.sin(lat1)
    slat2 = np.sin(lat2)
    clat1 = np.cos(lat1)
    clat2 = np.cos(lat2)
    num1 = clat2 * sdlon
    num2 = clat1 * slat2 - slat1 * clat2 * cdlon
    denominator = slat1 * slat2 + clat1 * clat2 * cdlon
    sep = np.arctan2(np.sqrt(num1 ** 2 + num2 ** 2), denominator)
    return np.rad2deg(sep)
