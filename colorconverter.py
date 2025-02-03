"""
Color Converter
===============

This script provides a comprehensive set of functions and CLI options to
convert between various color models and spaces using the Colour Science Python library.
It also offers plotting and analysis features.

"""

import argparse
import json
import sys
from matplotlib.pyplot import close, grid, savefig, xticks, yticks
import numpy as np


from colour import (
    CCS_ILLUMINANTS,
    chromatic_adaptation,
    CMY_to_CMYK,
    colorimetric_purity,
    colour_fidelity_index,
    colour_quality_scale,
    colour_rendering_index,
    complementary_wavelength,
    dominant_wavelength,
    excitation_purity,
    IPT_hue_angle,
    is_within_macadam_limits,
    is_within_pointer_gamut,
    Lab_to_LCHab,
    Lab_to_XYZ,
    LCHab_to_Lab,
    LCHuv_to_Luv,
    lightness,
    luminance,
    luminous_efficacy,
    luminous_efficiency,
    luminous_flux,
    Luv_to_LCHuv,
    Luv_to_XYZ,
    MEDIA_PARAMETERS_KIM2009,
    MSDS_CMFS,
    munsell_value,
    RGB_COLOURSPACES,
    RGB_luminance,
    RGB_to_CMY,
    RGB_to_HCL,
    RGB_to_HSL,
    RGB_to_HSV,
    RGB_to_IHLS,
    RGB_to_Prismatic,
    RGB_to_RGB,
    RGB_to_XYZ,
    RGB_to_YCbCr,
    RGB_to_YcCbcCrc,
    RGB_to_YCoCg,
    sd_to_XYZ,
    SDS_ILLUMINANTS,
    spectral_similarity_index,
    SpectralDistribution,
    SpectralShape,
    TVS_ILLUMINANTS,
    TVS_ILLUMINANTS_HUNTERLAB,
    UCS_to_uv,
    uv_to_CCT,
    VIEWING_CONDITIONS_CAM16,
    VIEWING_CONDITIONS_CIECAM02,
    VIEWING_CONDITIONS_CIECAM16,
    VIEWING_CONDITIONS_HELLWIG2022,
    VIEWING_CONDITIONS_HUNT,
    VIEWING_CONDITIONS_KIM2009,
    VIEWING_CONDITIONS_LLAB,
    VIEWING_CONDITIONS_RLAB,
    VIEWING_CONDITIONS_ZCAM,
    wavelength_to_XYZ,
    xy_to_Luv_uv,
    xy_to_XYZ,
    xyY_to_munsell_colour,
    XYZ_to_ATD95,
    XYZ_to_CAM16,
    XYZ_to_CIECAM02,
    XYZ_to_CIECAM16,
    XYZ_to_DIN99,
    XYZ_to_hdr_CIELab,
    XYZ_to_hdr_IPT,
    XYZ_to_Hellwig2022,
    XYZ_to_Hunt,
    XYZ_to_Hunter_Lab,
    XYZ_to_Hunter_Rdab,
    XYZ_to_ICaCb,
    XYZ_to_ICtCp,
    XYZ_to_IgPgTg,
    XYZ_to_IPT,
    XYZ_to_IPT_Ragoo2021,
    XYZ_to_Jzazbz,
    XYZ_to_K_ab_HunterLab1966,
    XYZ_to_Kim2009,
    XYZ_to_Lab,
    XYZ_to_LLAB,
    XYZ_to_Luv,
    XYZ_to_Nayatani95,
    XYZ_to_Oklab,
    XYZ_to_OSA_UCS,
    XYZ_to_ProLab,
    XYZ_to_RGB,
    XYZ_to_RLAB,
    XYZ_to_UCS,
    XYZ_to_UVW,
    XYZ_to_xy,
    XYZ_to_xyY,
    XYZ_to_Yrg,
    XYZ_to_ZCAM,
)
from colour.appearance import D_FACTOR_RLAB
from colour.colorimetry import (
    whiteness_CIE2004,
    whiteness_Berger1959,
    whiteness_Stensby1968,
    whiteness_ASTME313,
    yellowness_ASTMD1925,
    yellowness_ASTME313,
)
from colour.graph.conversion import (
    CAM16_to_JMh_CAM16,
    JMh_CAM16_to_CAM16LCD,
    JMh_CAM16_to_CAM16SCD,
    JMh_CAM16_to_CAM16UCS,
    CIECAM02_to_JMh_CIECAM02,
    JMh_CIECAM02_to_CAM02LCD,
    JMh_CIECAM02_to_CAM02SCD,
    JMh_CIECAM02_to_CAM02UCS,
    CIECAM16_to_JMh_CIECAM16,
    Hellwig2022_to_JMh_Hellwig2022,
)
from colour.models import XYZ_to_Iab, XYZ_to_Izazbz
from colour.plotting import plot_single_sd_colour_rendition_report, plot_multi_sds
from colour.recovery import XYZ_to_sd_Jakob2019
from colour.adaptation import chromatic_adaptation_VonKries
from colour.notation import RGB_to_HEX
from colour.utilities import filter_warnings


# FACTOR FOR NORMALIZATION OF VALUES TO 0:1 DEPENDING ON BITDEPTH
BITDEPTH_FACTOR = {"8": 255, "15+1": 32768, "16": 65535, "32": 1}
CIELAB_DEPTH_VALUES = {
    "8": {"L": (0, 100), "ab": (-200, 200)},
    "15+1": {"L": (0, 32768), "ab": (-25600, 25600)},
    "16": {"L": (0, 65535), "ab": (-51200, 51200)},
    "32": {"L": (0, 1), "ab": (-1, 1)},
}
CIELCHUV_DEPTH_VALUES = {
    "8": {"L": (0, 100), "Ch": (0, 230), "uv": (0, 360)},
    "15+1": {
        "L": (0, 32768),
        "Ch": (0, 230 * 327.68),
        "uv": (0, 360 * 327.68),
    },  # Assuming a linear scale with bitdepth
    "16": {
        "L": (0, 65535),
        "Ch": (0, 230 * 655.35),
        "uv": (0, 360 * 655.35),
    },  # Assuming a linear scale with bitdepth
    "32": {"L": (0, 1), "Ch": (0, 230), "uv": (0, 360)},
}
CIEXYZ_DEPTH_VALUES = {"8": (0, 1), "15+1": (0, 1), "16": (0, 1), "32": (0, 1)}
VIS_SPECTRUM_MIN = 360  # nm [CIE 015:2018, pg. 21]
VIS_SPECTRUM_MAX = 830  # nm [CIE 015:2018, pg. 21]


# CIE ILLUMINANTS
# https://www.liquisearch.com/standard_illuminant/white_point/white_points_of_standard_illuminants
# https://en.wikipedia.org/wiki/Standard_illuminant

# ISO 7589:2002(E) ILLUMINANTS
# "Photography - Illuminants for sensitometry - Specifications for daylight, incandescent
# tungsten and printer"
# https://www.iso.org/standard/33979.html

# ACES, Blackmagic Wide Gamut, and DCI-P3 illuminants are used only by a small number of
# spaces in cinema and visual effects

# Illuminants C and E are present in the RGB and CIE_ALL lists because we have color
# spaces that use them
RGB_ILLUMINANTS_LIST = [
    "A",
    "C",
    "D50",
    "D55",
    "D65",
    "D75",
    "E",
    "FL1",
    "FL2",
    "FL3",
    "FL3.1",
    "FL3.2",
    "FL3.3",
    "FL3.4",
    "FL3.5",
    "FL3.6",
    "FL3.7",
    "FL3.8",
    "FL3.9",
    "FL3.10",
    "FL3.11",
    "FL3.12",
    "FL3.13",
    "FL3.14",
    "FL3.15",
    "FL4",
    "FL5",
    "FL6",
    "FL7",
    "FL8",
    "FL9",
    "FL10",
    "FL11",
    "FL12",
    "HP1",
    "HP2",
    "HP3",
    "HP4",
    "HP5",
    "ID50",
    "ID65",
    "LED-B1",
    "LED-B2",
    "LED-B3",
    "LED-B4",
    "LED-B5",
    "LED-BH1",
    "LED-RGB1",
    "LED-V1",
    "LED-V2",
    "ISO 7589 Sensitometric Daylight",
    "ISO 7589 Sensitometric Studio Tungsten",
    "ISO 7589 Sensitometric Photoflood",
    "ISO 7589 Sensitometric Printer",
    "ACES",
    "Blackmagic Wide Gamut",
    "DCI-P3",
]

CIE_ILLUMINANTS_LIST = [
    "A",
    "D50",
    "D55",
    "D65",
    "D75",
    "FL1",
    "FL2",
    "FL3",
    "FL3.1",
    "FL3.2",
    "FL3.3",
    "FL3.4",
    "FL3.5",
    "FL3.6",
    "FL3.7",
    "FL3.8",
    "FL3.9",
    "FL3.10",
    "FL3.11",
    "FL3.12",
    "FL3.13",
    "FL3.14",
    "FL3.15",
    "FL4",
    "FL5",
    "FL6",
    "FL7",
    "FL8",
    "FL9",
    "FL10",
    "FL11",
    "FL12",
    "HP1",
    "HP2",
    "HP3",
    "HP4",
    "HP5",
    "ID50",
    "ID65",
    "LED-B1",
    "LED-B2",
    "LED-B3",
    "LED-B4",
    "LED-B5",
    "LED-BH1",
    "LED-RGB1",
    "LED-V1",
    "LED-V2",
]

CIE_NON_STANDARD_LIST = ["B", "C", "D60", "E"]

CIE_ALL_ILLUMINANTS_LIST = [
    "A",
    "B",
    "C",
    "D50",
    "D55",
    "D60",
    "D65",
    "D75",
    "E",
    "FL1",
    "FL2",
    "FL3",
    "FL3.1",
    "FL3.2",
    "FL3.3",
    "FL3.4",
    "FL3.5",
    "FL3.6",
    "FL3.7",
    "FL3.8",
    "FL3.9",
    "FL3.10",
    "FL3.11",
    "FL3.12",
    "FL3.13",
    "FL3.14",
    "FL3.15",
    "FL4",
    "FL5",
    "FL6",
    "FL7",
    "FL8",
    "FL9",
    "FL10",
    "FL11",
    "FL12",
    "HP1",
    "HP2",
    "HP3",
    "HP4",
    "HP5",
    "ID50",
    "ID65",
    "LED-B1",
    "LED-B2",
    "LED-B3",
    "LED-B4",
    "LED-B5",
    "LED-BH1",
    "LED-RGB1",
    "LED-V1",
    "LED-V2",
]

ISO_7589_ILLUMINANTS_LIST = [
    "ISO 7589 Sensitometric Daylight",
    "ISO 7589 Sensitometric Studio Tungsten",
    "ISO 7589 Sensitometric Photoflood",
    "ISO 7589 Sensitometric Printer",
]

# CHROMATIC ADAPTATION TRANSFORMS SUPPORTED IN COLOUR
CHROMATIC_ADAPTATION_TRANSFORMS = [
    "Bianco 2010",
    "Bianco PC 2010",
    "Bradford",
    "CAT02",
    "CAT02 Brill 2008",
    "CAT16",
    "CMCCAT97",
    "CMCCAT2000",
    "Fairchild",
    "Sharp",
    "Von Kries",
    "XYZ Scaling",
]

STANDARD_OBSERVERS = [
    "CIE 1931 2 Degree Standard Observer",
    "CIE 1964 10 Degree Standard Observer",
]

# RGB COLORSPACES SUPPORTED BY CSS4
RGB_CSS_COLORSPACES = [
    "sRGB",
    "Adobe RGB (1998)",
    "Display P3",
    "ITU-R BT.2020",
    "ProPhoto RGB",
]

# CAM constants
CCT_W = 6504

D_FACTOR_RLAB = D_FACTOR_RLAB["Hard Copy Images"]
HUNTER_D65 = TVS_ILLUMINANTS_HUNTERLAB["CIE 1931 2 Degree Standard Observer"]["D65"]
E_O = 5000.0
E_OR = 1000.0

E_O1 = 200.0
E_O2 = 200.0

K_1 = 0.0
K_2 = 50.0

L_A = L = Y_0 = 318.31
L_A_ZCAM = 264
L_A1 = 200
L_A2 = 200

Y_B = Y_O = 20.0
Y_B_ZCAM = 100
Y_N = 31.83

# Conversions to LMS space needed by some calculations
LMS_TO_LMS_P = lambda x: x**0.43
M_XYZ_TO_LMS = np.array(
    [
        [0.4002, 0.7075, -0.0807],
        [-0.2280, 1.1500, 0.0612],
        [0.0000, 0.0000, 0.9184],
    ]
)
M_LMS_P_TO_IAB = np.array(
    [
        [0.4000, 0.4000, 0.2000],
        [4.4550, -4.8510, 0.3960],
        [0.8056, 0.3572, -1.1628],
    ]
)

# CAM viewing conditions defaults
media_kim2009 = MEDIA_PARAMETERS_KIM2009["CRT Displays"]
sigma_rlab = VIEWING_CONDITIONS_RLAB["Average"]
surround_cam16 = VIEWING_CONDITIONS_CAM16["Average"]
surround_ciecam02 = VIEWING_CONDITIONS_CIECAM02["Average"]
surround_ciecam16 = VIEWING_CONDITIONS_CIECAM16["Average"]
surround_hellwig2022 = VIEWING_CONDITIONS_HELLWIG2022["Average"]
surround_hunt = VIEWING_CONDITIONS_HUNT["Normal Scenes"]
surround_kim2009 = VIEWING_CONDITIONS_KIM2009["Average"]
surround_llab = VIEWING_CONDITIONS_LLAB["ref_average_4_minus"]
surround_zcam = VIEWING_CONDITIONS_ZCAM["Average"]


# DICTIONARY OF COLOR MODELS, WITH SECTION GROUPINGS AND DESCRIPTION
COLOR_MODELS_TEMPLATES = {}

# RGB COLOR SPACES
COLOR_MODELS_TEMPLATES["ACES2065-1"] = {
    "name": "ACES2065-1",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "Used primarily in cinema for exchange of full fidelity images and "
        "archiving; not recommended for rendering"
    ),
    "illuminant": "ACES",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}

COLOR_MODELS_TEMPLATES["ACEScc"] = {
    "name": "ACEScc",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "Workspace for color correctors, target for ASC-CDL values created " "on-set"
    ),
    "illuminant": "ACES",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["ACEScct"] = {
    "name": "ACEScct",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "Alternative workspace for color correctors, intended to be transient "
        "and internal to software or hardware systems; not intended for "
        "interchange or archiving"
    ),
    "illuminant": "ACES",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["ACEScg"] = {
    "name": "ACEScg",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "Workspace for painting/compositing applications that don't support "
        "ACES2065-1 or ACEScc"
    ),
    "illuminant": "ACES",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["ACESproxy"] = {
    "name": "ACESproxy",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "A lightweight encoding for transmission over HD-SDI (or other "
        "production transmission schemes), on-set look management. Not "
        "intended to be stored or used in production imagery"
    ),
    "illuminant": "ACES",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["ARRI Wide Gamut 3"] = {
    "name": "ARRI Wide Gamut 3 / LogCv3",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "Based on virtual primaries optimized for the encoding of the color "
        "data generated by ARRI camera systems, often referred to as LogC3"
    ),
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["ARRI Wide Gamut 4"] = {
    "name": "ARRI Wide Gamut 4 / LogC4",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "Sometimes referred to as LogC4, this space provides the encoding "
        "used for media from the ARRI ALEXA 35"
    ),
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["Adobe RGB (1998)"] = {
    "name": "A98 RGB",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "Originally defined to meet the demands for an RGB working space "
        "suited for print production"
    ),
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["Adobe Wide Gamut RGB"] = {
    "name": "Adobe Wide Gamut RGB",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "Offers a large gamut by using pure spectral primary colors; covers "
        "77.6% of visible colors in CIELAB"
    ),
    "illuminant": "D50",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["Apple RGB"] = {
    "name": "Apple RGB",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "Describes the characteristics of a legacy Apple monitor, used for "
        "press workflows before the widespread adoption of color management"
    ),
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["Best RGB"] = {
    "name": "Best RGB",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "Almost identical to Don RGB 4 except for a modified red coordinate "
        "that helps encompass the reds and magentas in Fujichrome Velvia, "
        "and a slight increase in green saturation; created by Don Hutcheson "
        "at HutchColor"
    ),
    "illuminant": "D50",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["Beta RGB"] = {
    "name": "Beta RGB",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "Designed by Bruce Lindbloom as an optimized capture, archiving and "
        "editing space for high-end digital imaging applications"
    ),
    "illuminant": "D50",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["Blackmagic Wide Gamut"] = {
    "name": "Blackmagic Wide Gamut",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "A custom non-linear color space created by Blackmagic Design to "
        "preserve maximum color data and dynamic range"
    ),
    "illuminant": "Blackmagic Wide Gamut",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["CIE RGB"] = {
    "name": "CIE RGB",
    "section": "CIE Fundamentals",
    "ui_group": "cie",
    "description": (
        "Created by the CIE in 1931, leveraging the results of a series of "
        "experiments done in the late 1920s"
    ),
    "illuminant": "E",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["Cinema Gamut"] = {
    "name": "Canon Cinema Gamut",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "Cinema Gamut is a set of primaries specified by Canon, typically "
        "paired with the Canon Log 2 or Canon Log 3 curve"
    ),
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["ColorMatch RGB"] = {
    "name": "ColorMatch RGB",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "A legacy space created by Radius to work with their PressView " "displays"
    ),
    "illuminant": "D50",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["DCDM XYZ"] = {
    "name": "DCDM XYZ",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "Digital Cinema Distribution Master [DCDM] was devised by Digital "
        "Cinema Initiatives, LLC (DCI) for the purpose of exchanging content "
        "to encoding systems, as well as to the Digital Cinema playback "
        "system"
    ),
    "illuminant": "E",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["DCI-P3"] = {
    "name": "DCI-P3",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "First defined as part of the Digital Cinema Initiative in 2005 for "
        "digital motion picture distribution"
    ),
    "illuminant": "DCI-P3",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["DCI-P3-P"] = {
    "name": "DCI-P3+",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": "An expanded gamut version of DCI-P3",
    "illuminant": "DCI-P3",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["DJI D-Gamut"] = {
    "name": "DJI D-Gamut",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "Designed to encompass the capabilities of the DJI Zenmuse sensor, "
        "while simultaneously providing an ideal starting point for color "
        "grading"
    ),
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["DRAGONcolor"] = {
    "name": "DRAGONcolor",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "Acts as a color engine that converts between the camera's native "
        "gamut and monitor RGB, which is assumed to be close to Rec. 709, "
        "from RED"
    ),
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["DRAGONcolor2"] = {
    "name": "DRAGONcolor2",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": "A RED color space evolution of DRAGONcolor",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["DaVinci Wide Gamut"] = {
    "name": "DaVinci Wide Gamut",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "DWG is designed to accommodate the vast majority of colors that can "
        "be captured using the latest cameras and capture devices; it "
        "facilitates the storage and manipulation of intermediate image data "
        "in modern production pipelines"
    ),
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["Display P3"] = {
    "name": "Display P3",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "Display P3 is a variant of the DCI-P3 color space, advanced by "
        "Apple to better characterize the wider gamut possible on "
        "contemporary devices"
    ),
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["Don RGB 4"] = {
    "name": "Don RGB 4",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "Wide gamut workspace with a 2.2 gamma. Captures the Ektachrome "
        "color gamut with virtually no clipping, from Don Hutcheson at "
        "HutchColor"
    ),
    "illuminant": "D50",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["EBU Tech. 3213-E"] = {
    "name": "EBU Tech. 3213-E",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "In 1975, the European Broadcasting Union recommended this space for "
        "studio monitors for broadcasting organizations using PAL or SECAM "
        "signals"
    ),
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["ECI RGB v2"] = {
    "name": "ECI RGB v2",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "The Metamorfoze Preservation Imaging Guidelines promotes use of "
        "this European Color Initiative space - and requires it at their "
        "strictest level of compliance"
    ),
    "illuminant": "D50",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["ERIMM RGB"] = {
    "name": "ERIMM RGB",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "Extended Reference Input Medium Metric, intended to encode extended "
        "dynamic range scene-referred images"
    ),
    "illuminant": "D50",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["Ekta Space PS 5"] = {
    "name": "Ekta Space PS 5",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "Developed by Joseph Holmes for high quality storage of image data "
        "from scans of transparencies"
    ),
    "illuminant": "D50",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["F-Gamut"] = {
    "name": "Fujifilm F-Log",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": "The gamut of Fujifilm's F-log space complies with ITU-R BT.2020",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["FilmLight E-Gamut"] = {
    "name": "FilmLight T-Log / E-Gamut",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "The gamut of this working space is designed to cover most colors "
        "produced by contemporary digital cinema cameras"
    ),
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["ITU-R BT.2020"] = {
    "name": "ITU-R BT.2020",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "Often referred to simply as Rec.2020, this space provides parameter "
        "values for ultra-high definition television (UHDTV)"
    ),
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["ITU-R BT.470 - 525"] = {
    "name": "ITU-R BT.470 - 525",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "ITU-R BT.470 - 525 was recommended to define a standard for "
        "conventional television signals"
    ),
    "illuminant": "C",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["ITU-R BT.470 - 625"] = {
    "name": "ITU-R BT.470 - 625",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "ITU-R BT.470 - 525 was recommended to define a standard for "
        "conventional television signals"
    ),
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["ITU-R BT.709"] = {
    "name": "ITU-R BT.709",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "Standard developed by the ITU for image encoding and signal "
        "characteristics of high-definition television"
    ),
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["ITU-T H.273 - 22 Unspecified"] = {
    "name": "ITU-T H.273 - 22 Unspecified",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "Recommendation ITU-T H.273 row 22 color space as given in Table 2 - "
        '"Interpretation of colour primaries (ColourPrimaries)" value'
    ),
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["ITU-T H.273 - Generic Film"] = {
    "name": "ITU-T H.273 - Generic Film",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "Recommendation ITU-T H.273 Generic Film (color filters using "
        "illuminant C) color space"
    ),
    "illuminant": "C",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["Max RGB"] = {
    "name": "Max RGB",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "A wide gamut color space with colors outside the xyY limits, "
        "created by Don Hutcheson at HutchColor"
    ),
    "illuminant": "D50",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["N-Gamut"] = {
    "name": "N-Gamut",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["NTSC (1953)"] = {
    "name": "NTSC (1953)",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "The original 1953 color NTSC specification, still part of the "
        "United States Code of Federal Regulations"
    ),
    "illuminant": "C",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["NTSC (1987)"] = {
    "name": "NTSC (1987)",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "In 1987, SMPTE adopted the SMPTE C (Conrac) phosphors for general "
        "use, defining this space"
    ),
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["P3-D65"] = {
    "name": "P3-D65",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["Pal/Secam"] = {
    "name": "PAL/SECAM",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": "Defined in ITU-R BT.470-6, in 1998",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["PLASA ANSI E1.54"] = {
    "name": "PLASA ANSI E1.54",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "A standard codified by the Professional Lighting & Sound "
        "Association (PLASA) and ANSI for the purpose of facilitating "
        "communication between lighting controllers and color-changing "
        "luminaires in entertainment"
    ),
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["ProPhoto RGB"] = {
    "name": "ProPhoto RGB",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "This space, invented by Kodak, offers an especially large gamut. "
        "13% of its gamut is comprised of imaginary colors. Also known as "
        "ROMM RGB"
    ),
    "illuminant": "D50",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["Protune Native"] = {
    "name": "Protune Native",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": "Native color space in GoPro Protune settings",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["REDWideGamutRGB"] = {
    "name": "REDWideGamutRGB",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["REDcolor"] = {
    "name": "REDcolor",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["REDcolor2"] = {
    "name": "REDcolor2",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["REDcolor3"] = {
    "name": "REDcolor3",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["REDcolor4"] = {
    "name": "REDcolor4",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["RIMM RGB"] = {
    "name": "RIMM RGB",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "Reference Input Medium Metric, intended to encode standard dynamic "
        "range scene-referred images"
    ),
    "illuminant": "D50",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["ROMM RGB"] = {
    "name": "ROMM RGB",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "Reference Output Medium Metric RGB, invented by Kodak, offers an "
        "especially large gamut. 13% of its gamut is comprised of imaginary "
        "colors. Also known as ProPhoto RGB"
    ),
    "illuminant": "D50",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["Russell RGB"] = {
    "name": "Russell RGB",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "Russell Cottrell extended Beta RGB into the cyans and blues to "
        "accommodate printers with many pigment inks, such as the "
        "imagePROGRAF iPF6300 and PIXMA Pro9500 Mark II"
    ),
    "illuminant": "D55",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["S-Gamut"] = {
    "name": "S-Gamut",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["S-Gamut3"] = {
    "name": "S-Gamut3",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["S-Gamut3.Cine"] = {
    "name": "S-Gamut3.Cine",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["SMPTE 240M"] = {
    "name": "SMPTE 240M",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["SMPTE C"] = {
    "name": "SMPTE C",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["Sharp RGB"] = {
    "name": "Sharp RGB",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": "",
    "illuminant": "E",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["V-Gamut"] = {
    "name": "V-Gamut",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": "Panasonic V-Gamut color space",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["Venice S-Gamut3"] = {
    "name": "Venice S-Gamut3",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["Venice S-Gamut3.Cine"] = {
    "name": "Venice S-Gamut3.Cine",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["Xtreme RGB"] = {
    "name": "Xtreme RGB",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "Largest possible Adobe Photoshop-legal, tri-coordinate RGB space, "
        "from HutchColor"
    ),
    "illuminant": "D50",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
COLOR_MODELS_TEMPLATES["sRGB"] = {
    "name": "sRGB",
    "section": "RGB Color Spaces",
    "ui_group": "rgb",
    "description": (
        "Microsoft and Hewlett-Packard designed sRGB in 1996 as a color "
        "standard to use on monitors, printers, and the Internet"
    ),
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Red", "Green", "Blue"],
    "codes": ["R", "G", "B"],
}
# COLOUR LIBRARY ALIASES FOR RGB SPACES
COLOR_MODELS_ALIASES = ["aces", "adobe1998", "prophoto", "ALEXA Wide Gamut"]
# NON-RGB SPACES THAT DO NOT USE XYZ OR STANDARD OBSERVER
COLOR_MODELS_TEMPLATES["cmy"] = {
    "name": "CMY",
    "section": "Derived from RGB",
    "ui_group": "rgb >",
    "description": (
        "Subtractive counterpart to RGB, where higher values get closer to "
        "black, and white is the absence of values"
    ),
    "illuminant": "none",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Cyan", "Magenta", "Yellow"],
    "codes": ["C", "M", "Y"],
}
COLOR_MODELS_TEMPLATES["cmyk"] = {
    "name": "CMYK",
    "section": "Derived from RGB",
    "ui_group": "rgb >",
    "description": "Subtractive color model used in color printing",
    "illuminant": "none",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Cyan", "Magenta", "Yellow", "Black"],
    "codes": ["C", "M", "Y", "K"],
}
COLOR_MODELS_TEMPLATES["hcl"] = {
    "name": "HCL",
    "section": "Derived from RGB",
    "ui_group": "rgb >",
    "description": (
        "HCL (Hue-Chroma-Luminance) or LCh refers to any of the cylindrical "
        "color space models that are designed to accord with human "
        "perception of color"
    ),
    "illuminant": "none",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Hue", "Chroma", "Luminance"],
    "codes": ["H", "C", "L"],
}
COLOR_MODELS_TEMPLATES["hexadecimal"] = {
    "name": "Hexadecimal",
    "section": "Derived from RGB",
    "ui_group": "rgb >",
    "description": "Conversion of RGB values into base-16 notation",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "codes": ["Hex"],
}
COLOR_MODELS_TEMPLATES["hsl"] = {
    "name": "HSL",
    "section": "Derived from RGB",
    "ui_group": "rgb >",
    "description": (
        "HSL models the way different paints mix together in the real world, "
        "with Lightness resembling the varying amounts of black or white "
        "paint in the mixture"
    ),
    "illuminant": "none",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Hue", "Saturation", "Lightness"],
    "codes": ["H", "S", "L"],
}
COLOR_MODELS_TEMPLATES["hsv"] = {
    "name": "HSV",
    "section": "Derived from RGB",
    "ui_group": "rgb >",
    "description": (
        "Colors of each hue are arranged in a radial slice around a central "
        "axis of neutral colors which range from black at the bottom to "
        "white at the top"
    ),
    "illuminant": "none",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Hue", "Saturation", "Value"],
    "codes": ["H", "S", "V"],
}
COLOR_MODELS_TEMPLATES["hwb"] = {
    "name": "HWB",
    "section": "Derived from RGB",
    "ui_group": "rgb >",
    "description": (
        "Hue angle followed by the percentage of whiteness and percentage of "
        "blackness"
    ),
    "illuminant": "none",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Hue", "Whiteness", "Blackness"],
    "codes": ["H", "W", "B"],
}
COLOR_MODELS_TEMPLATES["ihls"] = {
    "name": "IHLS",
    "section": "Derived from RGB",
    "ui_group": "rgb >",
    "description": "Improved HLS [Hanbury, 2003]",
    "illuminant": "none",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Hue", "Luminance", "Saturation"],
    "codes": ["H", "Y", "S"],
}
COLOR_MODELS_TEMPLATES["prismatic"] = {
    "name": "Prismatic",
    "section": "Derived from RGB",
    "ui_group": "rgb >",
    "description": (
        "A simple transform of the RGB color cube into a light/dark "
        "dimension and a 2D hue"
    ),
    "illuminant": "none",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Luminance", "&rho;", "&gamma;", "&beta;"],
    "codes": ["L", "&rho;", "&gamma;", "&beta;"],
}
COLOR_MODELS_TEMPLATES["rgbluminance"] = {
    "name": "Luminance",
    "section": "Derived from RGB",
    "ui_group": "rgb >",
    "description": (
        "A photometric measure of the luminous intensity per unit area of "
        "light travelling in a given direction"
    ),
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Luminance"],
    "codes": ["Y"],
}
COLOR_MODELS_TEMPLATES["ycbcr"] = {
    "name": "YCbCr",
    "section": "Derived from RGB",
    "ui_group": "rgb >",
    "description": (
        "Among a family of color spaces used as a part of the color pipeline "
        "in video and digital photography systems"
    ),
    "illuminant": "none",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Luma", "Chroma (blue-difference)", "Chroma (red-difference)"],
    "codes": ["Y&rsquo;", "C&rsquo;<sub>B</sub>", "C&rsquo;<sub>R</sub>"],
}
COLOR_MODELS_TEMPLATES["yccbccrc"] = {
    "name": "YcCbcCrc",
    "section": "Derived from RGB",
    "ui_group": "rgb >",
    "description": (
        "Specifically for use with ITU-R BT.2020 when adopting the constant "
        "luminance signal format"
    ),
    "illuminant": "none",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Luma", "Chroma 1", "Chroma 2"],
    "codes": ["Y&rsquo;", "C&rsquo;<sub>BC</sub>", "C&rsquo;<sub>RC</sub>"],
}
COLOR_MODELS_TEMPLATES["ycocg"] = {
    "name": "YCoCg",
    "section": "Derived from RGB",
    "ui_group": "rgb >",
    "description": (
        "Formed from a simple transformation of an RGB space into a luma "
        "value (denoted as Y) and two chroma values called chrominance green "
        "(Cg) and chrominance orange (Co)"
    ),
    "illuminant": "none",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Luma", "Chrominance Orange", "Chrominance Green"],
    "codes": ["Y", "Co", "Cg"],
}
# SPACES THAT USE XYZ AS INTERMEDIARY AND MAY MAKE USE OF STANDARD OBSERVER
COLOR_MODELS_TEMPLATES["atd95"] = {
    "name": "ATD (1995)",
    "section": "Color Appearance Models",
    "ui_group": "cam",
    "description": (
        "Achromatic (brightness), Tritanopic (redness-greenness), and "
        "Deuteranopic (yellowishness-blueness). A model for color perception "
        "and visual adaptation [Guth, 1995]"
    ),
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "defaults": {"Y<sub>0</sub>": 318.31, "k<sub>1</sub>": 0.0, "k<sub>2</sub>": 50.0},
    "labels": [
        "Hue",
        "Saturation",
        "Correlate of Brightness",
        "A<sub>1</sub>",
        "T<sub>1</sub>",
        "D<sub>1</sub>",
        "A<sub>2</sub>",
        "T<sub>2</sub>",
        "D<sub>2</sub>",
    ],
    "codes": [
        "h",
        "C",
        "Q",
        "A<sub>1</sub>",
        "T<sub>1</sub>",
        "D<sub>1</sub>",
        "A<sub>2</sub>",
        "T<sub>2</sub>",
        "D<sub>2</sub>",
    ],
}
COLOR_MODELS_TEMPLATES["cam02lcd"] = {
    "name": "CAM02-LCD",
    "section": "Color Appearance Models",
    "ui_group": "cam",
    "description": (
        "CAM02-LCD is a version of CIECAM02 modified to fit Large Color "
        "Difference (LCD) data [Luo, Chi and Li (2006)]"
    ),
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "defaults": {
        "Ambient Illumination": "80 cd/m<sup>2</sup>, including veiling glare",
        "Adapting Field Luminance": "About 20% of a white object in the scene",
    },
    "labels": ["Lightness", "Redness-greenness", "Yellowness-blueness"],
    "codes": ["J&rsquo;", "a&rsquo;", "b&rsquo;"],
}
COLOR_MODELS_TEMPLATES["cam02scd"] = {
    "name": "CAM02-SCD",
    "section": "Color Appearance Models",
    "ui_group": "cam",
    "description": (
        "CAM02-SCD is a version of CIECAM02 modified to fit Small Color "
        "Difference (SCD) data [Luo, Chi and Li (2006)]"
    ),
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "defaults": {
        "Ambient Illumination": "80 cd/m<sup>2</sup>, including veiling glare",
        "Adapting Field Luminance": "About 20% of a white object in the scene",
    },
    "labels": ["Lightness", "Redness-greenness", "Yellowness-blueness"],
    "codes": ["J&rsquo;", "a&rsquo;", "b&rsquo;"],
}
COLOR_MODELS_TEMPLATES["cam02ucs"] = {
    "name": "CAM02-UCS",
    "section": "Color Appearance Models",
    "ui_group": "cam",
    "description": (
        "CAM02-UCS (Uniform Color Space) is a version of CIECAM02 modified "
        "to fit combined large and small color difference data sets "
        "[Luo, Chi and Li (2006)]"
    ),
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "defaults": {
        "Ambient Illumination": "80 cd/m<sup>2</sup>, including veiling glare",
        "Adapting Field Luminance": "About 20% of a white object in the scene",
    },
    "labels": ["Lightness", "Redness-greenness", "Yellowness-blueness"],
    "codes": ["J&rsquo;", "a&rsquo;", "b&rsquo;"],
}
COLOR_MODELS_TEMPLATES["cam16"] = {
    "name": "CAM16",
    "section": "Color Appearance Models",
    "ui_group": "cam",
    "description": (
        "CAM16 was created as a successor to CIECAM02 to address CIECAM02's "
        "computational failures in certain cases [Li et al (2017)]"
    ),
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "defaults": {
        "L<sub>A</sub>": 318.31,
        "Y<sub>b</sub>": 20,
        "Surround Viewing Conditions": "average",
        "Discount Illuminant?": "No",
    },
    "labels": [
        "Correlate of Lightness J",
        "Correlate of chroma C",
        "Hue angle h in degrees",
        "Correlate of saturation s",
        "Correlate of brightness Q",
        "Correlate of colorfulness M",
        "Hue h quadrature H",
    ],
    "codes": ["J", "C", "h", "s", "Q", "M", "H"],
}
COLOR_MODELS_TEMPLATES["cam16lcd"] = {
    "name": "CAM16-LCD",
    "section": "Color Appearance Models",
    "ui_group": "cam",
    "description": (
        "CAM16-LCD is a version of CAM16 attuned to fit Large Color "
        "Difference (LCD) data [Li et al (2017)]"
    ),
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "defaults": {
        "Ambient Illumination": "80 cd/m<sup>2</sup>, including veiling glare",
        "Adapting Field Luminance": "About 20% of a white object in the scene",
    },
    "labels": ["Lightness", "Redness-greenness", "Yellowness-blueness"],
    "codes": ["J&rsquo;", "a&rsquo;", "b&rsquo;"],
}
COLOR_MODELS_TEMPLATES["cam16scd"] = {
    "name": "CAM16-SCD",
    "section": "Color Appearance Models",
    "ui_group": "cam",
    "description": (
        "CAM16-SCD is a version of CAM16 attuned to fit Small Color "
        "Difference (SCD) data [Li et al (2017)]"
    ),
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "defaults": {
        "Ambient Illumination": "80 cd/m<sup>2</sup>, including veiling glare",
        "Adapting Field Luminance": "About 20% of a white object in the scene",
    },
    "labels": ["Lightness", "Redness-greenness", "Yellowness-blueness"],
    "codes": ["J&rsquo;", "a&rsquo;", "b&rsquo;"],
}
COLOR_MODELS_TEMPLATES["cam16ucs"] = {
    "name": "CAM16-UCS",
    "section": "Color Appearance Models",
    "ui_group": "cam",
    "description": (
        "CAM16-UCS is a version of CAM16 attuned to fit combined large and "
        "small color difference data sets [Li et al (2017)]"
    ),
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "defaults": {
        "Ambient Illumination": "80 cd/m<sup>2</sup>, including veiling glare",
        "Adapting Field Luminance": "About 20% of a white object in the scene",
    },
    "labels": ["Lightness", "Redness-greenness", "Yellowness-blueness"],
    "codes": ["J&rsquo;", "a&rsquo;", "b&rsquo;"],
}
COLOR_MODELS_TEMPLATES["cct"] = {
    "name": "Correlated Color Temperature [CCT]",
    "section": "Qualities",
    "ui_group": "qua",
    "description": (
        "CCT is defined by the CIE as 'the temperature of the Planckian "
        "radiator whose perceived color most closely resembles that of a "
        "given stimulus at the same brightness and under specified viewing "
        "conditions' [CIE/IEC 17.4:1987]'"
    ),
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["CCT"],
    "codes": ["T<sub>cp</sub>"],
}
COLOR_MODELS_TEMPLATES["mired"] = {
    "name": "Mired",
    "section": "Qualities",
    "ui_group": "qua",
    "description": (
        "A mired (Micro Reciprocal Degree) is a unit used in photography and "
        "cinematography to measure color temperature: 1,000,000 / "
        "Temperature in K = 1 Mired"
    ),
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "codes": ["MK<sup>-1</sup>"],
}
COLOR_MODELS_TEMPLATES["ciecam02"] = {
    "name": "CIECAM02",
    "section": "Color Appearance Models",
    "ui_group": "cam",
    "description": "The CAM published in 2002 by the CIE, and the successor of CIECAM97s",
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "defaults": {
        "L<sub>A</sub>": 318.31,
        "Y<sub>b</sub>": 20,
        "Surround Viewing Conditions": "average",
        "Discount Illuminant?": "No",
    },
    "labels": [
        "Correlate of Lightness J",
        "Correlate of chroma C",
        "Hue angle h in degrees",
        "Correlate of saturation s",
        "Correlate of brightness Q",
        "Correlate of colorfulness M",
        "Hue h quadrature H",
        "Hue h composition H<sup>C</sup>",
        "Hue h composition H<sup>C</sup>",
    ],
    "codes": ["J", "C", "h", "s", "Q", "M", "H", "HC"],
}
COLOR_MODELS_TEMPLATES["ciecam16"] = {
    "name": "CIECAM16",
    "section": "Color Appearance Models",
    "ui_group": "cam",
    "description": "",
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "defaults": {
        "L<sub>A</sub>": 318.31,
        "Y<sub>b</sub>": 20,
        "Surround Viewing Conditions": "average",
        "Discount Illuminant?": "No",
    },
    "labels": [
        "Correlate of Lightness J",
        "Correlate of chroma C",
        "Hue angle h in degrees",
        "Correlate of saturation s",
        "Correlate of brightness Q",
        "Correlate of colorfulness M",
        "Hue h quadrature H",
        "Hue h composition H<sup>C</sup>",
        "Hue h composition H<sup>C</sup>",
    ],
    "codes": ["J", "C", "h", "s", "Q", "M", "H", "HC"],
}
COLOR_MODELS_TEMPLATES["cielab"] = {
    "name": "CIELAB",
    "section": "CIE Fundamentals",
    "ui_group": "cie",
    "description": (
        "L* for perceptual lightness, a* for the magenta-green axis, and b* "
        "for blue-yellow axis"
    ),
    "illuminant": "all",
    "observer": "all",
    "labels": ["Lightness", "Redness-greenness", "Yellowness-blueness"],
    "codes": ["L*", "a*", "b*"],
}
COLOR_MODELS_TEMPLATES["cielchab"] = {
    "name": "CIE LCH<sub>ab</sub>",
    "section": "CIE Fundamentals",
    "ui_group": "cie",
    "description": "CIELAB expressed in cylinderical form",
    "illuminant": "all",
    "observer": "all",
    "labels": ["Lightness", "Chroma", "Hue"],
    "codes": ["L*", "C*", "H<sub>ab</sub>"],
}
COLOR_MODELS_TEMPLATES["cielchuv"] = {
    "name": "CIE LCH<sub>uv</sub>",
    "section": "CIE Fundamentals",
    "ui_group": "cie",
    "description": "CIELUV expressed in cylinderical form",
    "illuminant": "all",
    "observer": "all",
    "labels": ["Lightness", "Chroma", "Hue"],
    "codes": ["L", "C", "H<sub>uv</sub>"],
}
COLOR_MODELS_TEMPLATES["cieluv"] = {
    "name": "CIELUV",
    "section": "CIE Fundamentals",
    "ui_group": "cie",
    "description": (
        "A modification of CIEXYZ to display color differences more " "conveniently"
    ),
    "illuminant": "all",
    "observer": "all",
    "labels": ["Lightness", "Redness-greenness", "Yellowness-blueness"],
    "codes": ["L*", "u*", "v*"],
}
COLOR_MODELS_TEMPLATES["cieluvuv"] = {
    "name": "CIELUV uv",
    "section": "CIE Fundamentals",
    "ui_group": "cie",
    "description": "",
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Lightness", "Chroma", "Hue"],
    "codes": ["L*", "u*", "v*<sub>uv</sub>"],
}
COLOR_MODELS_TEMPLATES["cieucs"] = {
    "name": "CIE 1960 UCS",
    "section": "CIE Fundamentals",
    "ui_group": "cie",
    "description": "",
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "codes": ["U", "V", "W"],
}
COLOR_MODELS_TEMPLATES["cieucsuv"] = {
    "name": "CIE 1960 UCS uv",
    "section": "CIE Fundamentals",
    "ui_group": "cie",
    "description": "",
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "codes": ["u", "v"],
}
COLOR_MODELS_TEMPLATES["cieuvw"] = {
    "name": "CIE 1964 UVW",
    "section": "CIE Fundamentals",
    "ui_group": "cie",
    "description": (
        "CIE 1964 U*V*W* color spcace, invented in 1964 to calculate color "
        "differences without having to hold the luminance constant"
    ),
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "codes": ["U*", "V*", "W*"],
}
COLOR_MODELS_TEMPLATES["ciexyz"] = {
    "name": "CIEXYZ",
    "section": "CIE Fundamentals",
    "ui_group": "cie",
    "description": (
        "First attempt of a color space based on measurements of human "
        "perception, created in 1931"
    ),
    "illuminant": "all",
    "observer": "all",
    "codes": ["X", "Y", "Z"],
}
COLOR_MODELS_TEMPLATES["clrpurity"] = {
    "name": "Colorimetric Purity",
    "section": "Dominant Wavelength and Purity",
    "ui_group": "dom",
    "description": (
        "A color's dominant wavelength is the wavelength of monochromatic "
        "spectral light that evokes an identical perception of hue"
    ),
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "codes": ["P<sub>c</sub>"],
}
COLOR_MODELS_TEMPLATES["compwave"] = {
    "name": "Complementary Wavelength",
    "section": "Dominant Wavelength and Purity",
    "ui_group": "dom",
    "description": "",
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": [
        "Wavelength",
        "1st Intersection Coordinates",
        "2nd Intersection Coordinates",
    ],
    "codes": ["&gamma;<sub>c</sub>", "xy<sub>w</sub>l", "xy<sub>cw</sub>"],
}
COLOR_MODELS_TEMPLATES["din99"] = {
    "name": "DIN99",
    "section": "Similar to CIELAB",
    "ui_group": "lab",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Lightness", "Redness-greenness", "Yellowness-blueness"],
    "codes": ["L<sub>99</sub>", "a<sub>99</sub>", "b<sub>99</sub>"],
}
COLOR_MODELS_TEMPLATES["din99b"] = {
    "name": "DIN99b",
    "section": "Similar to CIELAB",
    "ui_group": "lab",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Lightness", "Redness-greenness", "Yellowness-blueness"],
    "codes": ["L<sub>99</sub>", "a<sub>99b</sub>", "b<sub>99b</sub>"],
}
COLOR_MODELS_TEMPLATES["din99c"] = {
    "name": "DIN99c",
    "section": "Similar to CIELAB",
    "ui_group": "lab",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Lightness", "Redness-greenness", "Yellowness-blueness"],
    "codes": ["L<sub>99</sub>", "a<sub>99c</sub>", "b<sub>99c</sub>"],
}
COLOR_MODELS_TEMPLATES["din99d"] = {
    "name": "DIN99d",
    "section": "Similar to CIELAB",
    "ui_group": "lab",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Lightness", "Redness-greenness", "Yellowness-blueness"],
    "codes": ["L<sub>99</sub>", "a<sub>99d</sub>", "b<sub>99d</sub>"],
}
COLOR_MODELS_TEMPLATES["domwave"] = {
    "name": "Dominant Wavelength",
    "section": "Dominant Wavelength and Purity",
    "ui_group": "dom",
    "description": "",
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": [
        "Wavelength",
        "1st Intersection Coordinates",
        "2nd Intersection Coordinates",
    ],
    "codes": ["&gamma;<sub>d</sub>", "xy<sub>w</sub>l", "xy<sub>cw</sub>"],
}
COLOR_MODELS_TEMPLATES["duv"] = {
    "name": "&Delta;<sub>uv</sub>",
    "section": "Light Source Quality",
    "ui_group": "spd",
    "description": "",
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Duv"],
    "codes": ["&Delta;<sub>uv</sub>"],
}
COLOR_MODELS_TEMPLATES["expurity"] = {
    "name": "Excitation Purity",
    "section": "Dominant Wavelength and Purity",
    "ui_group": "dom",
    "description": "",
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "codes": ["P<sub>e</sub>"],
}
COLOR_MODELS_TEMPLATES["hdrcielab"] = {
    "name": "hdr-CIELAB",
    "section": "Miscellaneous",
    "ui_group": "misc",
    "description": (
        "hdr-CIELAB was formulated by Fairchild and Chen (2011) to extend "
        "the CIELAB color space to function across high dynamic ranges"
    ),
    "illuminant": "all",
    "observer": "all",
    "defaults": {
        "Relative luminance of the surround Y<sub>s</sub>": 0.2,
        "Absolute luminance of the scene diffuse white Y<sub>abs</sub>": "100 cd/m<sup>2</sup>",
    },
    "labels": ["Lightness", "Redness-greenness", "Yellowness-blueness"],
    "codes": ["L<sub>hdr</sub>", "a<sub>hdr</sub>", "b<sub>hdr</sub>"],
}
COLOR_MODELS_TEMPLATES["hdript"] = {
    "name": "hdr-IPT",
    "section": "Miscellaneous",
    "ui_group": "misc",
    "description": (
        "hdr-IPT was created by Fairchild et al. (2011) to extend the IPT "
        "color space to function across high dynamic ranges"
    ),
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Intensity", "Blue-yellow Chroma", "Red-green Chroma"],
    "codes": ["I<sub>hdr</sub>", "P<sub>hdr</sub>", "T<sub>hdr</sub>"],
}
COLOR_MODELS_TEMPLATES["hellwig2022"] = {
    "name": "Hellwig and Fairchild (2022) CAM",
    "section": "Color Appearance Models",
    "ui_group": "cam",
    "description": "",
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "defaults": {
        "L<sub>A</sub>": 318.31,
        "Y<sub>b</sub>": 20,
        "Surround Viewing Conditions": "average",
        "Discount Illuminant?": "No",
    },
    "labels": [
        "Correlate of Lightness J",
        "Correlate of chroma C",
        "Hue angle h in degrees",
        "Correlate of saturation s",
        "Correlate of brightness Q",
        "Correlate of colorfulness M",
        "Hue h quadrature H",
        "Hue h composition H<sup>C</sup>",
        "Hue h composition H<sup>C</sup>",
        "Correlate of Lightness accounting for Helmholtz-Kohlrausch effect",
        "Correlate of brightness accounting for Helmholtz-Kohlrausch effect",
    ],
    "codes": [
        "J",
        "C",
        "h",
        "s",
        "Q",
        "M",
        "H",
        "HC",
        "J<sub>HK</sub>",
        "Q<sub>HK</sub>",
    ],
}
COLOR_MODELS_TEMPLATES["hunt"] = {
    "name": "Hunt",
    "section": "Color Appearance Models",
    "ui_group": "cam",
    "description": "",
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "defaults": {
        "Viewing Conditions": "Normal scenes",
        "L<sub>A</sub>": 318.31,
        "T<sub>cp</sub>": 6504,
        "L<sub>AS</sub>": "Approximated",
        "XYZ of Proximal Field": "Equal to background",
        "Simultaneous Contrast / Assimilation Factor p": "None",
        "Scotopic Response S": "Approximated using tristimulus values Y of the stimulus",
        "Scotopic Response for the Reference White S<sub>W</sub>": (
            "Approximated using the tristimulus values Y<sub>W</sub> of the reference white"
        ),
        "Account for the Helson-Judd Effect?": "No",
        "Discount Illuminant?": "Yes",
    },
    "labels": [
        "Correlate of Lightness J",
        "Correlate of chroma C",
        "Hue angle h in degrees",
        "Correlate of saturation s",
        "Correlate of brightness Q",
        "Correlate of colorfulness M",
        "Hue h quadrature H",
        "Hue h composition H<sup>C</sup>",
        "Hue h composition H<sup>C</sup>",
    ],
    "codes": ["J", "C", "h", "s", "Q", "M", "H", "HC"],
}

COLOR_MODELS_TEMPLATES["hunterkakb"] = {
    "name": "Hunter L, a, b K<sub>a</sub> K<sub>b</sub>",
    "section": "Similar to CIELAB",
    "ui_group": "lab",
    "description": "Chromaticity coordinates for the illuminant",
    "illuminant": "all",
    "observer": "all",
    "labels": ["Coefficient a", "Coefficient b"],
    "codes": ["K<sub>a</sub>", "K<sub>b</sub>"],
}
COLOR_MODELS_TEMPLATES["hunterlab"] = {
    "name": "Hunter L, a, b",
    "section": "Similar to CIELAB",
    "ui_group": "lab",
    "description": "",
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Lightness", "Redness-greenness", "Yellowness-blueness"],
    "codes": ["L", "a", "b"],
}
COLOR_MODELS_TEMPLATES["hunterrdab"] = {
    "name": "Hunter Rd, a, b",
    "section": "Similar to CIELAB",
    "ui_group": "lab",
    "description": "",
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Lightness", "Redness-greenness", "Yellowness-blueness"],
    "codes": ["Rd", "a", "b"],
}
COLOR_MODELS_TEMPLATES["iab"] = {
    "name": "Iab",
    "section": "",
    "description": "Similar to CIELAB",
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Lightness", "Redness-greenness", "Yellowness-blueness"],
    "codes": ["I", "a", "b"],
}
COLOR_MODELS_TEMPLATES["icacb"] = {
    "name": "IC<sub>A</sub>C<sub>B</sub>",
    "section": "Similar to CIELAB",
    "ui_group": "lab",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["", "", ""],
    "codes": ["I", "C<sub>A</sub>", "C<sub>B</sub>"],
}
COLOR_MODELS_TEMPLATES["ictcp_2100_1_hlg"] = {
    "name": "ITU-R BT.2100-1 HLG",
    "section": "Miscellaneous",
    "ui_group": "misc",
    "description": (
        "Recommendation ITU-R BT.2100 HLG (Hybrid Log-Gamma); "
        "IC<sub>T</sub>C<sub>P</sub> values for high dynamic range "
        "television for use in production and international program "
        "exchange"
    ),
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Intensity", "Blue-yellow Chroma", "Red-green Chroma"],
    "codes": ["I", "C<sub>T</sub>", "C<sub>P</sub>"],
}
COLOR_MODELS_TEMPLATES["ictcp_2100_1_pq"] = {
    "name": "ITU-R BT.2100-1 PQ",
    "section": "Miscellaneous",
    "ui_group": "misc",
    "description": (
        "Recommendation ITU-R BT.2100 PQ (Perceptual Quantizer); "
        "IC<sub>T</sub>C<sub>P</sub> values for high dynamic range "
        "television for use in production and international program "
        "exchange"
    ),
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Intensity", "Blue-yellow Chroma", "Red-green Chroma"],
    "codes": ["I", "C<sub>T</sub>", "C<sub>P</sub>"],
}
COLOR_MODELS_TEMPLATES["ictcp_2100_2_hlg"] = {
    "name": "ITU-R BT.2100-2 HLG",
    "section": "Miscellaneous",
    "ui_group": "misc",
    "description": (
        "Recommendation ITU-R BT.2100 HLG (Hybrid Log-Gamma); "
        "IC<sub>T</sub>C<sub>P</sub> values for high dynamic range "
        "television for use in production and international program "
        "exchange"
    ),
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Intensity", "Blue-yellow Chroma", "Red-green Chroma"],
    "codes": ["I", "C<sub>T</sub>", "C<sub>P</sub>"],
}
COLOR_MODELS_TEMPLATES["ictcp_2100_2_pq"] = {
    "name": "ITU-R BT.2100-2 PQ",
    "section": "Miscellaneous",
    "ui_group": "misc",
    "description": (
        "Recommendation ITU-R BT.2100 PQ (Perceptual Quantizer); "
        "IC<sub>T</sub>C<sub>P</sub> values for high dynamic range "
        "television for use in production and international program "
        "exchange"
    ),
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Intensity", "Blue-yellow Chroma", "Red-green Chroma"],
    "codes": ["I", "C<sub>T</sub>", "C<sub>P</sub>"],
}
COLOR_MODELS_TEMPLATES["igpgtg"] = {
    "name": "I<sub>G</sub>P<sub>G</sub>T<sub>G</sub>",
    "section": "Similar to CIELAB",
    "ui_group": "lab",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": [
        "Intensity (using Gaussian spectra)",
        "Blue-yellow Chroma (using Gaussian spectra)",
        "Red-green Chroma (using Gaussian spectra)",
    ],
    "codes": ["I<sub>G</sub>", "P<sub>G</sub>", "T<sub>G</sub>"],
}
COLOR_MODELS_TEMPLATES["ipt"] = {
    "name": "IPT",
    "section": "Similar to CIELAB",
    "ui_group": "lab",
    "description": "Image Processing Transform",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Intensity", "Blue-yellow Chroma", "Red-green Chroma"],
    "codes": ["I", "P", "T"],
}
COLOR_MODELS_TEMPLATES["ipt_hue"] = {
    "name": "IPT Hue Angle",
    "section": "Similar to CIELAB",
    "ui_group": "lab",
    "description": "Image Processing Transform hue angle in degrees",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Hue"],
    "codes": ["Hue"],
}
COLOR_MODELS_TEMPLATES["iptragoo"] = {
    "name": "IPT - Ragoo and Farup (2021)",
    "section": "Similar to CIELAB",
    "ui_group": "lab",
    "description": "",
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Intensity", "Blue-yellow Chroma", "Red-green Chroma"],
    "codes": ["I", "P", "T"],
}
COLOR_MODELS_TEMPLATES["jmhciecam02"] = {
    "name": "CIECAM02 JMh",
    "section": "Color Appearance Models",
    "ui_group": "cam",
    "description": "",
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Lightness", "Colorfulness", "Hue"],
    "codes": ["J", "M", "h"],
}
COLOR_MODELS_TEMPLATES["jmhciecam16"] = {
    "name": "CIECAM16 JMh",
    "section": "Color Appearance Models",
    "ui_group": "cam",
    "description": "",
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Lightness", "Colorfulness", "Hue"],
    "codes": ["J", "M", "h"],
}
COLOR_MODELS_TEMPLATES["jmhcam16"] = {
    "name": "CAM16 JMh",
    "section": "Color Appearance Models",
    "ui_group": "cam",
    "description": "",
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Lightness", "Colorfulness", "Hue"],
    "codes": ["J", "M", "h"],
}
COLOR_MODELS_TEMPLATES["jmhhellwig2022"] = {
    "name": "Hellwig and Fairchild (2022) CAM JMh",
    "section": "Color Appearance Models",
    "ui_group": "cam",
    "description": "",
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "defaults": "Viewing Conditions: average",
    "labels": ["Lightness", "Colorfulness", "Hue"],
    "codes": ["J", "M", "h"],
}
COLOR_MODELS_TEMPLATES["izazbz"] = {
    "name": "I<sub>Z</sub>a<sub>Z</sub>B<sub>Z</sub>",
    "section": "Similar to CIELAB",
    "ui_group": "lab",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Achromatic response", "Redness-greenness", "Yellowness-blueness"],
    "codes": ["I<sub>Z</sub>", "a<sub>Z</sub>", "b<sub>Z</sub>"],
}
COLOR_MODELS_TEMPLATES["jzazbz"] = {
    "name": "J<sub>Z</sub>a<sub>Z</sub>B<sub>Z</sub>",
    "section": "Similar to CIELAB",
    "ui_group": "lab",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Lightness", "Redness-greenness", "Yellowness-blueness"],
    "codes": ["J<sub>Z</sub>", "a<sub>Z</sub>", "b<sub>Z</sub>"],
}
COLOR_MODELS_TEMPLATES["kim2009"] = {
    "name": "Kim, Weyrich and Kautz (2009)",
    "section": "Color Appearance Models",
    "ui_group": "cam",
    "description": "",
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "defaults": {
        "L<sub>A</sub>": 318.31,
        "Viewing Conditions": "average",
        "Media Parameters": "CRT displays",
        "Cone Response Sigmoidal Curve Modulating Factor n<sub>c</sub>": 0.57,
    },
    "labels": [
        "Correlate of Lightness J",
        "Correlate of chroma C",
        "Hue angle h in degrees",
        "Correlate of saturation s",
        "Correlate of brightness Q",
        "Correlate of colorfulness M",
        "Hue h quadrature H",
        "Hue h composition H<sup>C</sup>",
        "Hue h composition H<sup>C</sup>",
    ],
    "codes": ["J", "C", "h", "s", "Q", "M", "H", "HC"],
}
COLOR_MODELS_TEMPLATES["lightness"] = {
    "name": "Lightness",
    "section": "Qualities",
    "ui_group": "qua",
    "description": "",
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Lightness"],
    "codes": ["L"],
}
COLOR_MODELS_TEMPLATES["llab"] = {
    "name": "LLAB(l:c)",
    "section": "Color Appearance Models",
    "ui_group": "cam",
    "description": "",
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "defaults": (
        "Viewing Conditions: Reference samples and images, average surround, "
        "subtending < 4"
    ),
    "labels": [
        "Correlate of Lightness L<sub>L</sub>",
        "Correlate of chroma Ch<sub>L</sub>",
        "Hue angle h<sub>L</sub> in degrees",
        "Correlate of saturation s<sub>L</sub>",
        "Correlate of colorfulness C<sub>L</sub>",
        "Hue h composition H<sup>C</sup>",
        "Opponent signal A<sub>L</sub>",
        "Opponent signal B<sub>L</sub>",
    ],
    "codes": ["J", "C", "h", "s", "M", "HC", "a", "b"],
}
COLOR_MODELS_TEMPLATES["luminance"] = {
    "name": "Luminance",
    "section": "Qualities",
    "ui_group": "qua",
    "description": "",
    "illuminant": "none",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Luminance"],
    "codes": ["Y"],
}
COLOR_MODELS_TEMPLATES["munsell_clr"] = {
    "name": "Munsell Color",
    "section": "Miscellaneous",
    "ui_group": "misc",
    "description": (
        "Devised in 1905 by Albert Munsell, the Munsell color system "
        "specifies colors based on three properties: hue, chroma, and value"
    ),
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Hue Chroma/Value"],
    "codes": ["Color"],
}
COLOR_MODELS_TEMPLATES["munsell_value"] = {
    "name": "Munsell Value",
    "section": "Miscellaneous",
    "ui_group": "misc",
    "description": "",
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Value"],
    "codes": ["MV"],
}
COLOR_MODELS_TEMPLATES["nayatani95"] = {
    "name": "Nayatani (1995)",
    "section": "Color Appearance Models",
    "ui_group": "cam",
    "description": "",
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "defaults": "E<sub>O</sub>: 5000.0; E<sub>OR</sub>: 1000; Y<sub>O</sub>: 20",
    "labels": [
        "Correlate of achromatic Lightness L*<sub>p&Star;</sub>",
        "Correlate of chroma C",
        "Hue angle &theta; in degrees",
        "Correlate of saturation S",
        "Correlate of brightness B<sub>r&Star;</sub>",
        "Correlate of colorfulness M",
        "Hue h quadrature H",
        "Hue h composition H<sup>C</sup>",
        "Hue h composition H<sup>C</sup>",
    ],
    "codes": [
        "L*<sub>p&Star;</sub>",
        "C",
        "h",
        "s",
        "Q",
        "M",
        "H",
        "HC",
        "L*<sub>n&Star;</sub>",
    ],
}
COLOR_MODELS_TEMPLATES["oklab"] = {
    "name": "Oklab",
    "section": "Similar to CIELAB",
    "ui_group": "lab",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Lightness", "Greenness-Redness", "Blueness-yellowness"],
    "codes": ["L", "a", "b"],
}
COLOR_MODELS_TEMPLATES["osaucs"] = {
    "name": "OSA UCS",
    "section": "Similar to CIELAB",
    "ui_group": "lab",
    "description": "",
    "illuminant": "all",
    "observer": "CIE 1964 10 Degree Standard Observer",
    "labels": ["Lightness", "jaune (yellowness)", "greenness"],
    "codes": ["L", "j", "g"],
}
COLOR_MODELS_TEMPLATES["prolab"] = {
    "name": "ProLab",
    "section": "Similar to CIELAB",
    "ui_group": "lab",
    "description": "",
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Lightness", "Greenness-redness", "Blueness-yellowness"],
    "codes": ["L", "a", "b"],
}
COLOR_MODELS_TEMPLATES["rlab"] = {
    "name": "RLAB",
    "section": "Color Appearance Models",
    "ui_group": "cam",
    "description": "",
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "defaults": (
        "Viewing Conditions: average; Discounting the "
        "Illuminant Factor: hard copy images"
    ),
    "labels": [
        "Correlate of Lightness L<sup>R</sup>",
        "Correlate of achromatic chroma C<sup>R</sup>",
        "Hue angle h<sup>R</sup> in degrees",
        "Correlate of saturation s<sup>R</sup>",
        "Hue h composition H<sup>C</sup>",
        "Red-green chromatic response a<sup>R</sup>",
        "Yellow-blue chromatic response b<sup>R</sup>",
    ],
    "codes": ["J", "C", "h", "s", "HC", "a", "b"],
}
COLOR_MODELS_TEMPLATES["wavelength"] = {
    "name": "Wavelength",
    "section": "",
    "description": "",
    "illuminant": "all",
    "observer": "all",
    "labels": ["Wavelength (nm)"],
    "codes": ["&gamma;"],
}
COLOR_MODELS_TEMPLATES["whiteness_stensby"] = {
    "name": "Whiteness Index (Stensby)",
    "section": "Qualities",
    "ui_group": "qua",
    "description": (
        "Values larger than 100 indicate a bluish-white and values smaller "
        "than 100 indicate a yellowish-white"
    ),
    "illuminant": "all",
    "observer": "all",
    "labels": ["Whiteness Index"],
    "codes": ["WI"],
}
COLOR_MODELS_TEMPLATES["whiteness_berger"] = {
    "name": "Whiteness Index (Berger)",
    "section": "Qualities",
    "ui_group": "qua",
    "description": (
        "Indices larger than 33.33 indicate a bluish-white and values "
        "smaller than 33.33 indicate a yellowish-white"
    ),
    "illuminant": "all",
    "observer": "all",
    "labels": ["Whiteness Index"],
    "codes": ["WI"],
}
COLOR_MODELS_TEMPLATES["whiteness_cie"] = {
    "name": "Whiteness Index (CIE 2004)",
    "section": "Qualities",
    "ui_group": "qua",
    "description": (
        "W increases with whiteness, reaching 100 for the perfect diffuser. "
        "T is the tint index; a positive value indicates a greener tint, "
        "while a negative value indicates a redder tint"
    ),
    "illuminant": "D65",
    "observer": "all",
    "labels": ["Whiteness Index"],
    "codes": ["W"],
}
COLOR_MODELS_TEMPLATES["whiteness_e313"] = {
    "name": "Whiteness Index (E313)",
    "section": "Qualities",
    "ui_group": "qua",
    "description": "Whiteness index of using the ASTM E313 method",
    "illuminant": "all",
    "observer": "all",
    "labels": ["Whiteness Index"],
    "codes": ["WI"],
}
COLOR_MODELS_TEMPLATES["tintindex_cie"] = {
    "name": "Tint Index",
    "section": "Qualities",
    "ui_group": "qua",
    "description": (
        "A positive value indicates a greener tint, while a negative value "
        "indicates a redder tint"
    ),
    "illuminant": "all",
    "observer": "all",
    "labels": ["Tint Index"],
    "codes": ["T"],
}
COLOR_MODELS_TEMPLATES["xy"] = {
    "name": "xy",
    "section": "CIE Fundamentals",
    "ui_group": "cie",
    "description": "",
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["", "", ""],
    "codes": ["x", "y"],
}
COLOR_MODELS_TEMPLATES["xyY"] = {
    "name": "CIE xyY",
    "section": "CIE Fundamentals",
    "ui_group": "cie",
    "description": "",
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["", "", ""],
    "codes": ["x", "y", "Y"],
}
COLOR_MODELS_TEMPLATES["yellowness_d1925"] = {
    "name": "Yellowness Index (D1925)",
    "section": "Qualities",
    "ui_group": "qua",
    "description": (
        "ASTM D1925 was developed for the definition of the yellowness of "
        "homogeneous, non-fluorescent, almost neutral-transparent, "
        "white-scattering or opaque plastics as reviewed under daylight "
        "conditions"
    ),
    "illuminant": "C",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "codes": ["YI"],
}
COLOR_MODELS_TEMPLATES["yellowness_e313"] = {
    "name": "Yellowness Index (E313)",
    "section": "Qualities",
    "ui_group": "qua",
    "description": (
        "ASTM E313 has successfully been used for a variety of white or near "
        "white materials, including coatings, plastics, and textiles"
    ),
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "codes": ["YI"],
}
COLOR_MODELS_TEMPLATES["yrg"] = {
    "name": "Yrg",
    "section": "Miscellaneous",
    "ui_group": "misc",
    "description": (
        "A luminance-chroma space [Kirk, 2019] that fits well with visual "
        "color contrasts as represented by the Munsell color book for "
        "illuminants in the D50-D65 range"
    ),
    "illuminant": ["all"],
    "observer": "CIE 1931 2 Degree Standard Observer",
    "labels": ["Luminance", "redness", "greenness"],
    "codes": ["Y", "r", "g"],
}
COLOR_MODELS_TEMPLATES["zcam"] = {
    "name": "ZCAM",
    "section": "Color Appearance Models",
    "ui_group": "cam",
    "description": "",
    "illuminant": "all",
    "observer": "CIE 1931 2 Degree Standard Observer",
    "defaults": "Viewing Conditions: average",
    "labels": [
        "Lightness",
        "Chroma",
        "Hue Angle",
        "Saturation",
        "Brightness",
        "Colorfulness",
        "Hue h quadrature H<sub>C</sub>",
        "Hue h composition H<sup>C</sup>",
        "Vividness",
        "Blackness",
        "Whiteness",
    ],
    "codes": ["J", "C", "h", "s", "Q", "M", "H", "HC", "V", "K", "W"],
}
# VOLUME CHECKS
COLOR_MODELS_TEMPLATES["pointer"] = {
    "name": "Within Pointer's Gamut?",
    "section": "Miscellaneous",
    "ui_group": "misc",
    "description": (
        "States whether the associated CIEXYZ values are within Pointer’s "
        "Gamut volume, the gamut of real world surface colors with diffuse "
        "reflection [Pointer, 1980]"
    ),
    "illuminant": "none",
}
COLOR_MODELS_TEMPLATES["macadam"] = {
    "name": "Within MacAdam Limits?",
    "section": "Miscellaneous",
    "ui_group": "misc",
    "description": (
        "States whether the associated xyY values are within MacAdam limits "
        "of the given illuminant (A, C, or D65)"
    ),
    "illuminant": ["A", "C", "D65"],
    "observer": "CIE 1931 2 Degree Standard Observer",
}
# CSS 4 VALUES
COLOR_MODELS_TEMPLATES["a98_rgb_css_color"] = {
    "name": "a98 RGB CSS 4",
    "section": "CSS Color Module Level 4 Styles",
    "ui_group": "css",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
}
COLOR_MODELS_TEMPLATES["cielab_css_color"] = {
    "name": "CIELAB CSS 4",
    "section": "CSS Color Module Level 4 Styles",
    "ui_group": "css",
    "description": "",
    "illuminant": "D50",
    "observer": "CIE 1931 2 Degree Standard Observer",
}
COLOR_MODELS_TEMPLATES["cielchab_css_color"] = {
    "name": "CIELCHab CSS 4",
    "section": "CSS Color Module Level 4 Styles",
    "ui_group": "css",
    "description": "",
    "illuminant": "D50",
    "observer": "CIE 1931 2 Degree Standard Observer",
}
COLOR_MODELS_TEMPLATES["ciexyz_css_color"] = {
    "name": "CIEXYZ CSS 4",
    "section": "CSS Color Module Level 4 Styles",
    "ui_group": "css",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
}
COLOR_MODELS_TEMPLATES["ciexyz_d50_css_color"] = {
    "name": "CIEXYZ D50 CSS 4",
    "section": "CSS Color Module Level 4 Styles",
    "ui_group": "css",
    "description": "",
    "illuminant": "D50",
    "observer": "CIE 1931 2 Degree Standard Observer",
}
COLOR_MODELS_TEMPLATES["ciexyz_d65_css_color"] = {
    "name": "CIEXYZ D65 CSS 4 Color Module Level 4",
    "section": "CSS Color Module Level 4 Styles",
    "ui_group": "css",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
}
COLOR_MODELS_TEMPLATES["display_p3_css_color"] = {
    "name": "Display P3 CSS 4",
    "section": "CSS Color Module Level 4 Styles",
    "ui_group": "css",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
}
COLOR_MODELS_TEMPLATES["hsl_css_color"] = {
    "name": "Hue-Saturation-Lightness CSS 4",
    "section": "CSS Color Module Level 4 Styles",
    "ui_group": "css",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
}
COLOR_MODELS_TEMPLATES["hwb_css_color"] = {
    "name": "HWB CSS 4",
    "section": "CSS Color Module Level 4 Styles",
    "ui_group": "css",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
}
COLOR_MODELS_TEMPLATES["linear_srgb_css_color"] = {
    "name": "Linear sRGB CSS 4",
    "section": "CSS Color Module Level 4 Styles",
    "ui_group": "css",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
}
COLOR_MODELS_TEMPLATES["oklab_css_color"] = {
    "name": "Oklab CSS 4",
    "section": "CSS Color Module Level 4 Styles",
    "ui_group": "css",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
}
COLOR_MODELS_TEMPLATES["prophoto_rgb_css_color"] = {
    "name": "ProPhoto RGB CSS 4",
    "section": "CSS Color Module Level 4 Styles",
    "ui_group": "css",
    "description": "",
    "illuminant": "D50",
    "observer": "CIE 1931 2 Degree Standard Observer",
}
COLOR_MODELS_TEMPLATES["rec_2020_css_color"] = {
    "name": "ITU-R BT.2020 CSS 4",
    "section": "CSS Color Module Level 4 Styles",
    "ui_group": "css",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
}
COLOR_MODELS_TEMPLATES["srgb_css_color"] = {
    "name": "sRGB CSS 4",
    "section": "CSS Color Module Level 4 Styles",
    "ui_group": "css",
    "description": "",
    "illuminant": "D65",
    "observer": "CIE 1931 2 Degree Standard Observer",
}
# ONLY FOR *EMISSIVE* SPECTRAL DISTRIBUTIONS
COLOR_MODELS_TEMPLATES["cri"] = {
    "name": "CRI",
    "section": "Light Source Quality",
    "ui_group": "spd",
    "description": (
        "Color Rendering Index is a quantitative measure of the ability of a "
        "light source to reveal the colors of various objects accurately in "
        "comparison with a natural or standard light source; devised by the "
        "CIE"
    ),
    "illuminant": "all",
    "codes": ["R<sub>a</sub>"],
}
COLOR_MODELS_TEMPLATES["cfi"] = {
    "name": "CFI",
    "section": "Light Source Quality",
    "ui_group": "spd",
    "description": (
        "The general Color Fidelity Index represents how closely the color "
        "appearances of the entire sample set are reproduced on average by a "
        "test light as compared to those under a reference illuminant"
    ),
    "illuminant": "all",
    "codes": ["R<sub>f</sub>"],
}
COLOR_MODELS_TEMPLATES["cqs"] = {
    "name": "CQS",
    "section": "Light Source Quality",
    "ui_group": "spd",
    "description": (
        "Color Quality Scale, a quantitative measure of the ability of a "
        "light source to reproduce colors of illuminated objects"
    ),
    "illuminant": "all",
    "codes": ["Q<sub>a</sub>"],
}
COLOR_MODELS_TEMPLATES["lum_efficacy"] = {
    "name": "Luminous Efficacy",
    "section": "Light Source Quality",
    "ui_group": "spd",
    "description": (
        "The quotient of luminous flux divided by the total radiant flux, "
        "using the efficiency function of the CIE 1924 Photopic Standard "
        "Observer"
    ),
    "illuminant": "all",
    "codes": ["K"],
    "units": ["  lm⋅W<sup>−1</sup>"],
}
COLOR_MODELS_TEMPLATES["lum_efficiency"] = {
    "name": "Luminous Efficiency",
    "section": "Light Source Quality",
    "ui_group": "spd",
    "description": (
        "The ratio of the total luminous flux to the total radiant flux of "
        "an emitting source, using the efficiency function of the CIE 1924 "
        "Photopic Standard Observer"
    ),
    "illuminant": "all",
}
COLOR_MODELS_TEMPLATES["lum_flux"] = {
    "name": "Luminous Flux",
    "section": "Light Source Quality",
    "ui_group": "spd",
    "description": (
        "The time rate of flow of radiant energy, evaluated in terms of "
        "standardized photopic vision"
    ),
    "illuminant": "all",
    "codes": ["&Phi;<sub>v</sub>"],
    "units": ["lm"],
}
COLOR_MODELS_TEMPLATES["ssi"] = {
    "name": "SSI",
    "section": "Light Source Quality",
    "ui_group": "spd",
    "description": "",
    "illuminant": "all",
    "codes": ["SSI"],
}
# SPECTRAL RECOVERY TO D65 2 degree
COLOR_MODELS_TEMPLATES["sr_sd"] = {
    "name": "Recovered Spectrum",
    "section": "Spectral Power Distribution",
    "ui_group": "spd",
    "description": (
        "Recovered the spectral distribution and adapted to CIEXYZ D65 values "
        "with the CIE 1931 2 Degree Standard Observer, using the Jakob and "
        "Hanika (2019) method"
    ),
    "illuminant": "D65",
    "codes": ["SPD"],
}


class NumpyEncoder(json.JSONEncoder):
    """Special JSON encoder for NumPy types"""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        else:
            return super().default(o)


def space_separated(lst):
    """return space separated values from list"""
    return " ".join(map(str, lst))


def space_separated_nosci(lst):
    """return space separated values from list and suppress scientific notation"""
    return " ".join(f"{x:.5f}" for x in lst)


def denormalize(a, *factors):
    """denormalize array of coordinates using different factors when needed (e.g. HSV)"""
    precision = 3
    if len(factors) == 0:
        factors = [1]

    # Extend factors to match the length of a if needed
    factors = list(factors) + [factors[0]] * (len(a) - len(factors))

    # Convert to NumPy arrays
    a = np.array(a)
    factors = np.array(factors)

    # Use NumPy's vectorized multiplication
    denormalized_values = np.round(a * factors, precision)

    return denormalized_values


def normalize(a, b, c, factor):
    """Normalize 3 coordinates to 0-1 using factor derived from bitdepth"""
    return np.array([a, b, c]) / factor


class SetToNone(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values == ["None"]:
            setattr(namespace, self.dest, None)
        else:
            setattr(namespace, self.dest, values)


class MyParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_help()
        sys.stderr.write("Error: %s\n" % message)
        sys.exit(2)


def determine_color_model(args):
    """Determine the color model based on provided arguments."""
    if args.lstar is not None and args.astar is not None and args.bstar is not None:
        return "CIELAB"
    if (
        args.lchab_l_val is not None
        and args.lchab_ch_val is not None
        and args.lchab_ab_val is not None
    ):
        return "CIELCHab"
    if args.l_val is not None and args.u_val is not None and args.v_val is not None:
        return "CIELUV"
    if (
        args.lchuv_l_val is not None
        and args.lchuv_ch_val is not None
        and args.lchuv_uv_val is not None
    ):
        return "CIELCHuv"
    if args.x_val is not None and args.y_val is not None and args.z_val is not None:
        return "CIEXYZ"
    if args.red is not None and args.green is not None and args.blue is not None:
        return "RGB"
    if (
        args.start is not None
        and args.stop is not None
        and args.interval is not None
        and args.data is not None
    ):
        return "Spectrum"
    if args.wave is not None:
        return "Wavelength"
    return None


def parse_arguments():
    """Set up argument parser and define command line arguments"""
    parser = MyParser(usage="")

    # if this argument is present, show the Python and Colour warnings
    parser.add_argument(
        "--show_warnings",
        action="store_true",
        help="If given, show Python and Colour warnings.",
    )

    # 8 bpc (0-255), 15+1 (0-32768), 16 bpc (0-65535), 32 bpc (0-1)
    parser.add_argument(
        "--bitdepth",
        type=str,
        choices=["8", "15+1", "16", "32"],
        help="Bit depth of input values",
        default="8",
    )

    # precalculated data
    parser.add_argument(
        "--precalc", type=str, help="Precalculated data in JSON format", default=""
    )

    # STANDARD OBSERVER
    parser.add_argument(
        "--observer",
        type=str,
        choices=[
            "CIE 1931 2 Degree Standard Observer",
            "CIE 1964 10 Degree Standard Observer",
        ],
        help="Standard Observer",
        default="CIE 1931 2 Degree Standard Observer",
    )

    # ILLUMINANT OF INPUT VALUES
    parser.add_argument(
        "--input_illuminant",
        type=str,
        choices=CIE_ILLUMINANTS_LIST,
        help="Standard illuminant of input values",
        default="D65",
    )

    # WHICH CHROMATIC ADAPTATION TRANSFORMS TO USE
    parser.add_argument(
        "--cat",
        type=str,
        nargs="*",
        choices=CHROMATIC_ADAPTATION_TRANSFORMS + ["None"],
        default=CHROMATIC_ADAPTATION_TRANSFORMS,
        action=SetToNone,
        help="Chromatic Adaptation Transform",
    )

    # WHICH ILLUMINANTS TO USE
    parser.add_argument(
        "--illuminant_list", type=str, choices=["CIE", "ISO_7589", "All"], default="All"
    )

    # output spectral plot path
    parser.add_argument(
        "--plotpath",
        type=str,
        help=(
            "Output spectral plot path; plot is not saved if argument is omitted. "
            "SVG extension is added if not provided"
        ),
    )

    # CIELAB ARGUMENTS
    cielab = parser.add_argument_group("CIELAB")
    cielab.add_argument("--lstar", type=float, help="Lightness value, 0-100")
    cielab.add_argument("--astar", type=float, help="a* value, any bit depth")
    cielab.add_argument("--bstar", type=float, help="b* value, any bit depth")

    # CIELCHab ARGUMENTS (cylinderical CIELAB)
    cielchab = parser.add_argument_group("CIELCHab")
    cielchab.add_argument("--lchab_l_val", type=float, help="Lightness value, 0-100")
    cielchab.add_argument(
        "--lchab_ch_val", type=float, help="Chroma value, any bit depth"
    )
    cielchab.add_argument("--lchab_ab_val", type=float, help="ab value, any bit depth")

    # CIELUV ARGUMENTS
    cieluv = parser.add_argument_group("CIELUV")
    cieluv.add_argument(
        "--l_val", type=float, help="Lightness value, any bit depth, 0-100"
    )
    cieluv.add_argument("--u_val", type=float, help="u* value, any bit depth")
    cieluv.add_argument("--v_val", type=float, help="v* value, any bit depth")

    # CIELCHuv ARGUMENTS (cylinderical CIELUV)
    cielchuv = parser.add_argument_group("CIELCHuv")
    cielchuv.add_argument("--lchuv_l_val", type=float, help="Lightness value, 0-100")
    cielchuv.add_argument(
        "--lchuv_ch_val", type=float, help="Chroma value, any bit depth"
    )
    cielchuv.add_argument("--lchuv_uv_val", type=float, help="uv value, any bit depth")

    # CIEXYZ ARGUMENTS
    ciexyz = parser.add_argument_group("CIEXYZ")
    ciexyz.add_argument(
        "--x_val",
        type=float,
        help="X value, any bit depth, a mix of all three cone responses, 0-1",
    )
    ciexyz.add_argument(
        "--y_val",
        type=float,
        help="Y value, luminance, any bit depth, a mix of L and M responses, 0-1",
    )
    ciexyz.add_argument(
        "--z_val",
        type=float,
        help=(
            "Z value, similar to Blue in CIERGB, solely made up of the S cone "
            "response, 0-1"
        ),
    )

    # RGB ARGUMENTS
    rgb = parser.add_argument_group("RGB")
    rgb.add_argument("--red", type=float, help="Red value, any bit depth")
    rgb.add_argument("--green", type=float, help="Green value, any bit depth")
    rgb.add_argument("--blue", type=float, help="Blue value, any bit depth")
    # color space name from colour-science lib
    rgb.add_argument(
        "--input_colorspace",
        type=str,
        help="RGB color space of input values",
        default="sRGB",
    )

    # SPECTRAL ARGUMENTS
    spectrum = parser.add_argument_group("SPECTRUM")
    spectrum.add_argument(
        "--start", type=int, help="Start of distribution, between 360-790 nm"
    )
    spectrum.add_argument(
        "--stop", type=int, help="End of distribution, between 400-830 nm"
    )
    spectrum.add_argument(
        "--interval", type=int, help="Interval between measurements, in nm, 1-20"
    )
    spectrum.add_argument(
        "--spectype", type=str, choices=["Emissive", "Reflective", "Transmissive"]
    )
    spectrum.add_argument(
        "--data",
        nargs="+",
        type=float,
        help=(
            "List of decimal values (preferrably to 6+ places), with spaces in "
            "between, unquoted, unbracketed, 0-1"
        ),
    )
    spectrum.add_argument(
        "--intrpl_intrvl",
        type=int,
        help="Smaller interval to interpolate to, between 1-5 nm",
    )
    spectrum.add_argument(
        "--tm30path",
        type=str,
        help=(
            "Path and filename for IES TM-30 Color Rendition Report; actual or "
            "interpolated interval must be 1-5 nm"
        ),
    )
    spectrum.add_argument(
        "--tm30format",
        type=str,
        choices=["Full", "Intermediate", "Simple"],
        help="Format for IES TM-30 Color Rendition Report",
    )

    # WAVELENGTH ARGUMENT
    wavelength = parser.add_argument_group("WAVELENGTH")
    wavelength.add_argument("--wave", type=float, help="Wavelength in nm")

    args = parser.parse_args()

    if args.plotpath is not None and args.plotpath[-4:] != ".svg":
        args.plotpath = args.plotpath + ".svg"

    model = determine_color_model(args)

    if model is None:
        parser.error("Full set of input values for one of the color models is required")

    return args, model


def validate_rgb_values(bitdepth, red, green, blue):
    """Check that RGB values are within the range of the chosen bit depth and
    additionally calculate normalization factor from bitdepth."""
    if bitdepth not in BITDEPTH_FACTOR:
        raise ValueError(f"Invalid bit depth specified: {bitdepth}")

    max_val = BITDEPTH_FACTOR[bitdepth]

    red, green, blue = float(red), float(green), float(blue)

    if not (0 <= red <= max_val and 0 <= green <= max_val and 0 <= blue <= max_val):
        raise ValueError(
            f"RGB values are out of range for the specified bit depth: {bitdepth}"
        )

    return red, green, blue, max_val


def validate_cielab_values(bitdepth, lstar, astar, bstar):
    """Check that CIELAB values are within the range of the chosen bit depth and
    additionally calculate normalization factor from bitdepth."""

    if bitdepth not in CIELAB_DEPTH_VALUES:
        raise ValueError(f"Invalid bit depth specified: {bitdepth}")

    min_val_lstar, max_val_lstar = CIELAB_DEPTH_VALUES[bitdepth]["L"]
    min_val_a_b_star, max_val_a_b_star = CIELAB_DEPTH_VALUES[bitdepth]["ab"]

    lstar, astar, bstar = float(lstar), float(astar), float(bstar)

    if not (
        min_val_lstar <= lstar <= max_val_lstar
        and min_val_a_b_star <= astar <= max_val_a_b_star
        and min_val_a_b_star <= bstar <= max_val_a_b_star
    ):
        raise ValueError(
            f"CIELAB values are out of range for the specified bit depth: {bitdepth}"
        )

    factor = BITDEPTH_FACTOR[bitdepth]
    return lstar, astar, bstar, factor


def validate_cielchab_values(bitdepth, lchab_l_val, lchab_ch_val, lchab_ab_val):
    """Check that CIELCHab values are within range and additionally calculate
    normalization factor from bitdepth."""

    if bitdepth not in BITDEPTH_FACTOR:
        raise ValueError(f"Invalid bit depth specified: {bitdepth}")

    value_ranges = {
        "lchab_l_val": (0, 100),
        "lchab_ch_val": (0, 230),
        "lchab_ab_val": (0, 360),
    }
    values = {
        "lchab_l_val": float(lchab_l_val),
        "lchab_ch_val": float(lchab_ch_val),
        "lchab_ab_val": float(lchab_ab_val),
    }
    for val_name, (min_val, max_val) in value_ranges.items():
        if not min_val <= values[val_name] <= max_val:
            raise ValueError(f"{val_name} is out of range")

    factor = BITDEPTH_FACTOR[bitdepth]
    return values["lchab_l_val"], values["lchab_ch_val"], values["lchab_ab_val"], factor


def validate_cieluv_values(bitdepth, l_val, u_val, v_val):
    """Check that CIELUV values are within the range of the chosen bit depth and
    additionally calculate normalization factor from bitdepth."""

    if bitdepth not in BITDEPTH_FACTOR:
        raise ValueError(f"Invalid bit depth specified: {bitdepth}")

    value_ranges = {
        "8": {"l_val": (0, 100), "u_v_val": (-200, 200)},
        "15+1": {"l_val": (0, 32768), "u_v_val": (-25600, 25600)},
        "16": {"l_val": (0, 65535), "u_v_val": (-51200, 51200)},
        "32": {"l_val": (0, 1), "u_v_val": (-1, 1)},
    }
    ranges = value_ranges[bitdepth]
    values = {
        "l_val": float(l_val),
        "u_val": float(u_val),
        "v_val": float(v_val),
    }
    if not (
        ranges["l_val"][0] <= values["l_val"] <= ranges["l_val"][1]
        and ranges["u_v_val"][0] <= values["u_val"] <= ranges["u_v_val"][1]
        and ranges["u_v_val"][0] <= values["v_val"] <= ranges["u_v_val"][1]
    ):
        raise ValueError(
            f"CIELUV values are out of range for the specified bit depth: {bitdepth}"
        )

    factor = BITDEPTH_FACTOR[bitdepth]
    return values["l_val"], values["u_val"], values["v_val"], factor


def validate_cielchuv_values(bitdepth, lchuv_l_val, lchuv_ch_val, lchuv_uv_val):
    """Check that CIELCHuv values are within the range of the chosen bit depth and
    additionally calculate normalization factor from bitdepth."""

    if bitdepth not in CIELCHUV_DEPTH_VALUES:
        raise ValueError(f"Invalid bit depth specified: {bitdepth}")

    min_val_lchuv_l, max_val_lchuv_l = CIELCHUV_DEPTH_VALUES[bitdepth]["L"]
    min_val_lchuv_ch, max_val_lchuv_ch = CIELCHUV_DEPTH_VALUES[bitdepth]["Ch"]
    min_val_lchuv_uv, max_val_lchuv_uv = CIELCHUV_DEPTH_VALUES[bitdepth]["uv"]

    lchuv_l_val, lchuv_ch_val, lchuv_uv_val = (
        float(lchuv_l_val),
        float(lchuv_ch_val),
        float(lchuv_uv_val),
    )

    if not (
        min_val_lchuv_l <= lchuv_l_val <= max_val_lchuv_l
        and min_val_lchuv_ch <= lchuv_ch_val <= max_val_lchuv_ch
        and min_val_lchuv_uv <= lchuv_uv_val <= max_val_lchuv_uv
    ):
        raise ValueError(
            f"CIELCHuv values are out of range for the specified bit depth: {bitdepth}"
        )

    factor = BITDEPTH_FACTOR[bitdepth]
    return lchuv_l_val, lchuv_ch_val, lchuv_uv_val, factor


def validate_ciexyz_values(bitdepth, x_val, y_val, z_val):
    """Check that CIEXYZ values are within 0-1 range and additionally calculate
    normalization factor from bitdepth."""

    if bitdepth not in CIEXYZ_DEPTH_VALUES:
        raise ValueError(f"Invalid bit depth specified: {bitdepth}")

    min_val, max_val = CIEXYZ_DEPTH_VALUES[bitdepth]

    x_val, y_val, z_val = float(x_val), float(y_val), float(z_val)

    if not (
        min_val <= x_val <= max_val
        and min_val <= y_val <= max_val
        and min_val <= z_val <= max_val
    ):
        raise ValueError(
            f"CIEXYZ values are out of range for the specified bit depth: {bitdepth}"
        )

    factor = BITDEPTH_FACTOR[bitdepth]
    return x_val, y_val, z_val, factor


def validate_spectrum_values(start, stop, interval, data):
    """check that spectrum values are within range"""

    max_val_start = 750
    min_val_start = 380
    max_val_stop = 780
    min_val_stop = 400
    # Ensure a minimum range for meaningful spectrum data

    max_val_interval = 20
    min_val_interval = 1

    # Check if start, stop, and interval values are within their respective ranges
    if not (
        min_val_start <= start <= max_val_start
        and min_val_stop <= stop <= max_val_stop
        and min_val_interval <= interval <= max_val_interval
    ):
        raise ValueError("Spectral parameters are out of range")

    # Check if the stop is greater than start
    if stop <= start:
        raise ValueError("Stop value should be greater than start value")

    expected_data_length = int((stop - start) / interval) + 1
    if len(data) != expected_data_length:
        raise ValueError(
            f"Data length mismatch. Expected {expected_data_length} values but got "
            f"{len(data)} values."
        )

    data = np.array(data)

    if np.any(data < 0) or np.any(data > 1):
        raise ValueError("Spectral data values are out of 0-1 range")

    return start, stop, interval, data


def validate_wave_value(wave):
    """check that wavelength in nm is within the visible spectrum"""

    if not VIS_SPECTRUM_MIN <= wave <= VIS_SPECTRUM_MAX:
        raise ValueError(
            "Wavelength value must be between 360 - 830 nm [CIE 015:2018, pg. 21]"
        )

    return wave


def convert_rgb_to_rgb(colorspaceid):
    """convert RGB to RGB colorspace"""

    if colorspaceid not in COLOR_MODELS_ALIASES:
        for cat in use_cats:
            if (
                RGB_COLOURSPACES[input_colorspace].whitepoint_name
                == RGB_COLOURSPACES[colorspaceid].whitepoint_name
            ):
                # check if the illuminants match, and if they do,
                # don't apply chromatic adaptation
                cat = None
            rgb = RGB_to_RGB(
                linear_rgb,
                RGB_COLOURSPACES[input_colorspace],
                RGB_COLOURSPACES[colorspaceid],
                cat,
                False,
                True,
            )

            # denormalize RGB values back into the correct bit depth
            rgb_denorm = denormalize(rgb, factor)

            result(
                colorspaceid,
                rgb_denorm,
                illuminant=RGB_COLOURSPACES[colorspaceid].whitepoint_name,
                cat=cat,
            )
            if (
                RGB_COLOURSPACES[input_colorspace].whitepoint_name
                == RGB_COLOURSPACES[colorspaceid].whitepoint_name
            ):
                # leave cat loop if chromatic adaptation is not needed
                break


def convert_rgb_no_xyz(rgb_norm, p, wt, illuminant=False):
    """convert RGB to spaces not requiring XYZ intermediary"""

    cmy = RGB_to_CMY(np.array(rgb_norm))
    cmyk = np.around(CMY_to_CMYK(cmy), 2)
    cmy = np.around(cmy, 2)
    hcl = np.around(RGB_to_HCL(np.array(rgb_norm)), 6)

    hsl = RGB_to_HSL(np.array(rgb_norm))
    hsv = RGB_to_HSV(np.array(rgb_norm))
    ihls = np.around(RGB_to_IHLS(np.array(rgb_norm)), 6)
    prismatic = np.around(RGB_to_Prismatic(np.array(rgb_norm)), 4)
    rgbluminance = np.around(RGB_luminance(np.array(rgb_norm), p, wt), 4)
    ycbcr = RGB_to_YCbCr(np.array(rgb_norm))
    yccbccrc = np.around(
        RGB_to_YcCbcCrc(
            np.array(rgb_norm),
            out_legal=True,
            out_bits=10,
            out_int=False,
            is_12_bits_system=False,
        ),
        2,
    )
    ycocg = np.around(RGB_to_YCoCg(np.array(rgb_norm)), 4)

    hsl_denorm = denormalize(hsl, 360, 100, 100)
    hsl_h_val_css = np.round(hsl_denorm[0], 3)
    hsl_s_val_css = np.round(hsl_denorm[1], 3)
    hsl_l_val_css = np.round(hsl_denorm[2], 3)
    hsl_css_color = f"hsl({hsl_h_val_css}deg {hsl_s_val_css}% {hsl_l_val_css}%);"

    hwb_white_val = np.min(rgb_norm)
    hwb_black_val = 1 - np.max(rgb_norm)
    hwb = np.array([hsl[0], hwb_white_val, hwb_black_val])  # hsl[0] is not a typo
    hwb_denorm = denormalize(hwb, 360, 100, 100)
    hwb_h_val_css = np.round(hwb_denorm[0], 3)
    hwb_white_val_css = np.round(hwb_denorm[1], 3)
    hwb_black_val_css = np.round(hwb_denorm[2], 3)
    hwb_css_color = (
        f"hwb({hwb_h_val_css}deg {hwb_white_val_css}% {hwb_black_val_css}%);"
    )

    # BUILD THE ARRAY OF OUTPUT VALUES, INCLUDING NEEDED DENORMALIZATIONS
    result("cmy", denormalize(cmy, 100), illuminant)
    result("cmyk", denormalize(cmyk, 100), illuminant)
    result("hcl", hcl, illuminant)
    result("hsl", hsl_denorm, illuminant)
    result("hsl_css_color", hsl_css_color, illuminant)
    result("hsv", denormalize(hsv, 360, 100, 100), illuminant)
    result("hwb", hwb_denorm, illuminant)
    result("hwb_css_color", hwb_css_color, illuminant)
    result("ihls", ihls, illuminant)
    result("prismatic", prismatic, illuminant)
    result("rgbluminance", rgbluminance, illuminant)
    result("ycbcr", denormalize(ycbcr), illuminant)
    result("yccbccrc", yccbccrc, illuminant)
    result("ycocg", ycocg, illuminant)


def chromatic_adaptations_in_convert_with_xyz(
    ciexyz, illuminant_xyz_d65, illuminant_in_xyz
):
    transform = "Bradford"
    ciexyz_d65 = chromatic_adaptation_VonKries(
        ciexyz, illuminant_xyz_d65, illuminant_in_xyz, transform
    )
    ciexyz_c = chromatic_adaptation_VonKries(
        ciexyz, illuminant_xyz_c, illuminant_in_xyz, transform
    )
    return ciexyz_d65, ciexyz_c


def process_cielchuv():
    intermediate_cieluv = LCHuv_to_Luv(cielchuv)
    ciexyz_val = Luv_to_XYZ(intermediate_cieluv, illuminant_in_xy)
    ciexyz_d65_val = Luv_to_XYZ(intermediate_cieluv, illuminant_xy_d65)
    ciexyz_c_val = Luv_to_XYZ(intermediate_cieluv, illuminant_xy_c)
    return ciexyz_val, ciexyz_d65_val, ciexyz_c_val


def process_cielab_or_cielchab():
    ciexyz_val = Lab_to_XYZ(cielab, illuminant_in_xy)
    ciexyz_d65_val = Lab_to_XYZ(cielab, illuminant_xy_d65)
    ciexyz_c_val = Lab_to_XYZ(cielab, illuminant_xy_c)
    return ciexyz_val, ciexyz_d65_val, ciexyz_c_val


def process_cieluv():
    ciexyz_val = Luv_to_XYZ(cieluv, illuminant_in_xy)
    ciexyz_d65_val = Luv_to_XYZ(cieluv, illuminant_xy_d65)
    ciexyz_c_val = Luv_to_XYZ(cieluv, illuminant_xy_c)
    return ciexyz_val, ciexyz_d65_val, ciexyz_c_val


def process_ciexyz():
    ciexyz_val = (float(args.x_val), float(args.y_val), float(args.z_val))
    ciexyz_d65_val = chromatic_adaptations_in_convert_with_xyz(
        ciexyz_val, illuminant_xyz_d65, illuminant_in_xyz
    )
    return ciexyz_val, ciexyz_d65_val, None


def process_spectrum():
    sd_cmfs = MSDS_CMFS[observer]
    illuminant_sds = SDS_ILLUMINANTS[illuminant]
    ciexyz_val = (
        sd_to_XYZ(
            sd, sd_cmfs, illuminant_sds, method="Integration", shape=sd_shape, k=None
        )
        / 100
    )
    ciexyz_d65_val = chromatic_adaptations_in_convert_with_xyz(
        ciexyz_val, illuminant_xyz_d65, illuminant_in_xyz
    )
    return ciexyz_val, ciexyz_d65_val, None


def process_wavelength():
    wave = args.wave
    wave_cmfs = MSDS_CMFS[observer]
    ciexyz_val = wavelength_to_XYZ(wave, wave_cmfs)
    return ciexyz_val, ciexyz_val, None


def convert_with_xyz(observer, illuminant):
    """convert to spaces requiring xyz intermediary"""
    global cielab, cielchab, cieluv, cielchuv, hsl, illuminant_in_xyz, illuminant_in_xy
    global ciexyz, ciexyz_d65, ciexyz_c, sd, sd_shape, model

    if model == "Spectrum" and illuminant not in SDS_ILLUMINANTS:
        return
    if illuminant not in CCS_ILLUMINANTS[observer]:
        return

    # xy AND XYZ COORDINATES FOR THE ARRAY OF ILLUMINANTS
    illuminant_in_xy = CCS_ILLUMINANTS[observer][illuminant]
    illuminant_in_xyz = xy_to_XYZ(illuminant_in_xy)
    illuminant_in_xyz_x_100 = np.array(illuminant_in_xyz * 100)

    illuminant_xy_c = CCS_ILLUMINANTS[observer]["C"]
    illuminant_xyz_c_x_100 = TVS_ILLUMINANTS[observer]["C"]
    illuminant_xyz_c = np.array(illuminant_xyz_c_x_100 / 100)

    illuminant_xy_d65 = CCS_ILLUMINANTS[observer]["D65"]
    illuminant_xyz_d65_x_100 = TVS_ILLUMINANTS[observer]["D65"]
    illuminant_xyz_d65 = np.array(illuminant_xyz_d65_x_100 / 100)

    model_functions = {
        "CIELCHuv": process_cielchuv,
        "CIELAB": process_cielab_or_cielchab,
        "CIELCHab": process_cielab_or_cielchab,
        "CIELUV": process_cieluv,
        "CIEXYZ": process_ciexyz,
        "Spectrum": process_spectrum,
        "Wavelength": process_wavelength,
    }

    # Using the model to invoke the corresponding function
    if model != "RGB":
        ciexyz, ciexyz_d65, ciexyz_c = model_functions[model]()

    # loop through chromatic adaptation transforms for conversions from RGB;
    # XYZ has no native white point so for other models it's not needed

    if model == "RGB":
        cats = use_cats
    else:
        cats = [None]

    for cat in cats:
        if model == "RGB":
            illuminants = [illuminant_in_xy, illuminant_xy_d65, illuminant_xy_c]
            ciexyz_values = []

            for ill in illuminants:
                xyz = RGB_to_XYZ(
                    linear_rgb,
                    RGB_COLOURSPACES[input_colorspace],
                    ill,
                    cat,
                    apply_cctf_decoding=True,
                )
                ciexyz_values.append(xyz)

            ciexyz, ciexyz_d65, ciexyz_c = ciexyz_values

    ciexyz_x_100 = ciexyz * 100

    # CIELAB RELATED
    if model != "CIELAB" and model != "CIELCHab":
        cielab = np.round(XYZ_to_Lab(ciexyz, illuminant_in_xy), 4)
        cielchab = np.round(Lab_to_LCHab(cielab), 4)

    lstar = cielab[0]
    astar = cielab[1]
    bstar = cielab[2]

    if model == "CIELAB":
        cielab_adapted = np.round(np.array(XYZ_to_Lab(ciexyz)), 4)
    elif model == "CIELCHab":
        cielchab_adapted = np.round(np.array(Lab_to_LCHab(cielab)), 4)
    elif model == "CIELUV":
        cieluv_adapted = np.array(XYZ_to_Luv(ciexyz))
        cielchuv = np.round(np.array(Luv_to_LCHuv(cieluv)), 4)
        cieluv_adapted = np.round(cieluv_adapted, 4)
    elif model == "CIELCHuv":
        cieluv = np.array(XYZ_to_Luv(ciexyz))
        cielchuv_adapted = np.round(np.array(Luv_to_LCHuv(cieluv)), 4)
        cieluv = np.round(cieluv, 4)

    hdrcielab = np.round(XYZ_to_hdr_CIELab(ciexyz, illuminant_in_xyz), 4)

    # xyY RELATED
    xyy = XYZ_to_xyY(ciexyz)
    xy = XYZ_to_xy(ciexyz)
    # munsell_clr = xyY_to_munsell_colour(xyy)

    cmfs = MSDS_CMFS[observer]

    if model != "CIELUV" and model != "CIELCHuv":
        cieluv = XYZ_to_Luv(ciexyz, illuminant_in_xy)
        cielchuv = np.round(Luv_to_LCHuv(cieluv), 6)

    cieluvuv = np.round(xy_to_Luv_uv(xy), 6)

    # CAM CONVERSIONS FROM XYZ
    atd95 = list(XYZ_to_ATD95(ciexyz_x_100, illuminant_in_xyz_x_100, Y_0, K_1, K_2))
    atd95 = list(
        np.round(item[1], 4) if item[1] is not None and item[1] == item[1] else "-"
        for item in atd95
    )

    cam16 = XYZ_to_CAM16(
        ciexyz_x_100, illuminant_in_xyz_x_100, L_A, Y_B, surround_cam16
    )
    jmhcam16 = CAM16_to_JMh_CAM16(cam16)
    cam16lcd = np.round(JMh_CAM16_to_CAM16LCD(jmhcam16), 4)
    cam16scd = np.round(JMh_CAM16_to_CAM16SCD(jmhcam16), 4)
    cam16ucs = np.round(JMh_CAM16_to_CAM16UCS(jmhcam16), 4)
    cam16 = list(
        np.round(item[1], 4) if item[1] is not None and item[1] == item[1] else "-"
        for item in cam16
    )

    ciecam02 = XYZ_to_CIECAM02(
        ciexyz_x_100, illuminant_in_xyz_x_100, L_A, Y_B, surround_ciecam02
    )
    jmhciecam02 = CIECAM02_to_JMh_CIECAM02(ciecam02)
    cam02lcd = np.round(JMh_CIECAM02_to_CAM02LCD(jmhciecam02), 4)
    cam02scd = np.round(JMh_CIECAM02_to_CAM02SCD(jmhciecam02), 4)
    cam02ucs = np.round(JMh_CIECAM02_to_CAM02UCS(jmhciecam02), 4)
    ciecam02 = list(
        np.round(item[1], 4) if item[1] is not None and item[1] == item[1] else "-"
        for item in ciecam02
    )

    ciecam16 = XYZ_to_CIECAM16(
        ciexyz_x_100, illuminant_in_xyz_x_100, L_A, Y_B, surround_ciecam16
    )
    jmhciecam16 = np.round(CIECAM16_to_JMh_CIECAM16(ciecam16), 4)
    ciecam16 = list(
        np.round(item[1], 4) if item[1] is not None and item[1] == item[1] else "-"
        for item in ciecam16
    )

    hellwig2022 = XYZ_to_Hellwig2022(
        ciexyz_x_100, illuminant_in_xyz_x_100, L_A, Y_B, surround_hellwig2022
    )
    jmhhellwig2022 = np.round(Hellwig2022_to_JMh_Hellwig2022(hellwig2022), 4)
    hellwig2022 = list(
        np.round(item[1], 4) if item[1] is not None and item[1] == item[1] else "-"
        for item in hellwig2022
    )

    hunt = XYZ_to_Hunt(
        ciexyz_x_100,
        illuminant_in_xyz_x_100,
        illuminant_in_xyz_x_100,
        L_A,
        surround_hunt,
        CCT_w=CCT_W,
    )
    hunt = list(
        (
            np.round(float(format(item[1], ".15g")), 4)
            if item[1] is not None and item[1] == item[1]
            else "-"
        )
        for item in hunt
    )

    kim2009 = XYZ_to_Kim2009(
        ciexyz_x_100, illuminant_in_xyz_x_100, L_A, media_kim2009, surround_kim2009
    )
    kim2009 = list(
        (
            np.round(float(format(item[1], ".15g")), 4)
            if item[1] is not None and item[1] == item[1]
            else "-"
        )
        for item in kim2009
    )

    llab = XYZ_to_LLAB(ciexyz_x_100, illuminant_in_xyz_x_100, Y_B, L, surround_llab)
    llab = list(
        (
            np.round(float(format(item[1], ".15g")), 4)
            if item[1] is not None and item[1] == item[1]
            else "-"
        )
        for item in llab
    )

    nayatani95 = list(
        XYZ_to_Nayatani95(ciexyz_x_100, illuminant_in_xyz_x_100, Y_O, E_O, E_OR)
    )
    nayatani95 = list(
        (
            np.round(float(format(item[1], ".15g")), 4)
            if item[1] is not None and item[1] == item[1]
            else "-"
        )
        for item in nayatani95
    )

    rlab = list(
        XYZ_to_RLAB(
            ciexyz_x_100, illuminant_in_xyz_x_100, Y_N, sigma_rlab, D_FACTOR_RLAB
        )
    )
    rlab = list(
        (
            np.round(float(format(item[1], ".15g")), 4)
            if item[1] is not None and item[1] == item[1]
            else "-"
        )
        for item in rlab
    )

    zcam = list(
        XYZ_to_ZCAM(
            ciexyz_x_100, illuminant_in_xyz_x_100, L_A_ZCAM, Y_B_ZCAM, surround_zcam
        )
    )
    zcam = list(
        (
            np.round(float(format(item[1], ".15g")), 4)
            if item[1] is not None and item[1] == item[1]
            else "-"
        )
        for item in zcam
    )

    # XYZ IS ADAPTED TO D65 FIRST FOR THESE SPACES
    hdript = np.round(XYZ_to_hdr_IPT(ciexyz_d65), 4)
    icacb = np.round(XYZ_to_ICaCb(ciexyz_d65), 4)

    # IPT-based spaces
    igpgtg = np.round(XYZ_to_IgPgTg(ciexyz_d65), 4)
    ipt = XYZ_to_IPT(ciexyz_d65)
    ipt_hue = np.round(IPT_hue_angle(ipt), 4)
    ipt = np.round(ipt, 4)
    iptragoo = np.round(XYZ_to_IPT_Ragoo2021(ciexyz_d65), 4)

    jzazbz = np.round(XYZ_to_Jzazbz(ciexyz_d65), 5)
    izazbz = np.round(XYZ_to_Izazbz(ciexyz_d65), 5)
    oklab = np.round(XYZ_to_Oklab(ciexyz_d65), 4)
    osaucs = np.round(XYZ_to_OSA_UCS(ciexyz), 4)
    cieucs = XYZ_to_UCS(ciexyz)
    cieucsuv = UCS_to_uv(cieucs)

    cieuvw = np.round(XYZ_to_UVW(ciexyz_x_100, illuminant_in_xy), 6)

    din99 = np.round(XYZ_to_DIN99(ciexyz, illuminant_in_xy), 4)
    din99b = np.round(XYZ_to_DIN99(ciexyz, illuminant_in_xy, method="DIN99b"), 4)
    din99c = np.round(XYZ_to_DIN99(ciexyz, illuminant_in_xy, method="DIN99c"), 4)
    din99d = np.round(XYZ_to_DIN99(ciexyz, illuminant_in_xy, method="DIN99d"), 4)

    hunterkakb = np.round(XYZ_to_K_ab_HunterLab1966(illuminant_in_xyz_x_100), 4)
    hunterlab = np.round(
        XYZ_to_Hunter_Lab(ciexyz_x_100, HUNTER_D65.XYZ_n, HUNTER_D65.K_ab), 4
    )
    hunterrdab = np.round(
        XYZ_to_Hunter_Rdab(ciexyz_x_100, HUNTER_D65.XYZ_n, HUNTER_D65.K_ab), 4
    )

    iab = np.round(XYZ_to_Iab(ciexyz, LMS_TO_LMS_P, M_XYZ_TO_LMS, M_LMS_P_TO_IAB), 4)

    ictcp_2100_1_hlg = np.round(
        XYZ_to_ICtCp(ciexyz, illuminant_in_xy, cat, method="ITU-R BT.2100-1 HLG"), 4
    )
    ictcp_2100_1_pq = np.round(
        XYZ_to_ICtCp(ciexyz, illuminant_in_xy, cat, method="ITU-R BT.2100-1 PQ"), 4
    )
    ictcp_2100_2_hlg = np.round(
        XYZ_to_ICtCp(ciexyz, illuminant_in_xy, cat, method="ITU-R BT.2100-2 HLG"), 4
    )
    ictcp_2100_2_pq = np.round(
        XYZ_to_ICtCp(ciexyz, illuminant_in_xy, cat, method="ITU-R BT.2100-2 PQ"), 4
    )

    prolab = np.round(XYZ_to_ProLab(ciexyz, illuminant_in_xy), 4)

    clrpurity = np.round(colorimetric_purity(xy, illuminant_in_xy, cmfs), 5)
    compwave = complementary_wavelength(xy, illuminant_in_xy, cmfs)
    domwave = dominant_wavelength(xy, illuminant_in_xy, cmfs)
    expurity = np.round(excitation_purity(xy, illuminant_in_xy, cmfs), 5)

    lmnce = luminance(lstar)
    ltns = np.round(lightness(lmnce, method="Abebe 2017"), 4)
    cct_duv = uv_to_CCT(cieucsuv)
    cct = cct_duv[0]
    duv = np.round(cct_duv[1], 6)
    mired = np.round(1000000 / cct, 2)
    muns_value = np.round(munsell_value(lmnce), 6)

    # ASTM D1925 requires adapting the input XYZ values to illuminant C normalized to 0-100
    ciexyz_c_x_100 = ciexyz_c * 100

    # including more methods as they seem to be in use, e.g. Datacolor spectros
    yellowness_d1925 = np.round(yellowness_ASTMD1925(ciexyz_c_x_100), 4)
    yellowness_e313 = np.round(yellowness_ASTME313(ciexyz_x_100), 4)

    # Whiteness using CIE 2004 produces two values, the WI (Whiteness Index)
    # and the TI (Tint Index) so we parse them here
    whiteness_cie = whiteness_CIE2004([xy], 100, [illuminant_in_xy], observer)
    whiteness_cie = np.round(whiteness_cie[0], 4)
    tintindex_cie = np.round(whiteness_cie[1], 4)

    # including more methods as they seem to be in use, e.g. Datacolor spectros
    whiteness_e313 = np.round(whiteness_ASTME313(ciexyz_x_100), 4)
    whiteness_berger = np.round(
        whiteness_Berger1959(ciexyz_x_100, illuminant_in_xyz_x_100), 4
    )
    whiteness_stensby = np.round(whiteness_Stensby1968(cielab), 4)

    yrg = XYZ_to_Yrg(ciexyz)

    # evaluate if the color is in Pointer's Gamut and MacAdam limits
    pointer = str(is_within_pointer_gamut(ciexyz)).capitalize()
    if illuminant == "A" or illuminant == "C" or illuminant == "D65":
        macadam = str(is_within_macadam_limits(xyy, illuminant)).capitalize()
        result("macadam", macadam, illuminant)

    # GENERATE THE CSS 4 COLORS
    convert_to_css(cielab, cielchab, oklab, illuminant, observer, cat)

    # ROUND THESE NOW THAT WE ARE DONE USING THEM IN CALCULATIONS
    cct = np.round(cct, 2)
    cielab = np.round(cielab, 4)
    cieluv = np.round(cieluv, 4)
    cieucs = np.round(cieucs, 4)
    cieucsuv = np.round(cieucsuv, 4)
    ciexyz = np.round(ciexyz, 8)
    jmhcam16 = np.round(jmhcam16, 4)
    jmhciecam02 = np.round(jmhciecam02, 4)
    lmnce = np.round(lmnce, 4)
    xy = np.round(xy, 6)
    xyy = np.round(xyy, 6)

    if model == "Wavelength":
        wavelen = wave
    else:
        wavelen = ""

    result("atd95", atd95, illuminant, observer, cat)
    result("cam02lcd", cam02lcd, illuminant, observer, cat)
    result("cam02scd", cam02scd, illuminant, observer, cat)
    result("cam02ucs", cam02ucs, illuminant, observer, cat)
    result("cam16", cam16, illuminant, observer, cat)
    result("cam16lcd", cam16lcd, illuminant, observer, cat)
    result("cam16scd", cam16scd, illuminant, observer, cat)
    result("cam16ucs", cam16ucs, illuminant, observer, cat)
    result("cct", cct, illuminant, observer, cat)
    result("ciecam02", ciecam02, illuminant, observer, cat)
    result("ciecam16", ciecam16, illuminant, observer, cat)
    if model == "CIELAB":
        result("cielab", cielab_adapted, illuminant, observer, cat)
    else:
        result("cielab", cielab, illuminant, observer, cat)
    if model == "CIELCHab":
        result("cielchab", cielchab_adapted, illuminant, observer, cat)
    else:
        result("cielchab", cielchab, illuminant, observer, cat)
    if model == "CIELUV":
        result("cieluv", cieluv_adapted, illuminant, observer, cat)
    else:
        result("cieluv", cieluv, illuminant, observer, cat)
    if model == "CIELCHuv":
        result("cielchuv", cielchuv_adapted, illuminant, observer, cat)
    else:
        result("cielchuv", cielchuv, illuminant, observer, cat)
    result("cieluvuv", cieluvuv, illuminant, observer, cat)
    result("cieucs", cieucs, illuminant, observer, cat)
    result("cieucsuv", cieucsuv, illuminant, observer, cat)
    result("cieuvw", cieuvw, illuminant, observer, cat)
    result("ciexyz", ciexyz, illuminant, observer, cat)
    result("clrpurity", clrpurity, illuminant, observer, cat)
    result("compwave", compwave, illuminant, observer, cat)
    result("din99", din99, illuminant, observer, cat)
    result("din99b", din99b, illuminant, observer, cat)
    result("din99c", din99c, illuminant, observer, cat)
    result("din99d", din99d, illuminant, observer, cat)
    result("domwave", domwave, illuminant, observer, cat)
    result("duv", duv, illuminant, observer, cat)
    result("expurity", expurity, illuminant, observer, cat)
    result("hdrcielab", hdrcielab, illuminant, observer, cat)
    result("hdript", hdript, illuminant, observer, cat)
    result("hellwig2022", hellwig2022, illuminant, observer, cat)
    result("hunt", hunt, illuminant, observer, cat)
    result("hunterkakb", hunterkakb, illuminant, observer, cat)
    result("hunterlab", hunterlab, illuminant, observer, cat)
    result("hunterrdab", hunterrdab, illuminant, observer, cat)
    result("iab", iab, illuminant, observer, cat)
    result("icacb", icacb, illuminant, observer, cat)
    result("ictcp_2100_1_hlg", ictcp_2100_1_hlg, illuminant, observer, cat)
    result("ictcp_2100_1_pq", ictcp_2100_1_pq, illuminant, observer, cat)
    result("ictcp_2100_2_hlg", ictcp_2100_2_hlg, illuminant, observer, cat)
    result("ictcp_2100_2_pq", ictcp_2100_2_pq, illuminant, observer, cat)
    result("igpgtg", igpgtg, illuminant, observer, cat)
    result("ipt", ipt, illuminant, observer, cat)
    result("ipt_hue", ipt_hue, illuminant, observer, cat)
    result("iptragoo", iptragoo, illuminant, observer, cat)
    result("izazbz", izazbz, illuminant, observer, cat, no_scientific_notation=True)
    result("jmhcam16", jmhcam16, illuminant, observer, cat)
    result("jmhciecam02", jmhciecam02, illuminant, observer, cat)
    result("jmhciecam16", jmhciecam16, illuminant, observer, cat)
    result("jmhhellwig2022", jmhhellwig2022, illuminant, observer, cat)
    result("jzazbz", jzazbz, illuminant, observer, cat, no_scientific_notation=True)
    result("kim2009", kim2009, illuminant, observer, cat)
    result("lightness", ltns, illuminant, observer, cat)
    result("llab", llab, illuminant, observer, cat)
    result("luminance", lmnce, illuminant, observer, cat)
    result("mired", mired, illuminant, observer, cat)
    # commented out due to excessive length of processing / unreliable output
    # result('munsell_clr', munsell_clr, illuminant, observer, cat)
    result("munsell_value", muns_value, illuminant, observer, cat)
    result("nayatani95", nayatani95, illuminant, observer, cat)
    result("oklab", oklab, illuminant, observer, cat)
    result("osaucs", osaucs, illuminant, observer, cat)
    result("pointer", pointer)
    result("prolab", prolab, illuminant, observer, cat)
    result("rlab", rlab, illuminant, observer, cat)
    result("tintindex_cie", tintindex_cie, illuminant, observer, cat)
    result("wavelength", wavelen)
    result("whiteness_cie", whiteness_cie, illuminant, observer, cat)
    result("whiteness_e313", whiteness_e313, illuminant, observer, cat)
    result("whiteness_berger", whiteness_berger, illuminant, observer, cat)
    result("whiteness_stensby", whiteness_stensby, illuminant, observer, cat)
    result("xy", xy, illuminant, observer, cat)
    result("xyY", xyy, illuminant, observer, cat)
    result("yellowness_d1925", yellowness_d1925, illuminant, observer, cat)
    result("yellowness_e313", yellowness_e313, illuminant, observer, cat)
    result("yrg", yrg)
    result("zcam", zcam, illuminant, observer, cat)

    # CONVERT TO RGB COLOR SPACES
    if model != "RGB":
        for colorspaceid in RGB_COLOURSPACES.keys():
            convert_xyz_to_rgb(colorspaceid, observer, illuminant, cat)


def convert_xyz_to_rgb(colorspaceid, observer, illuminant, cat):
    global illuminant_in_xy, illuminant_in_xyz, ciexyz

    if colorspaceid in COLOR_MODELS_ALIASES:
        return
    if (
        res[colorspaceid]["illuminant"] != "all"
        and res[colorspaceid]["illuminant"] != illuminant
    ):
        return

    # EXTRACT THE PRIMARIES AND WHITE XY COORDINATES FOR THE RGB COLOR SPACE
    wt = np.array(RGB_COLOURSPACES[colorspaceid].whitepoint)
    rd = np.array(RGB_COLOURSPACES[colorspaceid].primaries[0])
    gr = np.array(RGB_COLOURSPACES[colorspaceid].primaries[1])
    bl = np.array(RGB_COLOURSPACES[colorspaceid].primaries[2])
    illuminant_rgb_in_xy = wt
    illuminant_rgb_in_xyz = xy_to_XYZ(illuminant_rgb_in_xy)

    # PRODUCE AN ARRAY WITH THE PRIMARIES XY COORDINATES
    p = np.array([rd, gr, bl])

    # APPLY CHROMATIC ADAPTATION BETWEEN THE ILLUMINANT AND THE ILLUMINANT
    # OF THE RGB
    ca_method = "CMCCAT2000"
    cmccat2000_l_a = 200
    if illuminant_in_xy.all == illuminant_rgb_in_xy.all:
        ciexyz_adapted = ciexyz
    else:
        ciexyz_adapted = chromatic_adaptation(
            ciexyz,
            illuminant_in_xyz,
            illuminant_rgb_in_xyz,
            ca_method,
            L_A1=cmccat2000_l_a,
            L_A2=cmccat2000_l_a,
        )

    # CONVERT XYZ TO GAMMA-CORRECTED RGB
    encoded_rgb = XYZ_to_RGB(
        ciexyz_adapted,
        RGB_COLOURSPACES[colorspaceid],
        illuminant_in_xy,
        cat,
        apply_cctf_encoding=True,
    )

    # DENORMALIZE RGB VALUES BACK TO THE CORRECT BIT DEPTH
    rgb_denorm = denormalize(encoded_rgb, factor)

    result(colorspaceid, rgb_denorm, illuminant, observer, cat)

    convert_rgb_no_xyz(encoded_rgb, p, wt, illuminant)


def convert_to_css(
    cielab, cielchab, oklab, illuminant=False, observer=False, cat=False
):
    """
    GENERATE THE CSS 4 COLORS
    Reference: https://www.w3.org/TR/css-color-4/
    Note that we do not yet support oklch, as colour does not support it
    """

    illuminant_d65_css = CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
    illuminant_d50_css = CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"]
    ciexyz_d65_css_src = Lab_to_XYZ(cielab, illuminant_d65_css)
    ciexyz_d50_css_src = Lab_to_XYZ(cielab, illuminant_d50_css)

    lstar_css = np.round(cielab[0], 4)
    astar_css = np.array(np.round(cielab[1], 4))
    bstar_css = np.array(np.round(cielab[2], 4))

    # CSS VALUES FOR ASTAR AND BSTAR NEED TO BE CLAMPED TO -125 TO 125
    astar_css = np.clip(astar_css, -125, 125, out=astar_css)
    bstar_css = np.clip(bstar_css, -125, 125, out=bstar_css)

    cielab_css_color = f"lab({lstar_css} {astar_css} {bstar_css});"

    l_val_css = np.round(cielchab[0], 4)
    ch_val_css = np.round(cielchab[1], 4)
    ab_val_css = np.round(cielchab[2], 4)
    cielchab_css_color = f"lch({l_val_css} {ch_val_css} {ab_val_css});"

    x_val_d65_css = np.round(ciexyz_d65_css_src[0], 4)
    y_val_d65_css = np.round(ciexyz_d65_css_src[1], 4)
    z_val_d65_css = np.round(ciexyz_d65_css_src[2], 4)
    x_val_d50_css = np.round(ciexyz_d50_css_src[0], 4)
    y_val_d50_css = np.round(ciexyz_d50_css_src[1], 4)
    z_val_d50_css = np.round(ciexyz_d50_css_src[2], 4)
    ciexyz_d65_css_color = (
        f"color(xyz-d65 {x_val_d65_css} {y_val_d65_css} {z_val_d65_css});"
    )
    ciexyz_css_color = f"color(xyz {x_val_d65_css} {y_val_d65_css} {z_val_d65_css});"
    ciexyz_d50_css_color = (
        f"color(xyz-d50 {x_val_d50_css} {y_val_d50_css} {z_val_d50_css});"
    )

    ok_l_val_css = oklab[0]
    ok_a_val_css = oklab[1]
    ok_b_val_css = oklab[2]
    oklab_css_color = f"oklab({ok_l_val_css} {ok_a_val_css} {ok_b_val_css});"

    # RGB CSS 4 COLOR VALUES
    for rgb_css_colorspace in RGB_CSS_COLORSPACES:

        if rgb_css_colorspace == "ProPhoto RGB":
            rgb_css_decoded = XYZ_to_RGB(
                ciexyz_d50_css_src,
                rgb_css_colorspace,
                illuminant_d50_css,
                cat,
                apply_cctf_encoding=True,
            )
        else:
            rgb_css_decoded = XYZ_to_RGB(
                ciexyz_d65_css_src,
                rgb_css_colorspace,
                illuminant_d65_css,
                cat,
                apply_cctf_encoding=True,
            )

        rgb_css = np.round(
            np.array(
                RGB_COLOURSPACES[rgb_css_colorspace].cctf_encoding(rgb_css_decoded)
            ),
            4,
        )
        rgb_css_string = space_separated(rgb_css)

        if rgb_css_colorspace == "sRGB":
            srgb_css_color = f"color(srgb {rgb_css_string});"
            hexadecimal = RGB_to_HEX(rgb_css)
            linear_srgb_css = np.round(np.array(rgb_css_decoded), 3)
            linear_srgb_css_string = space_separated(linear_srgb_css)
            linear_srgb_css_color = f"color(srgb-linear {linear_srgb_css_string});"
        elif rgb_css_colorspace == "Adobe RGB (1998)":
            a98_rgb_css_color = f"color(a98-rgb {rgb_css_string});"
        elif rgb_css_colorspace == "Display P3":
            display_p3_css_color = f"color(display-p3 {rgb_css_string});"
        elif rgb_css_colorspace == "ITU-R BT.2020":
            rec_2020_css_color = f"color(rec2020 {rgb_css_string});"
        elif rgb_css_colorspace == "ProPhoto RGB":
            prophoto_rgb_css_color = f"color(prophoto-rgb {rgb_css_string});"

    result("a98_rgb_css_color", a98_rgb_css_color, illuminant, observer, cat)
    result("cielab_css_color", cielab_css_color, illuminant, observer, cat)
    result("cielchab_css_color", cielchab_css_color, illuminant, observer, cat)
    result("ciexyz_css_color", ciexyz_css_color, illuminant, observer, cat)
    result("ciexyz_d50_css_color", ciexyz_d50_css_color, illuminant, observer, cat)
    result("ciexyz_d65_css_color", ciexyz_d65_css_color, illuminant, observer, cat)
    result("display_p3_css_color", display_p3_css_color, illuminant, observer, cat)
    result("hexadecimal", hexadecimal)
    result("linear_srgb_css_color", linear_srgb_css_color, illuminant, observer, cat)
    result("oklab_css_color", oklab_css_color, illuminant, observer, cat)
    result("prophoto_rgb_css_color", prophoto_rgb_css_color, illuminant, observer, cat)
    result("rec_2020_css_color", rec_2020_css_color, illuminant, observer, cat)
    result("srgb_css_color", srgb_css_color, illuminant, observer, cat)


def result(
    model,
    value,
    illuminant=False,
    observer=False,
    cat=None,
    use_space_separated=False,
    no_scientific_notation=False,
):
    """add value to main result dictionary"""
    if use_space_separated and (
        isinstance(value) == list or isinstance(value) == np.ndarray
    ):
        if no_scientific_notation:
            value = space_separated_nosci(value)
        else:
            value = space_separated(value)

    if "codes" in res[model]:
        if not isinstance(value, (list, np.ndarray)):
            value = [value]
        value = dict(zip(res[model]["codes"], value))

    if model not in res:
        res[model] = {}
    if observer:
        if observer not in res[model]:
            res[model][observer] = {}
        if illuminant not in res[model][observer]:
            res[model][observer][illuminant] = {}

        if cat is None:
            res[model][observer][illuminant] = value
        else:
            res[model][observer][illuminant][cat] = value

    elif illuminant:
        if illuminant not in res[model]:
            res[model][illuminant] = {}
        if cat is None:
            res[model][illuminant] = value
        else:
            res[model][illuminant][cat] = value
    else:
        res[model]["None"] = value


if __name__ == "__main__":

    # PARSE COMMAND LINE ARGUMENTS
    args, model = parse_arguments()
    input_colorspace = args.input_colorspace
    input_observer = args.observer
    input_illuminant = args.input_illuminant
    illuminant_list = args.illuminant_list
    precalc = args.precalc
    use_cats = args.cat

    if not args.show_warnings:
        # Filter out Colour and Python runtime and usage warnings
        # https://colour.readthedocs.io/en/develop/generated/colour.utilities.filter_warnings.html#colour-utilities-filter-warnings
        filter_warnings(python_warnings=["ignore"])

    # INITIALIZE OUTPUT WITH COLOR MODEL TEMPLATE NAMES AND DESCRIPTIONS TO FILL LATER
    # WITH VALUES
    res = COLOR_MODELS_TEMPLATES

    # replace already calculated items from pre-calculated
    if precalc != "":
        res = res | json.loads(precalc)

    if model == "RGB":
        red, green, blue, factor = validate_rgb_values(
            args.bitdepth, args.red, args.green, args.blue
        )
        rgb_norm = normalize(red, green, blue, factor)
        if input_colorspace in RGB_COLOURSPACES:
            # APPLY THE DECODING CCTF TO CREATE LINEAR RGB VALUES BEFORE CONVERTING
            linear_rgb = np.array(
                RGB_COLOURSPACES[input_colorspace].cctf_decoding(rgb_norm)
            )
            # CONVERSIONS FROM RGB TO RGB COLOR SPACES, WITH ENCODING CCTF AT THE END,
            # CONVERTING TO GAMMA CORRECTED VALUES
            for colorspaceid in RGB_COLOURSPACES.keys():
                convert_rgb_to_rgb(colorspaceid)
        else:
            res = {"error": "RGB color space lookup failure"}

    elif (
        model == "CIELAB"
        and args.lstar is not None
        and args.astar is not None
        and args.bstar is not None
        or model == "CIELCHab"
        and args.lchab_l_val is not None
        and args.lchab_ch_val is not None
        and args.lchab_ab_val is not None
    ):
        if model == "CIELCHab":
            (
                validated_lchab_l_val,
                validated_lchab_ch_val,
                validated_lchab_ab_val,
                factor,
            ) = validate_cielchab_values(
                args.bitdepth, args.lchab_l_val, args.lchab_ch_val, args.lchab_ab_val
            )
            cielchab = np.array(
                [validated_lchab_l_val, validated_lchab_ch_val, validated_lchab_ab_val]
            )
            cielab = np.array(LCHab_to_Lab(cielchab))
        elif model == "CIELAB":
            validated_lstar, validated_astar, validated_bstar, factor = (
                validate_cielab_values(
                    args.bitdepth, args.lstar, args.astar, args.bstar
                )
            )
            cielab = np.array([validated_lstar, validated_astar, validated_bstar])
            cielchab = np.array(Lab_to_LCHab(cielab))

    elif (
        model == "CIELUV"
        and args.l_val is not None
        and args.u_val is not None
        and args.v_val is not None
        or model == "CIELCHuv"
        and args.lchuv_l_val is not None
        and args.lchuv_ch_val is not None
        and args.lchuv_uv_val is not None
    ):
        if model == "CIELCHuv":
            (
                validated_lchuv_l_val,
                validated_lchuv_ch_val,
                validated_lchuv_uv_val,
                factor,
            ) = validate_cielchuv_values(
                args.bitdepth, args.lchuv_l_val, args.lchuv_ch_val, args.lchuv_uv_val
            )
            cielchuv = np.array(
                [validated_lchuv_l_val, validated_lchuv_ch_val, validated_lchuv_uv_val]
            )
        elif model == "CIELUV":
            validated_l_val, validated_u_val, validated_v_val, factor = (
                validate_cieluv_values(
                    args.bitdepth, args.l_val, args.u_val, args.v_val
                )
            )
            cieluv = np.array([validated_l_val, validated_u_val, validated_v_val])

    elif (
        model == "CIEXYZ"
        and args.x_val is not None
        and args.y_val is not None
        and args.z_val is not None
    ):
        validated_x_val, validated_y_val, validated_z_val, factor = (
            validate_ciexyz_values(args.bitdepth, args.x_val, args.y_val, args.z_val)
        )
        ciexyz = np.array([validated_x_val, validated_y_val, validated_z_val])

    elif (
        model == "Spectrum"
        and args.start is not None
        and args.stop is not None
        and args.interval is not None
        and args.data is not None
    ):
        validated_start, validated_stop, validated_interval, validated_data = (
            validate_spectrum_values(args.start, args.stop, args.interval, args.data)
        )
        FACTOR = 255
        interpolated_interval = args.intrpl_intrvl
        sd_shape = SpectralShape(validated_start, validated_stop, validated_interval)
        sd = SpectralDistribution(validated_data, sd_shape, name="")

        if interpolated_interval:
            interpolated_shape = SpectralShape(
                validated_start, validated_stop, interpolated_interval
            )
            interpolated_sd = sd.interpolate(interpolated_shape)

        # PLOT TM-30 REPORT
        if args.tm30path and validated_interval <= 5:
            if args.tm30format:
                plot_single_sd_colour_rendition_report(
                    sd, args.tm30format, standalone=False
                )
                close()
            else:
                plot_single_sd_colour_rendition_report(sd, "Full", standalone=False)
            savefig(fname=args.tm30path)
            close()
        elif args.tm30path and validated_interval > 5:
            if args.tm30format:
                plot_single_sd_colour_rendition_report(
                    interpolated_sd, args.tm30format, standalone=False
                )
                close()
            else:
                plot_single_sd_colour_rendition_report(
                    interpolated_sd, "Full", standalone=False
                )

        # PLOT THE SPECTRUM USING COLOUR / MATPLOTLIB
        if args.plotpath:
            plot_kwargs = [
                {
                    "illuminant": SDS_ILLUMINANTS[input_illuminant],
                    "cmfs": MSDS_CMFS[input_observer],
                },
            ]
            plot_multi_sds(
                [sd * 100],
                y_label="",
                x_label="",
                standalone=False,
                legend=False,
                transparent_background=True,
                aspect=1.75,
                box_aspect=1.75,
                bounding_box=[300, 800, 0, 100],
                plot_kwargs=plot_kwargs,
            )
            locs, labels = xticks()
            xticks(np.arange(300, 800, step=50))
            locsy, labelsy = yticks()
            yticks(np.arange(0, 110, step=10))
            grid(visible=True, which="both", axis="both", color="0.94")
            savefig(fname=args.plotpath)
            close()

        if args.spectype == "Emissive":
            # CALCULATIONS SPECIFIC TO EMISSIVE SPECTRAL DISTRIBUTIONS
            cri = np.round(colour_rendering_index(sd), 6)
            cqs = np.round(colour_quality_scale(sd), 6)
            lum_efficacy = np.round(luminous_efficacy(sd), 6)
            lum_efficiency = np.round(luminous_efficiency(sd), 6)
            lum_flux = np.round(luminous_flux(sd), 6)
            if validated_interval <= 5:
                cfi = np.round(colour_fidelity_index(sd), 6)
                result("cfi", cfi)
            result("cqs", cqs)
            result("cri", cri)
            result("lum_efficacy", lum_efficacy)
            result("lum_efficiency", lum_efficiency)
            result("lum_flux", lum_flux)

            # Calculate SSI for the input spectrum vs all the standard illuminants
            for illuminant in CIE_ILLUMINANTS_LIST:
                sd_test = SDS_ILLUMINANTS[illuminant]
                ssi = spectral_similarity_index(sd_test, sd)
                result("ssi", ssi, illuminant)

    elif model == "Wavelength" and args.wave is not None:
        validated_wave = validate_wave_value(args.wave)
        wave = validated_wave
        FACTOR = 255

    # CONVERSIONS REQUIRING XYZ AND STANDARD OBSERVER
    for observer in STANDARD_OBSERVERS:
        if illuminant_list == "All":
            for illuminant in RGB_ILLUMINANTS_LIST:
                convert_with_xyz(observer, illuminant)
        elif illuminant_list == "CIE":
            for illuminant in CIE_ILLUMINANTS_LIST:
                convert_with_xyz(observer, illuminant)
        elif illuminant_list == "ISO_7589":
            for illuminant in ISO_7589_ILLUMINANTS_LIST:
                convert_with_xyz(observer, illuminant)

    # SPECTRAL RECOVERY USING Jakob and Hanika (2019) Method
    # we only produce output for the input illuminant and 2 degree observer
    if model != "Spectrum":
        sr_cmfs = (
            MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
            .copy()
            .align(SpectralShape(360, 830, 5))
        )
        sr_illuminant = SDS_ILLUMINANTS[input_illuminant].copy().align(sr_cmfs.shape)

        sr_sd = XYZ_to_sd_Jakob2019(ciexyz_d65, sr_cmfs, sr_illuminant)
        result(
            "sr_sd",
            dict(zip(sr_sd.wavelengths, sr_sd.values)),
            input_illuminant,
            "CIE 1931 2 Degree Standard Observer",
        )
        if args.plotpath:
            plot_kwargs = [
                {
                    "illuminant": SDS_ILLUMINANTS["D65"],
                    "cmfs": MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
                },
            ]
            plot_multi_sds(
                [sr_sd * 100],
                y_label="",
                x_label="",
                standalone=False,
                legend=False,
                transparent_background=True,
                aspect=1.75,
                box_aspect=1.75,
                bounding_box=[360, 830, 0, 100],
                plot_kwargs=plot_kwargs,
            )
            locs, labels = xticks()
            xticks(np.arange(380, 820, step=40))
            locsy, labelsy = yticks()
            yticks(np.arange(0, 110, step=10))
            grid(visible=True, which="both", axis="both", color="0.94")
            savefig(fname=args.plotpath)
            close()

    # OUTPUT COMPLETE ARRAY OF CONVERSIONS IN JSON FORMAT
    dumped = json.dumps(res, cls=NumpyEncoder, separators=(",", ":"))
    print(dumped)
