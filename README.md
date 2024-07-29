COLOR CONVERTER

Color converter is a comprehensive script for converting an input color to a wide range of color models, color spaces, illuminants, chromatic adaptation transforms, and observers.  It also has basic support for spectra.  The script also intends to act as a testing script for the accuracy of the superb Colour Science library, https://www.colour-science.org/   There are a number of plots that can be emitted as well, including spectral plots and IES TM-30 reports.  The code also prints a massive compact JSON array of the conversions and parameters.

Please note that there are a number of hard-coded constants for CAMs and for spectral recovery.  It's possible to extend the script to include more permutations in the future for these, if desired. Aside from these known constraints, almost all color models and spaces supported by Colour are supported in this script.

Basic command:
python3 color_converter.py --red 124 --green 244 --blue 196

We recommend https://jsonviewer.stack.hu/ to see the output formatted.

Options:
  -h, --help            show this help message and exit
  --bitdepth            Bit depth of input values {8, 15+1, 16, 32}
  --precalc             Precalculated data in JSON format
  --observer            Standard observer {CIE 1931 2 Degree Standard Observer, CIE 1964 10 Degree Standard Observer}
  --input_illuminant    Standard illuminant of input values
                        {A, D50, D55, D65, D75, FL1, FL2, FL3, FL3.1, FL3.2, FL3.3, FL3.4, FL3.5, FL3.6, FL3.7, FL3.8,
                        FL3.9, FL3.10, FL3.11, FL3.12, FL3.13, FL3.14, FL3.15, FL4, FL5, FL6, FL7, FL8, FL9, FL10, FL11,
                        FL12, HP1, HP2, HP3, HP4, HP5, ID50, ID65, LED-B1, LED-B2, LED-B3, LED-B4, LED-B5, LED-BH1,
                        LED-RGB1, LED-V1, LED-V2}

  --cat                 Chromatic Adaptation Transform
                        {Bianco 2010, Bianco PC 2010, Bradford, CAT02, CAT02 Brill 2008, CAT16, CMCCAT97, CMCCAT2000,
                        Fairchild, Sharp, Von Kries, XYZ Scaling, None}
                        
  --illuminant_list    Which list of illuminants to use {CIE, ISO_7589, All}
  --plotpath           Output spectral plot path; plot is not saved if argument is omitted. SVG extension is added if not provided

CIELAB:
  --lstar LSTAR         Lightness value, 0-100
  --astar ASTAR         a* value, any bit depth
  --bstar BSTAR         b* value, any bit depth

CIELCHab:
  --lchab_l_val         Lightness value, 0-100
  --lchab_ch_val        Chroma value, any bit depth
  --lchab_ab_val        ab value, any bit depth

CIELUV:
  --l_val L_VAL         Lightness value, any bit depth, 0-100
  --u_val U_VAL         u* value, any bit depth
  --v_val V_VAL         v* value, any bit depth

CIELCHuv:
  --lchuv_l_val         Lightness value, 0-100
  --lchuv_ch_val        Chroma value, any bit depth
  --lchuv_uv_val        uv value, any bit depth

CIEXYZ:
  --x_val X_VAL         X value, any bit depth, a mix of all three cone responses, 0-1
  --y_val Y_VAL         Y value, luminance, any bit depth, a mix of L and M responses, 0-1
  --z_val Z_VAL         Z value, similar to Blue in CIERGB, solely made up of the S cone response, 0-1

RGB:
  --red RED             Red value, any bit depth
  --green GREEN         Green value, any bit depth
  --blue BLUE           Blue value, any bit depth
  --input_colorspace    RGB color space of input values

SPECTRUM:
  --start START         Start of distribution, between 360-790 nm
  --stop STOP           End of distribution, between 400-830 nm
  --interval            Interval between measurements, in nm, 1-20
  --spectype            Spectrum type {Emissive, Reflective, Transmissive}
  --data                Spectral data - a list of decimal values (preferrably to 6+ places), with spaces in between, unquoted, unbracketed, 0-1
  --intrpl_intrvl       Smaller interval to interpolate to, between 1-5 nm
  --tm30path            Path and filename for IES TM-30 Color Rendition Report; actual or interpolated interval must be 1-5 nm
  --tm30format          Format for IES TM-30 Color Rendition Report {Full, Intermediate, Simple}
                        

WAVELENGTH:
  --wave WAVE           Wavelength in nm

