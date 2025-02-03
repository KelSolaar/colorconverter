<h1>COLOR CONVERTER</h1>
<h3></h3>&copy; 2024-2025, Colorhythm LLC</h3>
<br/><br/>
<p>
Color converter is a comprehensive script for converting an input color to a wide range of color models, color spaces, illuminants, chromatic adaptation transforms, and observers.  Additionally, it has basic support for spectra.  The script also intends to act as a testing method for conversions executed by the superb Colour Science library, https://www.colour-science.org/   There are a number of plots that can be emitted as well, including spectral plots and IES TM-30 reports.  The code also prints a massive JSON array of the conversions and parameters.  We recommend https://jsonviewer.stack.hu/ to see the output formatted.</p>

<p>Please note that there are a number of hard-coded constants for CAMs and for spectral recovery.  It's possible to extend the script to include more permutations in the future for these, if desired. Aside from these known constraints, almost all color models and spaces supported by Colour are supported in this script.</p>

<p>Pull requests and discussions are more than welcome - with a script of this complexity and scope, we expect this to be a community effort!  We're grateful to the Colour Science developers and community for promoting color science and supporting education, and hope this script proves to be a worthy contribution.    : )</p>

Simplest input of RGB values:
```sh
python3 colorconverter.py --red 132 --green 205 --blue 12
```
This would activate a number of default values, as described below.
<br/><br/>

Complex input of RGB values:
```sh
python3 colorconverter.py --red 132 --green 205 --blue 12 --bitdepth '15+1' --observer 'CIE 1964 10 Degree Standard Observer' --input_illuminant 'A' --cat 'CMCCAT2000' --illuminant_list 'CIE'
```
Note that the color model is inferred from the value arguments and doesn't need to be specified.
<br/><br/>

Options:

  --bitdepth            Bit depth of input values {8, 15+1, 16, 32}
                        Default: '8'

  --precalc             Precalculated data in JSON format

  --observer            Standard observer {CIE 1931 2 Degree Standard Observer, CIE 1964 10 Degree Standard Observer}
                        Default: 'CIE 1931 2 Degree Standard Observer'

  --input_illuminant    Standard illuminant of input values
                        {A, D50, D55, D65, D75, FL1, FL2, FL3, FL3.1, FL3.2, FL3.3, FL3.4, FL3.5, FL3.6, FL3.7, FL3.8,
                        FL3.9, FL3.10, FL3.11, FL3.12, FL3.13, FL3.14, FL3.15, FL4, FL5, FL6, FL7, FL8, FL9, FL10, FL11,
                        FL12, HP1, HP2, HP3, HP4, HP5, ID50, ID65, LED-B1, LED-B2, LED-B3, LED-B4, LED-B5, LED-BH1,
                        LED-RGB1, LED-V1, LED-V2}
                        Default: All

  --cat                 Chromatic Adaptation Transform
                        {Bianco 2010, Bianco PC 2010, Bradford, CAT02, CAT02 Brill 2008, CAT16, CMCCAT97, CMCCAT2000,
                        Fairchild, Sharp, Von Kries, XYZ Scaling, None}
                        Default: All
                        
  --illuminant_list     Which list of illuminants to use {CIE, ISO_7589, All}
                        Default: 'All'

  --plotpath            Output spectral plot path; plot is not saved if argument is omitted. SVG extension is added if not provided

CIELAB:
  --lstar               Lightness value, 0-100
  --astar               a* value, any bit depth
  --bstar               b* value, any bit depth

CIELCHab:
  --lchab_l_val         Lightness value, 0-100
  --lchab_ch_val        Chroma value, any bit depth
  --lchab_ab_val        ab value, any bit depth

CIELUV:
  --l_val               Lightness value, any bit depth, 0-100
  --u_val               u* value, any bit depth
  --v_val               v* value, any bit depth

CIELCHuv:
  --lchuv_l_val         Lightness value, 0-100
  --lchuv_ch_val        Chroma value, any bit depth
  --lchuv_uv_val        uv value, any bit depth

CIEXYZ:
  --x_val               X value, any bit depth, a mix of all three cone responses, 0-1
  --y_val               Y value, luminance, any bit depth, a mix of L and M responses, 0-1
  --z_val               Z value, similar to Blue in CIERGB, solely made up of the S cone response, 0-1

RGB:
  --red                 Red value, any bit depth
  --green               Green value, any bit depth
  --blue                Blue value, any bit depth
  --input_colorspace    RGB color space of input values
                        Default: sRGB
SPECTRUM:
  --start               Start of distribution, between 360-790 nm
  --stop                End of distribution, between 400-830 nm
  --interval            Interval between measurements, in nm, 1-20
  --spectype            Spectrum type {Emissive, Reflective, Transmissive}
  --data                Spectral data - a list of decimal values (preferrably to 6+ places), with spaces in between, unquoted, unbracketed, 0-1
  --intrpl_intrvl       Smaller interval to interpolate to, between 1-5 nm
  --tm30path            Path and filename for IES TM-30 Color Rendition Report; actual or interpolated interval must be 1-5 nm
  --tm30format          Format for IES TM-30 Color Rendition Report {Full, Intermediate, Simple}
                        
WAVELENGTH:
  --wave                Wavelength in nm

