"""Materials used in simulated environments."""

import numpy as np


# Factor to convert neper (Np) into dezibel (dB)
Np_per_dB = 20 / np.log(10)


class Material(object):
    """Defines properties of a material used in simulated environments.

    This class is hashable and computes this hash based on its attributes. Changing an
    attribute changes the hash as well which may be very problematic when used in
    hashable collections (set, key in dict). It is recommended not to change attributes
    after initializing a material.

    Parameters
    ----------
    name : str
        The assigned name of the material.
    sound_speed : int or float
        The speed of sound inside the material in m/s.
    density : int or float, optional
        The volumetric mass density of the material in kg/m^3.
    attenuation : int or float, optional
        The attenuation coefficient that describes how strongly the amplitude
        of a transmitted ultrasound wave is decreased inside the material.
        The value is treated asnapier per meter (Np/m).
    shearwave_speed : int or float, optional
    thermal_expansion_coeff : int or float, optional
    thermal_conductivity_coeff : int or float, optional
    heat_transfer_coeff : int or float, optional
    """

    @classmethod
    def from_elastic_modulus(cls, name, elastic_modulus, poissons_ratio, density, **kwargs):
        # https://en.wikipedia.org/wiki/Elastic_modulus
        # https://en.wikipedia.org/wiki/Sound_speed

        shearwave_speed = np.sqrt(elastic_modulus / (2 * density * (1 + poissons_ratio)))
        soundspeed = np.sqrt(
            elastic_modulus * (1 - poissons_ratio) / (density * (1 + poissons_ratio) * (1 - 2 * poissons_ratio)))
        return cls(name, sound_speed=soundspeed, shearwave_speed=shearwave_speed, density=density, **kwargs)

    @classmethod
    def from_lame_constants(cls, name, lam, mu, density, **kwargs):
        # https://en.wikipedia.org/wiki/Elastic_modulus
        # https://en.wikipedia.org/wiki/Sound_speed
        shearwave_speed = np.sqrt(mu / density)
        soundspeed = np.sqrt(lam / density + 2 * shearwave_speed ** 2)
        return cls(name, sound_speed=soundspeed, shearwave_speed=shearwave_speed, density=density, **kwargs)

    def __init__(self, name, sound_speed, density=None, attenuation=None, shearwave_speed=0.0,
                 thermal_expansion_coeff=None, thermal_conductivity_coeff=None, heat_transfer_coeff=None):
        self.name = name
        self.sound_speed = sound_speed  # m/s
        self.density = density  # kg/m3
        self.attenuation = attenuation  # Np/m
        self.shearwave_speed = shearwave_speed  # m/s
        self.thermal_expansion_coeff = thermal_expansion_coeff  # 1/K
        self.thermal_conductivity_coeff = thermal_conductivity_coeff  # W/(K·m)
        self.heat_transfer_coeff = heat_transfer_coeff  # W/(m2·K)

        self.eta_v = 1e-10  # bulk viscosity (Pa s), also called lambda_prime in  Molero et. al https://doi.org/10.1016/j.cpc.2014.05.016"
        self.eta_s = 1e-10  # (Pa s) shear viscosity, also called mu_prime in  Molero et. al https://doi.org/10.1016/j.cpc.2014.05.016"
        # could be calculate from eq (5) in  Molero et. al

        self.nonphysical_sound_absorption = 0  # dB/s artificially reduce stress
        #  and velocity by this rate ( anything != 0 violates physics)

    def __hash__(self):
        return hash(
            (
                self.name,
                self.sound_speed,
                self.density,
                self.attenuation,
                self.shearwave_speed,
                # self.thermal_expansion_coeff,
                # self.thermal_conductivity_coeff,
                # self.heat_transfer_coeff,
                # self.eta_v,
                # self.eta_s,
                # self.nonphysical_sound_absorption,
            )
        )

    def __eq__(self, other):
        return hash(self) == hash(other)

    @property
    def acoust_impedance(self):
        """  kg/m2/s """
        if self.sound_speed is None or self.density is None:
            return np.nan
        return self.sound_speed * self.density

    @property
    def lam(self):
        """
        Lame's first parameter lambda [N/m2]
        https://en.wikipedia.org/wiki/Lam%C3%A9_parameters

        """
        if self.density is None:
            raise ValueError("No density is given for Material '{}'!".format(self.name))
        else:
            return self.density * (self.sound_speed ** 2 - 2 * (self.shearwave_speed ** 2))

    @property
    def mu(self):
        """
        shear modulus mu or G [N/m2],
        https://en.wikipedia.org/wiki/Shear_modulus

        :return:
        """
        if self.density is None:
            raise ValueError("No density is given for Material '{}'!".format(self.name))
        else:
            return self.density * (self.shearwave_speed ** 2)

    @property
    def c11(self):
        return self.lam + 2 * self.mu

    @property
    def c12(self):
        return self.lam

    @property
    def c22(self):
        return self.lam + 2 * self.mu

    @property
    def c44(self):
        return self.mu

    @property
    def elastic_modulus(self):  # for compatibility
        """  Young's modulus E, [N/m2]
        https://en.wikipedia.org/wiki/Young%27s_modulus

        """
        # return self.sound_speed ** 2 * self.density * (1 + self.poissons_ratio) * (1 - 2 * self.poissons_ratio) / (
        #            1 - self.poissons_ratio) # c_{\mathrm{solid,p}} = \sqrt{\frac{E(1 - \nu)}{\rho (1 + \nu)(1 - 2 \nu)}}
        return self.mu * (3 * self.lam + 2 * self.mu) / (
                self.lam + self.mu)  # https://en.wikipedia.org/wiki/Elastic_modulus

    @elastic_modulus.setter
    def elastic_modulus(self, value):  # for compatibility
        pass

    @property
    def poissons_ratio(self):
        """
            https://en.wikipedia.org/wiki/Poisson%27s_ratio
        """
        # https://en.wikipedia.org/wiki/Elastic_modulus
        return self.lam / 2 / (self.lam + self.mu)

    @poissons_ratio.setter
    def poissons_ratio(self, value):
        pass

    # @property
    # def eta_v(self):
    #     "bulk viscosity (Pa s), also called lambda_prime in  Molero et. al https://doi.org/10.1016/j.cpc.2014.05.016"
    #     # could be calculate from eq (5) in  Molero et. al
    #     return 1e-30  # FIXME
    #
    # @property
    # def eta_s(self):
    #     "(Pa s) shear viscosity, also called mu_prime in  Molero et. al https://doi.org/10.1016/j.cpc.2014.05.016"
    #     # could be calculate from eq (5) in  Molero et. al
    #     return 1e-30  # FIXME

    def __str__(self):
        s = "Material \"{}\": c={} m/s ".format(self.name, np.mean(self.sound_speed))
        if hasattr(self, "shearwave_speed") and self.shearwave_speed is not None:
            s += " c_s={:3.0f} m/s ".format(np.mean(self.shearwave_speed))
        if hasattr(self, "acoust_impedance") and not np.all(np.isnan(self.acoust_impedance)):
            s += " Z={:3.1f} M kg/m2/s".format(np.mean(self.acoust_impedance) / 1e6)
        if hasattr(self, "thermal_conductivity_coeff") and self.thermal_conductivity_coeff is not None:
            s += " therm_cond= {:3.1f} W/(K m)".format(np.mean(self.thermal_conductivity_coeff))
        if not np.isscalar(self.sound_speed):
            s += " shape {}".format(self.sound_speed.shape)
        return s

    def __repr__(self):
        return str(self)


# @book{Lide2003,
#   title={CRC Handbook of Chemistry and Physics, 84th Edition},
#   author={Lide, D.R.},
#   isbn={9780849304842},
#   series={CRC HANDBOOK OF CHEMISTRY AND PHYSICS},
#   url={https://books.google.de/books?id=kTnxSi2B2FcC},
#   year={2003},
#   publisher={Taylor \& Francis}
# }


# sound speed in water as function of temperature
# http://resource.npl.co.uk/acoustics/techguides/soundpurewater/
water = Material("Water", 1497, 999.9720, 0)
water_at_323K = Material("Water_at_323K", 1543, 992.21, 0)

# medium with matched acoustic regarding to water
z_matched_water = Material("Z_Matched_Water", 6050, 247.4311, 0)

# medium with matched velocity regarding to water
v_matched_water = Material("V_Matched_Water", 1497, 2230, 0)

# 20degC 5MHz
# https://pure.ltu.se/portal/files/77084/artikel.pdf
# pmma = Material("PMMA", 2750, 1.18e3, 60.84)

# TODO: update to: http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=1539491
# http://journals.jps.jp/doi/pdf/10.1143/JPSJ.15.718
# attenuation=60.84 Neper / m
pmma = Material("PMMA", 2650, 1.18e3, shearwave_speed=1335)

# derivated form http://periodictable.com/Properties/A/SoundSpeed.al.html
Pb60Sn40 = Material("Pb60Sn40", 1756)

# http://www.iaea.org/inis/collection/NCLCollectionStore/_Public/43/095/43095088.pdf
# p157
# T=601K
Pb_liquid = Material("Pb_liquid_601K", 1804.9737, density=10672.0205)

# borosilicate glass
# https://hal.archives-ouvertes.fr/jpa-00255486/document
borosilicate_glass = Material("Glass", 6050, 2230,
                              shearwave_speed=3690,
                              thermal_expansion_coeff=3.3e-6,
                              thermal_conductivity_coeff=1.2,
                              heat_transfer_coeff=11.6
                              )
borosilicate_glass2 = Material("Glass", 5786.4, 2223.4,
                               shearwave_speed=3690,
                               thermal_expansion_coeff=3.3e-6,
                               thermal_conductivity_coeff=1.2,
                               heat_transfer_coeff=11.6
                               )

glass = borosilicate_glass

# GaInSn Ga67In20.5Sn12.5
# http://www.fusion.ucla.edu/neil/Publications/GaInSnUsage_Morley.pdf
gainsn = Material("GaInSn", 2730, 6360, shearwave_speed=0)

# Polystyrol (PS) (self-made water tank) 20degC
# http://scitation.aip.org/content/asa/journal/jasa/69/6/10.1121/1.385941
# (doi: http://dx.doi.org/10.1121/1.385941)
# density: https://www.kern.de/cgi-bin/riweta.cgi?nr=2101&lng=1
polystyrol = Material("PS", 2304, 1050)

# Polystyrene (PS) (self-made water tank) 20degC
# Lide, David R. "CRC handbook of physics and chemistry." (2001).
# temperature not specified, see:
# https://books.google.de/books?hl=de&lr=&id=bNDMBQAAQBAJ&oi=fnd&pg=PP1&dq=
# Handbook+of+Chemistry+and+Physics+david+lide&ots=H7bCzhox0H&sig=
# d5S9F120Wo5OLDKoVCt5H8ILgRo#v=onepage&q=polystyrene&f=false
polystyrene = Material("polystyrene", 2350, 1060, shearwave_speed=1120)

# Polyoxymethylen (POM)
# Martiensen, Werner: "Springer handbook of condensed matter and materials data." (2006)
pom = Material("POM", 2446, 1410)

# http://scitation.aip.org/content/asa/journal/jasa/79/5/10.1121/1.393664
# standard dry air at 0 degC and at a barometric pressure of 101.325 kPa.
# https://en.wikipedia.org/wiki/Density_of_air
air = Material("Air", 331.29, 1.2922, shearwave_speed=0)

# http://www.instrumart.com/assets/ge-sound-speeds-and-pipe-size-data.pdf
# 25 degC
# http://hyperphysics.phy-astr.gsu.edu/hbase/Tables/thrcn.html
aluminium = Material("Al", sound_speed=6320, density=2700, shearwave_speed=3130, thermal_conductivity_coeff=205)

# KOH 30% with binder
# values for 50% KOH without binder from:
# http://www.chetascontrol.com/data.html
koh = Material("KOH", 2285, 1500, 0)  # TODO: find the exact soundspeed for KOH with binder from experiment!!!
koh30 = Material("KOH30", 2011, 1280, 0)  # speed of sound  measured, density from list

# zinc
# from: https://en.wikipedia.org/wiki/Speeds_of_sound_of_the_elements_%28data_page%29
zinc = Material("Zinc", 4210, 7140, shearwave_speed=2440)

# stainless steel
# from: http://www.ondacorp.com/images/Solids.pdf # TODO: find reference from publication
stainless_steel = Material("Stainless steel", sound_speed=5790, density=7890, shearwave_speed=3100, )

# nickel
nickel = Material(name="Nickel", sound_speed=6040, shearwave_speed=3000, density=8900)

# ZN-slurries: values taken from characterization measurements:
# bulk fluid: 30% KOH, 1% binder, 6% ZnO
xx2_zp001_at3MHz = Material("xx2_zp001_at3MHz", 1975)
xx4_zp001_at3MHz = Material("xx4_zp001_at3MHz", 2020)
xx8_zp001_at3MHz = Material("xx8_zp001_at3MHz", 2015)
xx15_zp001_at3MHz = Material("xx15_zp001_at3MHz", 2035)
electrolyte_0_5 = Material("electrolyte_0_5", 1985)
electrolyte_0_6 = Material("electrolyte_0_6", 1985)
electrolyte_0_8 = Material("electrolyte_0_5", 1985)
electrolyte_1 = Material("electrolyte_1", 1985)
electrolyte_xx2_glassspheres = Material("electrolyte_xx2_glassspheres", 1985)
electrolyte_xx8_glassspheres = Material("electrolyte_xx8_glassspheres", 1985)
electrolyte_xx16_glassspheres = Material("electrolyte_xx16_glassspheres", 1985)
electrolyte_xx24_glassspheres = Material("electrolyte_xx24_glassspheres", 1985)

xx4_Gzp020_at3MHz = Material("xx4_Gzp020_at3MHz", 2015)
xx8_Gzp020_at3MHz = Material("xx8_Gzp020_at3MHz", 2015)
xx12_Gzp020_at3MHz = Material("xx12_Gzp020_at3MHz", 2015)
xx16_Gzp020_at3MHz = Material("xx16_Gzp020_at3MHz", 2015)

xx4_Gzp020_el_0_5 = Material("xx4_Gzp020_el_0_5", 2015)
xx8_Gzp020_el_0_5 = Material("xx8_Gzp020__el_0_5", 2015)
xx12_Gzp020_el_0_5 = Material("xx12_Gzp020_el_0_5", 2015)
xx16_Gzp020_el_0_5 = Material("xx16_Gzp020_el_0_5", 2015)

xx4_Gzp020_el_0_6 = Material("xx4_Gzp020_el_0_6", 2015)
xx8_Gzp020_el_0_6 = Material("xx8_Gzp020__el_0_6", 2015)
xx12_Gzp020_el_0_6 = Material("xx12_Gzp020_el_0_6", 2015)
xx16_Gzp020_el_0_6 = Material("xx16_Gzp020_el_0_6", 2015)

xx4_Gzp020_el_0_8 = Material("xx4_Gzp020_el_0_8", 2015)
xx8_Gzp020_el_0_8 = Material("xx8_Gzp020__el_0_8", 2015)
xx12_Gzp020_el_0_8 = Material("xx12_Gzp020_el_0_8", 2015)
xx16_Gzp020_el_0_8 = Material("xx16_Gzp020_el_0_8", 2015)

xx4_zp001_el_0_8 = Material("xx4_zp001_el_0_8", 2015)
xx8_zp001_el_0_8 = Material("xx8_zp001__el_0_8", 2015)
xx12_zp001_el_0_8 = Material("xx12_zp001_el_0_8", 2015)
xx16_zp001_el_0_8 = Material("xx16_zp001_el_0_8", 2015)

# Glycerin-Slurry-Phantom: value taken from characterization measurement:
# fluid: Glycerol with binder (99,9% pure Glycerol with 1% binder), 8%vol Zn
phantom_xx08_at3MHz = Material("phantom_xx08_at3MHz", 1925)
phantom_xx08_at4MHz = Material("phantom_xx08_at4MHz", 1900)

# us-gel speed of sound is measured in SonoGlide Fe
# Measurement_dir:Y:\Alte_Laufwerke\US\measurementData\Harmonic Imaging\soundspeed_sonoglide_fe_us_gel
us_gel = Material("us_gel", 1587.30)

# water-gel: 50:50 mixture of water and us-gel SonoGlide Fe
# speed of sound is approximated as mean value of us-gel and water
water_gel = Material("water_gel", 1542)

# 3d-printer-material speed of sound
# measurement values : https://fusionforge.zih.tu-dresden.de/plugins/mediawiki/wiki/mst/index.php/3dprinter-material-speedofsound
d3_printer = Material("3D_material", 2186.29)

# ABS plastic beige
# http://www.ndt.net/links/proper.htm
abs_beige = Material("abs_beige", 2230, 1030)

# ABS plastic black
# http://www.ndt.net/links/proper.htm
abs_black = Material("abs_black", 2250, 1050)

# copper
#
copper = Material("copper", 4760, shearwave_speed=2325, density=8920)

# silicon
# doi:10.1109/JMEMS.2009.2039697
# v=8433m/s 	v_shear=5843m/s 		from C11=165.64 GPa, C44=79.51 GPa, ro1=2.329 g/cm^3
# at room temperature
silicon = Material("silicon", 8433, shearwave_speed=5843, density=2329,
                   thermal_conductivity_coeff=149, thermal_expansion_coeff=2.6e-6,  # wikipedia
                   )

# http://www.flowanalytic.com/schall.pdf
# theta=20deg
glycerin = Material("Glycerin", 1923, density=1300)
glycerol = Material("Glycerol", 1923, density=1300)

# tin (liquid)
# https://www.engineeringtoolbox.com/sound-speed-liquids-d_715.html
# c(240degC)=2471m/s
# https://link.springer.com/content/pdf/10.1007%2FBF02755565.pdf
# rho (506.8K)=7032.4kg/m3
tin_lq = Material("liquid tin", 2471, density=7032.4)

# ZERODUR
# heat_transfer_coeff taken from borosilcat
# Values obtained from https://www.schott.com/advanced_optics/english/products/optical-materials/zerodur-extremely-low-expansion-glass-ceramic/zerodur/index.html#3
# https://www.schott.com/advanced_optics/english/knowledge-center/technical-articles-and-tools/tie.html?com-origin=de-DE -> TIE-43
# https://www.schott.com/d/advanced_optics/e532f55f-d6c1-4748-8c60-886eaca1daf5/1.2/schott-zerodur-general-may-2013-eng.pdf

zerodur = Material("zerodur",
                   6486,
                   density=2530,
                   thermal_conductivity_coeff=1.46,
                   thermal_expansion_coeff=0.05e-6,
                   heat_transfer_coeff=11.6)

# from: CRC Lide2003
titanium = Material("titanium",
                    6070,
                    density=4.506 * 1000,
                    thermal_conductivity_coeff=21.9,
                    thermal_expansion_coeff=8.6e-6,
                    heat_transfer_coeff=16.7,  # Kazys2015
                    shearwave_speed=3125)

# http://asm.matweb.com/search/SpecificMaterial.asp?bassnum=MTP641
Ti6Al4V = Material.from_elastic_modulus("Ti6Al4V",
                                        elastic_modulus=113.8e9,
                                        poissons_ratio=0.342,
                                        density=4.43 * 1000,
                                        thermal_conductivity_coeff=6.7,
                                        thermal_expansion_coeff=9.2e-6,  # at 250degC
                                        heat_transfer_coeff=16.7,  # Kazys2015, for titanium
                                        )

# http://www.matweb.com/search/DataSheet.aspx?MatGUID=a4b8f9f14b5d4c9a8176c3aa4af71976&ckck=1
Alloy600 = Material.from_elastic_modulus("Alloy600",
                                        elastic_modulus=207e9,
                                        poissons_ratio=0.29,
                                        density=8.42 * 1000,
                                        thermal_conductivity_coeff=14.8,
                                        thermal_expansion_coeff=13.7e-6,  # at 20-315degC
                                        )

# Wang1999, scientific name polybenzimidazol
CELAZOLE = Material("CELAZOLE",
                    3430,  # loss: 1.8dB/mm
                    density=1280,
                    thermal_conductivity_coeff=0.41,
                    thermal_expansion_coeff=23e-6,
                    # heat_transfer_coeff=16.7,
                    shearwave_speed=1470)  # loss: 7.2 dB/mm


# Cranial tissue material
# TODO shear-wave speed
# Source:
# https://www.sciencedirect.com/science/article/pii/S030156291000075X
# Other:
# https://arxiv.org/pdf/1802.00876.pdf
# https://onlinelibrary.wiley.com/doi/book/10.1002/9780470561478
brain_tissue = Material(
    "Brain_Tissue",
    sound_speed=1560,
    density=1040,
    # attenuation=(0.6 * Np_per_dB * 100)  # 0.6 Np/cm -> dB/m
)
cortical_bone = Material(
    "Cortical_Bone",
    sound_speed=3476,
    density=1975,
    # Roughly? https://asa.scitation.org/doi/pdf/10.1121/1.402637?class=pdf
    # shearwave_speed=1700,
    # attenuation=(6.9 * Np_per_dB * 100)  # 0.6 Np/cm -> dB/m
)


# Accoustic properties of polylactic acid (PLA) 3D printed with filament deposition
# modelling (FDM) at 1 MHz
# https://link.springer.com/article/10.1007%2Fs40870-019-00198-8
polylactict_acid = Material(
    "Polylactic_Acid", sound_speed=1860, density=1140, # shearwave_speed=1190
)


# https://link.springer.com/article/10.1007%2FBF02254743
epoxy_resin = Material(
    "Epoxy_Resin", sound_speed=2820, density=1150,
)

# Rough values derived by experiment
formlabs_resin = Material(
    "Formlabs_Resin", sound_speed=2700, density=1585,
)


# https://en.wikipedia.org/wiki/Fused_quartz
#Softening point: ≈1665 °C
#Annealing point: ≈1140 °C
#Strain point: 1070 °C
fused_quartz = Material.from_lame_constants(name="fused quartz",
                                            lam=15.87e9, mu=31.26e9,
                                            density=2203,
                                            thermal_conductivity_coeff=1.3,
                                            thermal_expansion_coeff=5.5e-7,
                                            heat_transfer_coeff=glass.heat_transfer_coeff,
                                            )

# a map of materials to their respective colour and hatches
# material: (colour, hatch)
# hatch:
# /   - diagonal hatching
# \   - back diagonal
# |   - vertical
# -   - horizontal
# +   - crossed
# x   - crossed diagonal
# o   - small circle
# O   - large circle
# .   - dots
# *   - stars


DEFAULT_MATERIAL_COLOUR_MAP = {air: ("white", None),
                               gainsn: ("0.4", "."),
                               water: ("#5F9EA0", "o"),
                               pmma: ("#9370DB", "x"),
                               glass: ("#ADD8E6", "+"),
                               borosilicate_glass2: ("#ADD8E6", "+"),
                               aluminium: ("#9370DB", "-"),
                               zinc: ("#cdb79e", "*"),
                               koh: ("#4682b4", "/"),
                               koh30: ("#4682b4", "/"),
                               tin_lq: ("#696969", "*"),
                               zerodur: ("#E29E1A", "|"),
                               brain_tissue: ("#c2aac4", None),
                               cortical_bone: ("#d1d0cc", None),
                               polylactict_acid: ("#f4f4f4", None),
                               epoxy_resin: ("#bfc9c6", None),
                               formlabs_resin: ("black", None),
                               copper : ("#CB6D51", None)
                               }

if __name__ == "__main__":
    _key = None
    _m = None
    _mlist = [_m for _key, _m in locals().items() if not _key.startswith("_") and isinstance(_m, Material)]
    _mlist.sort(key=lambda i: i.acoust_impedance if not np.isnan(i.acoust_impedance) else 0)
    print("    Z [1e6 * kg/m2/s] \t\t Material")
    print("=========================================")
    for _m in _mlist:
        print("{:8.1f} \t\t {}".format(_m.acoust_impedance / 1e6, _m))
