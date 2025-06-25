"""Constants for the radtran package."""

MIN_SOLAR_ANGLE = 0
MAX_SOLAR_ANGLE = 90
MIN_OBSERVER_ANGLE = 0
MAX_OBSERVER_ANGLE = 90

mol_to_N = 6.02214076e23  # Avogadro's number
m2_to_cm2 = 1e4  # Conversion factor from m^2 to cm^2
h2o_concentration = 673  # mol/m2 - taken from my own analysis of the suominet data
co2_concentration = 154  # mol/m2 - US Standad Atmosphere veritcal profiles of CO2 scaled to 421 ppm at sea level
ch4_concentration = 0.66  # mol/m2 - taken from Varon 2021 scaled to https://gml.noaa.gov/ccgg/trends_ch4/
