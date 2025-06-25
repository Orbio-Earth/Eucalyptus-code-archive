# Methane Computer Vision and Radiative Transfer

This repository provides a raw code dump of certain parts of [Orbio Earth's](https://www.orbio.earth/) development environment to train, validate and test methane detection algorithms using synthetic training data. It is part of [Project Eucalyptus](TODO). **TODO: LINK TO FINAL EUCAL DOCS**

**Disclaimer: This is not intended to be easily run, but to be used as a reference for code snippets/ways of approaching the methane detection problem.**

Code to do a) inference with our best Sentinel-2 and EMIT models and b) generate synthetic training data can be found in the [Project Eucalyptus Notebook repository](https://github.com/Orbio-Earth/Project-Eucalyptus).


# Who are you?
[Orbio Earth](https://www.orbio.earth/) is a climate-tech company founded in 2021 to turn open satellite data into actionable methane intelligence.

Project Eucalyptus is authored by the Orbio Earth team—Robert Huppertz, Maxime Rischard, Timothy Davis, Zani Fooy, Mahrukh Niazi, Philip Popien, Robert Edwards,, Vikram Singh, Jack Angela, Thomas Edwards, Maria Navarette, Diehl Sillers and Wojciech Adamczyk — with support from external collaborators.

# What can I find where?
- [Radiative Transfer tooling](radtran/README.md)
- [Single Blind Release Q1 2025 tooling](methane-cv/sbr_2025/)
- [Notebooks with various data/radtran workflows and tests](methane-cv/notebooks/)
- [Synthetic data generation code](methane-cv/src/data/)
- [How we train models](methane-cv/src/training/training_script.py)
- [How we validate, visualize and create detection threshold results](methane-cv/src/validation)
