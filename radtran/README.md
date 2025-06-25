# Radiative Transfer

Library for calculating the Radiative Transfer (radtran) over satellite images.  

# What is the model of radiative transfer that you use?
Initially at Orbio we relied on a conventional multilayer radiative-transfer solver. Although such models are physically exact, running it for tens of millions of pixels needed in our synthetic-data pipeline became a hurdle. We therefore built a single-pass radiative transfer approximation that: (a) preserves the physics that matters for methane retrieval, (b) drops or simplifies the terms that are expensive to compute, but have negligible impact on retrieval accuracy. Due to the modular nature of the model, we can drop approximations one by one, trading off the speed for the accuracy of the model.
