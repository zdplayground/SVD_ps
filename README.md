# SVD_ps
In this project, we study the possibility of detecting the Baryon Acoustic Oscillations from kinematic weak lensing surveys with tomographic
bins.
1. We attach the input files and code to simulate cosmic shear power spectrum from different stages of weak lensing surveys.
2. Taking the simulated shear power spectrum as the input, we extract the 3d spatial power spectrum using SVD matrix inversion.
3. Using the emcee package, we fit the output spatial power spectrum by the input one, and obtain the measurement error for the BAO
   scale and amplitude of BAO wiggles.
