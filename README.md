# SVD_ps
In this project, we study the feasibility of detecting the Baryon Acoustic Oscillations from kinematic weak lensing surveys with tomographic
bin method. We attach the main input files and code of the project.
1. We simulated 2D cosmic shear power spectrum from input linear spactial power spectrum theoretically, for both photometric and 
   spectroscopic weak lensing surveys.
2. Applying tomographic bins over the survey redshift range, we calculated auto and cross power spectra, as well as the Gaussian covariance 
   matrix for a given angular scale. 
3. We assumed the $\chi^2$ difference between the observed and theoretical shear power spectrum, based on which we extracted the spatial power
   spectrum using singular value decomposition technique.
4. Using the emcee package, we fit the output spatial power spectrum by the BAO fitting model with the input spatial power spectrum, and 
   obtaied the measurement error for the BAO scale and the amplitude of BAO wiggles.

