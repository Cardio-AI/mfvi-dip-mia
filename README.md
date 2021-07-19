# Posterior Temperature Optimization in Variational Inference

Max-Heinrich Laves, Malte Tölle, Alexander Schlaefer

Code for our AutoML@ICML 2021 workshop submission on optimizing the temperature of Bayesian posteriors.

## Abstract

Cold posteriors have been reported to perform better in practice in the context of Bayesian deep learning (Wenzel et al., 2020).
In variational inference, it is common to employ only a partially tempered posterior by scaling the complexity term in the log-evidence lower bound (ELBO). In this work, we first derive the ELBO for a fully tempered posterior in mean-field variational inference and subsequently use Bayesian optimization to automatically find the optimal posterior temperature. Choosing an appropriate posterior temperature leads to better predictive performance and improved uncertainty calibration, which we demonstrate for the task of denoising medical images.

## BibTeX

```
under review
```

## Contact

Max-Heinrich Laves  
[max.laves@tuhh.de](mailto:max.laves@tuhh.de)  
[@MaxLaves](https://twitter.com/MaxLaves)

Institute of Medical Technology and Intelligent Systems  
Hamburg University of Technology  
Am Schwarzenberg-Campus 3, 21073 Hamburg, Germany