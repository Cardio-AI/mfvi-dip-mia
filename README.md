# Posterior temperature optimized Bayesian models for inverse problems in medical imaging

Max-Heinrich Laves*, Malte Tölle*, Alexander Schlaefer, Sandy Engelhardt  
(* contributed equally)

Code for our MIDL 2021 Special Issue in MedIA journal submission on optimizing the temperature of Bayesian posteriors for inverse tasks in medical imaging.
This submission considerably extends a preliminary version of this work presented at the "Fourth Conference on Medical Imaging with Deep Learning" (Tölle et al., 2021).

## Abstract

We present Posterior Temperature Optimized Bayesian Inverse Models (POTOBIM), an unsupervised Bayesian approach to inverse problems in medical imaging using mean-field variational inference with a fully tempered posterior.
Bayesian methods exhibit useful properties for approaching inverse tasks, such as tomographic reconstruction or image denoising.
A suitable prior distribution introduces regularization, which is needed to solve the ill-posed problem and reduces overfitting the data.
In practice, however, this often results in a suboptimal posterior temperature, and the full potential of the Bayesian approach is not being exploited.
In POTOBIM, we optimize both the parameters of the prior distribution and the posterior temperature with respect to reconstruction accuracy using Bayesian optimization with Gaussian process regression.
Our method is extensively evaluated on four different inverse tasks on a variety of modalities with images from public data sets and we demonstrate that an optimized posterior temperature outperforms both non-Bayesian and Bayesian approaches without temperature optimization.
The use of an optimized prior distribution and posterior temperature leads to improved accuracy and uncertainty estimation and we show that it is sufficient to find these hyperparameters per task domain.
Well-tempered posteriors yield calibrated uncertainty, which increases the reliability in the predictions.

## BibTeX

MedIA 2021

```
under review
```

MIDL 2021

```
@inproceedings{toelle2021mean,
  title={A Mean-Field Variational Inference Approach to Deep Image Prior for Inverse Problems in Medical Imaging},
  author={T{\"o}lle, Malte and Laves, Max-Heinrich and Schlaefer, Alexander},
  booktitle={Medical Imaging with Deep Learning},
  year={2021},
}
```

See https://github.com/maltetoelle/mfvi-dip for our initial MIDL2021 code repository.

## Contact

Max-Heinrich Laves  
[max.laves@tuhh.de](mailto:max.laves@tuhh.de)  
[@MaxLaves](https://twitter.com/MaxLaves)

Institute of Medical Technology and Intelligent Systems  
Hamburg University of Technology  
Am Schwarzenberg-Campus 3, 21073 Hamburg, Germany
