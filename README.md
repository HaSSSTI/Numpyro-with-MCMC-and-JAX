# Gravitational De-lensing with Numpyro, MCMC, and JAX

 This method is designed to reconstruct distorted galaxy images, restoring their true appearance by addressing gravitational lensing effects. Here's a comprehensive overview of this innovative approach:

## The Objective: Reconstructing Distorted Galaxy Images
At the core of this repository lies the primary objective of reconstructing galaxy images that have been distorted due to gravitational lensing. We leverage the advanced capabilities of Numpyro, MCMC, and JAX to uncover the hidden features of these distant galaxies.

## Method: Numpyro-with-MCMC-and-JAX
The centerpiece of our approach involves the utilization of Numpyro, a probabilistic programming library, in conjunction with Markov Chain Monte Carlo (MCMC) methods for parameter estimation. JAX, a numerical computing library, further enhances the computational efficiency of the method.

## Techniques for De-lensing
We offer several techniques within the Numpyro-with-MCMC-and-JAX framework:

### Method: HS + Starlet
The `model_HS_likelihood` function employs Horseshoe regularization and the starlet transformation. This combination ensures accurate de-lensing results by effectively addressing regularization and image transformation.

### Method: HS (Horseshoe)
For pure Horseshoe (HS) regularization, the `model_HS_withoutStarlet_likelihood` function focuses solely on tax-based regularization. This method serves as a baseline for evaluating the hybrid approach.

### Method: L1 + Starlet
The `model_L1_likelihood` function incorporates L1 regularization alongside the starlet transformation, presenting an alternative regularization strategy complemented by the transformation benefits.

### Method: L2 + Starlet
Similar to the L1 method, the `model_L2_likelihood` function introduces L2 regularization paired with the starlet transformation. This approach offers a distinct regularization perspective.

## Insightful Summaries and Execution
By configuring the method using the `args` object, the relevant functions are invoked. The `main` function oversees MCMC inference and produces comprehensive summaries, deepening the understanding of the reconstructed images.
