# References & Literature for Augmentation Modelling

This document contains the literature references used to justify the
augmentation and imaging-effect models applied during synthetic SEM
image generation.

The references provide support for modelling effects such as point-spread
function (PSF), image blur, scanning drift, and image distortion, which
are incorporated into the synthetic data to make the generated
reference/search image pairs more representative of realistic SEM
imaging conditions.

---

## [1] Point Spread Function and SEM Imaging Blur

**The Determination and Application of the Point Spread Function in the
Scanning Electron Microscope**

*Microscopy and Microanalysis, 2018*

https://academic.oup.com/mam/article/24/4/396/6901595v

**Relevance to this work:**  
This reference supports the use of point-spread-function-related
modelling and image blurring effects in the synthetic SEM imaging
pipeline. The PSF represents the finite imaging response of the SEM
system and motivates the inclusion of blur during image generation.

---

## [2] Scanning Drift and Image Composition

**Real-Time Scanning Charged-Particle Microscope Image Composition with
Correction of Drift**

https://doi.org/10.1017/S1431927610094250

**Relevance to this work:**  
This reference supports the modelling of scanning-related image drift.
Drift can cause the apparent position of structures to shift during
image acquisition. A drift component is therefore included in the
synthetic search images to improve robustness to positional variation.

---

## [3] Image Drift and Distortion in SEM

**Correction of Image Drift and Distortion in a Scanning Electron
Microscopy**

*Journal of Microscopy*

https://onlinelibrary.wiley.com/doi/10.1111/jmi.12293

**Relevance to this work:**  
This reference supports modelling image drift and geometric distortion
in SEM images. These effects motivate the use of transformations such
as drift, shear, and related spatial perturbations in the synthetic
search-image generation process.

---

## Augmentations / Imaging Effects Justified by the Literature

| Synthetic augmentation / effect | Purpose | Supporting reference |
|---|---|---|
| **Blur / PSF modelling** | Represents the finite spatial response of the SEM imaging system | [1] |
| **Drift / positional shift** | Models displacement caused during scanning and image acquisition | [2], [3] |
| **Geometric distortion / shear** | Represents spatial distortion in acquired SEM images | [3] |
| **Image degradation effects** | Improves robustness to realistic imaging variation | [1], [2], [3] |

> **Note:** The references are used to justify the physical/imaging
> motivation for these augmentation categories. The exact parameter
> ranges and probabilities used in this project are implementation
> choices for the synthetic-data generation pipeline.