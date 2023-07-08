## Astronexus 🔭

Tools for Astrophotography (post-processing). [WIP]

![image](https://github.com/googlesamples/mediapipe/assets/48355572/511c2723-570c-4a6b-9dd8-3665e6052223)

### [WIP] Functionalities

* Background subtraction (gradient removal)
* 32-bit, 16-bit, 8-bit image conversion
* Color matrix conversion
* Color space conversion
* Chromatic adaptation (XYZ scaling/Bradford/Von Kries/White-Point method)
* Gamma correction
* High dynamic range 2D Deconvolution (Richardson-Lucy algorithm / Deringing algorithm)
* Local histogram equalization
* Simple star detection
* Star fitting with 2D Gaussian
* Star size reduction
* Starlet transformation
* Implement Masks (star mask, global threshold mask, etc.)
* RNC stretch
* Logging 


#### Dependencies:

- functools
- argparse
- numpy
- os
- datetime
- tqdm
- torch
- torchvision
- dcgan
- PIL
- typing
- glob
- opencv-python
- skimage