# Reference
[Conference Paper](https://ieeexplore.ieee.org/document/8011107/)

# Action Recognition using MHI Based Hu Moments With HMMs
Action video recognition with HMM. In this model training and testing used HMM (Framework [HMMLearn](https://github.com/hmmlearn/hmmlearn).) Modified HMM as a feature og HMM. Modified HMM is extansion of [Bobick and Davis's MHI](https://ieeexplore.ieee.org/document/910878/).

Model train and test with Weizmann dataset.

[Weizmann](http://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html)

## Pre-request libraries
* Python3
* Numpy
* Opencv2 for python3
* HMMLearn
* Scipy
* Sklearn

## Train and Test Model
Run the HMM train and test model.
```
 hmmTrainTest.py [-h] [-f FEATURE_TYPE] [-g GMM_STATE_NUMBER]
                        [-s STATE_NUMBER] [-p PREPROCESS_METHOD]
                        [-dc DECOMPOSITION_COMPONENT] [-r RESIZE] [-w WINDOW]
                        [-l2r LEFT2RIGHT] [-mhi MHI]'
 
 ```
 Run the default options with.
 
 `python hmmTrainTest.py`
 
## Options
```
  optional arguments:
   -h, --help            show this help message and exit
   -f FEATURE_TYPE, --feature-type FEATURE_TYPE
                         Feature type. * "Hu" *"Projection" (default: Hu)
   -g GMM_STATE_NUMBER, --gmm-state-number GMM_STATE_NUMBER
                         Number of states in the GMM. (default: 1)
   -s STATE_NUMBER, --state-number STATE_NUMBER
                         Number of states in the model. (default: 7)
   -p PREPROCESS_METHOD, --preprocess-method PREPROCESS_METHOD
                         Data preprocess method.* "PCA" *"StandardScaler"
                         *"FastICA" *"Normalizer" (default: FastICA)
   -dc DECOMPOSITION_COMPONENT, --decomposition-component DECOMPOSITION_COMPONENT
                         Principal axes in feature space, representing the
                         directions of maximum variance in the data. The
                         components are sorted by ``explained_variance_``.
                         (default: 7)
   -r RESIZE, --resize RESIZE
                         Frame resize ratio [0 - 1]. (default: 1)
   -w WINDOW, --window WINDOW
                         Frame window size. (default: 30)
   -l2r LEFT2RIGHT, --left-2-right LEFT2RIGHT
                         Left to right HMM model. (default: False)
   -mhi MHI, --mhi MHI   Do use MHI Feature extraction? (default: True)
   ```

