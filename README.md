# DeepSurvival

<p align='center'><i>Deep Learning models for Survival Analysis</i></p>
<p align='center'>
	<img src="https://img.shields.io/badge/Python_3.8%2B-3776AB.svg?style=flat-square&logo=python&color=3776AB&logoColor=white" alt="Python 3.8+">
	<img src="https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
	<img src="https://img.shields.io/badge/paper-DeepSurv-0F766E.svg?style=flat-square&logo=readthedocs&logoColor=white" alt="Survival Analysis">
</p>

This repo comes at first as an attempt to replicate *DeepSurv*[^1] using pytorch, a first modern-deep learning approach that has been proved to be effective. _To be extended to replicate other models_  

This approach is cool in a sense it's still based on Cox Proportional Hazard Model, and can accomodate for neural network architectires to predict time-to-event data. 
Opening up also the possibilities of enhancements in deep learning survival models as defined in subsequent models like *Cox-Time*[^2] that let go of the proportional hazard assumption, and *LogisticHazard* that is based on a discrete-time model.  

> [!TIP]
> a super comprehensive list of deep learning models for survival analysis can be found in this github repo [survival-org/DL4Survival](https://github.com/survival-org/DL4Survival) from Wiegrebe et al. (2024) [^4].

## Ideas

The model is in fact pretty simple, and requires some understanding of how Cox Model works, and some ideas on neural network optimization. 
As this will be a pytorch implementation concepts will be portrayed in a raw form within code, involving components like creating a custom loss function.

### CPHM
As defined by the authors [^1], CPHM assumed a linear combination of covariates, and the hazard function is defined as: 
$$h(t|x) = h_0(t) \exp(\beta^T x)$$

Where $h_0(t)$ is the baseline hazard function, and $\beta$ is the vector of coefficients.
Slightly similar to linear models, the optimization of the model's coefficients is done through maximizing *partial likelihood*, which is defined as:
$$PL(\beta) = \prod_{i=1}^n \left( \frac{\exp(\beta^T x_i)}{\sum_{j \in R(t_i)} \exp(\beta^T x_j)} \right)^{\delta_i}$$
here $R(t_i)$ is the risk set at time $t_i$ (those still alive), and $\delta_i$ is the event indicator (1 if the event occurred, 0 if censored), which is an important addition in implementing the network to be computed.

### DeepSurv
The idea of DeepSurv is to replace the linear combination of covariates with a non-linear function, pretty much like a neural network replacing a linear model.
The hazard function is defined as:
$$h(t|x) = h_0(t) \exp(g(x))$$
where here what's new is the introduction of $g(x)$ that is a non-linear function of the covariates, and can be implemented as a _neural network_.
Very much like _maximizing partial likelihood_, but reversed to suit the neural network setting, the objective function here is a loss to be minimized, that is defined as the *negative log partial likelihood*:
$$\ell(\theta) = -\sum_{i=1}^n \delta_i \left( g(x_i) - \log \sum_{j \in R(t_i)} \exp(g(x_j)) \right)$$
where $\theta$ are the parameters of the nn, and $g(x)$ is the output of the nn for input $x$. 

> [!NOTE]
> the summation is equivalent to the log transformation fo the product in the original partial likelihood, and the negative sign is there to convert the maximization problem into a minimization one, since we're looking to minimize a *loss*


## Implementation

Essentially, the main components are:
- neural network model class inheriting from `torch.nn.Module` (as is the case for any pytorch model) $\iff g(x)$
- custom loss function for the negative log partial likelihood $\iff \ell(\theta)$
- training loop to optimize the model parameters using the defined loss function


## Beyond `DeepSurv`





[^1]: Katzman, Jared L., et al. "DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network." BMC medical research methodology 18.1 (2018): 1-12.

[^2]: Kvamme, Håvard, Ørnulf Borgan, and Ida Scheel. "Time-to-event prediction with neural networks and Cox regression." Journal of Machine Learning Research 20.129 (2019): 1-30.

<!-- [^3]: -->

[^4]: Wiegrebe, S., Kopper, P., Sonabend, R., Bischl, B., & Bender, A. (2024). Deep learning for survival analysis: a review. Artificial Intelligence Review, 57(3), 65.

