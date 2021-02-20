* [What is neural architecture search (NAS)](./nas.md)
* [What is one-shot NAS](./one_shot_nas.md)

# What is one-shot neural architecture search?
To improve the search efficiency of earliest NAS which evaluate each architecture by training it from scratch, one-shot NAS methods were proposed to encode the entire search space into an over-parameterized neural network, called ***a supernet***. The "one-shot" in one-shot NAS means that the only neural architecture needs to be trained to evaluate the entire search space instead of training each architecture from scratch.
Once the supernet is trained, all sub-networks in the supernet can be evaluated by inheriting the weights of the supernet without additional training. One-shot NAS methods can be divided into two categories : differentiable NAS (DNAS) and single-path NAS.

## Single-path one-shot neural architecture search
## Differentiable one-shot neural architecture search
Differentiable neural architecture search (DNAS) searches the best neural architecture by gradient descent. However, searching process is non-differentiable because the architectures in search space are discrete. To search the best architecture by gradient descent, DNAS utilizes additional differentiable parameters, called ***architecture parameters***, to indicate the architecture distribution in the search space. Besides, to make architecture parameters can indicate **the best architecture under the specific hardware constraint**, DNAS optimizes the architecture parameters with **the hardware constraint loss and cross entropy loss** at the same time.
