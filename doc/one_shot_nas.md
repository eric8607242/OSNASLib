# What Is One-shot NAS 

* [What Is Neural Architecture Search (NAS)](./nas.md)
* [What Is One-shot NAS](./one_shot_nas.md)

To improve searching efficiency, many recent NAS approaches are based on one-shot scheme. Instead of training thousands of architectures to get the true performance of the entire search space, the main idea of one-shot NAS is constructing a supernet to represent performance of the entire search space. Therefore, the term 'one-shot' in one-shot NAS means that only one neural architecture needs to be trained to evaluate the entire search space. Once the supernet is trained, all sub-networks in the supernet can be evaluated by inheriting the weights of the supernet without additional training. To demonstrate the details of how a supernet works, we illustrate supernet construction in the following figure.
![supernet_encode](../resource/supernet_encode.png)
> Illustration of supernet construction. The blocks in different colors denote the blocks of different configurations in the search space (e.g., convolution blocks with different kernel sizes {3, 5, 7}). In the supernet, all candidate blocks are constructed layer by layer. Therefore, by activating different blocks in different layers, the supernet can represent all architectures in the entire search space easily.

Therefore, how to train a supernet and how to search the best architecture are very important for one-shot NAS. Such one-shot methods can be generally divided into two categories: differentiable NAS and single-path NAS. Below we will briefly introduce differentiable NAS and single-path NAS.



## Differentiable One-shot NAS
Given a supernet <img src="https://render.githubusercontent.com/render/math?math=A"> represented by weights <img src="https://render.githubusercontent.com/render/math?math=\boldsymbol{w}">, to find an architecture that achieves the best performance while meeting a specific hardware constraint, we need to find the best sub-network <img src="https://render.githubusercontent.com/render/math?math=a^*"> from <img src="https://render.githubusercontent.com/render/math?math=A"> which achieves the minimum validation loss <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{val}(a, \boldsymbol{w})">. Sampling <img src="https://render.githubusercontent.com/render/math?math=a"> from <img src="https://render.githubusercontent.com/render/math?math=A"> is a non-differentiable process. To optimize <img src="https://render.githubusercontent.com/render/math?math=a"> by the gradient descent algorithm, DNAS relaxes the non-differentiable problem as finding a set of continuous architecture parameters <img src="https://render.githubusercontent.com/render/math?math=\boldsymbol{\alpha}">. Some methods compute the values for weighting output of candidate blocks by the softmax function:

<div style="text-align:center"><img src="https://render.githubusercontent.com/render/math?math=x_{l+1} = \sum_i m^i_l\cdot b^i_l(x_l),"></div>
<div style="text-align:center"><img src="https://render.githubusercontent.com/render/math?math=m^i_l = \frac{exp(\alpha^i_l)}{\sum^K_{k=1}exp(\alpha^k_l)},"></div>


where <img src="https://render.githubusercontent.com/render/math?math=x_l"> is the input tensor of the <img src="https://render.githubusercontent.com/render/math?math=l">-th layer, <img src="https://render.githubusercontent.com/render/math?math=b_l^i"> is the <img src="https://render.githubusercontent.com/render/math?math=i">-th block of the <img src="https://render.githubusercontent.com/render/math?math=l">-th layer, and thus <img src="https://render.githubusercontent.com/render/math?math=b^i_l(x_l)"> denotes the output of the <img src="https://render.githubusercontent.com/render/math?math=i">-th block. The term <img src="https://render.githubusercontent.com/render/math?math=\alpha_l^i"> is the weight of the <img src="https://render.githubusercontent.com/render/math?math=i">-th block in the <img src="https://render.githubusercontent.com/render/math?math=l">-th layer. The value <img src="https://render.githubusercontent.com/render/math?math=m_l^i"> is the weight for the output <img src="https://render.githubusercontent.com/render/math?math=b^i_l(x_l)">. 

Some methods compute the values for weighting output of candidate blocks by the Gumbel softmax function: 
<div style="text-align:center"><img src="https://render.githubusercontent.com/render/math?math=m^i_l = \frac{exp(\alpha^i_l+g^i_l/\tau)}{\sum^K_{k=1}exp(\alpha^k_l+g^k_l/\tau)},"></div>

where the term <img src="https://render.githubusercontent.com/render/math?math=g^i_l"> is a random variable sampled from the Gumbel distribution <img src="https://render.githubusercontent.com/render/math?math=Gumbel(0, 1)">, and <img src="https://render.githubusercontent.com/render/math?math=\tau"> is the temperature parameter.


After relaxation, DNAS can be formulated as a bi-level optimization: 

<div style="text-align:center"><img src="https://render.githubusercontent.com/render/math?math=\boldsymbol{\alpha}^* = \operatorname*{min}_{\boldsymbol{\alpha}}\mathcal{L}_{val}(\boldsymbol{w}^*, \boldsymbol{\alpha})"></div>
<div style="text-align:center"><img src="https://render.githubusercontent.com/render/math?math=\text{s.t. }\boldsymbol{w}^* = \operatorname*{argmin}_{\boldsymbol{w}}\mathcal{L}_{train}(\boldsymbol{w}, \boldsymbol{\alpha})"></div>

where <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{train}(\boldsymbol{w}, \boldsymbol{\alpha})"> is the training loss. 

Because of the bi-level optimization of $\boldsymbol{w}$ and $\boldsymbol{\alpha}$, the best architecture $\boldsymbol{\alpha}^*$ sampled from the supernet is only suitable to a specific hardware constraint. With this searching process, for $N$ different hardware constraints, the supernet and architecture parameters should be retrained for $N$ times. This makes DNAS less flexible. 

We illustrate the architecture parameters updating in the following figure.
![dnas](../resource/dnas.png "")


## Single-path One-shot NAS
Single-path methods decouple supernet training from architecture searching. For supernet training, only a single path consisting of one block in each layer is activated and is optimized in one iteration to simulate discrete neural architecture in the search space. We can formulate the process as: 

<div style="text-align:center"><img src="https://render.githubusercontent.com/render/math?math=\boldsymbol{w}^* = \operatorname*{argmin}_{\boldsymbol{w}}\mathbb{E}_{a\sim \Gamma(A)}(\mathcal{L}_{train}(\boldsymbol{w}(a)))"></div>

where 
<img src="https://render.githubusercontent.com/render/math?math=\boldsymbol{w}(a)"> denotes the subset of <img src="https://render.githubusercontent.com/render/math?math=\boldsymbol{w}"> corresponding to the sampled architecture <img src="https://render.githubusercontent.com/render/math?math=a">, and <img src="https://render.githubusercontent.com/render/math?math=\Gamma(A)"> is a prior distribution of 
<img src="https://render.githubusercontent.com/render/math?math=a \in A"> (e.g., uniform distribution). The best weights <img src="https://render.githubusercontent.com/render/math?math=\boldsymbol{w}^*"> to be determined are the ones yielding the minimum expected training loss. After training, the supernet is treated as a performance estimator to all architectures in the search space. With the pretrained supernet weights <img src="https://render.githubusercontent.com/render/math?math=\boldsymbol{w}^*">, we can search the best architecture 
<img src="https://render.githubusercontent.com/render/math?math=a^*">: 

<div style="text-align:center"><img src="https://render.githubusercontent.com/render/math?math=a^* = \operatorname*{argmin}_{a \in A}\mathcal{L}_{val}(\boldsymbol{w}^*(a))"></div>

Single-path methods are more flexible than DNAS, because supernet training and architecture search are decoupled. Once the supernet is trained, for $N$ different constraints, only architecture search should be conducted for $N$ times.

We illustrate the single-paht NAS the following figure.
![single-path](../resource/single_path.png "")
