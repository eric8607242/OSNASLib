* [What is neural architecture search (NAS)](./nas.md)
* [What is one-shot NAS](./one_shot_nas.md)

# What is neural architecture search (NAS)?
By now, deep learning networks has brought surprising advances in various research field (e.g., image classification, semantic segmentation, and object detection). However, to boost the performance, the model architecture and the model size become more complex and huge significantly, which make the deep learning models are hard to deploy on the devices with limited hardware resource. But designing good neural networks manually specific to various hardware constraints requires much domain knowledge, rich experience on model training and tuning, and lots of time on trial and errors. Besides, the manually designed architectures are often stuck into the local optimal.
To get the better trade off between the hardware resource and the performance, neural architecture search (NAS) aiming at automatically searching the best architecture under the specific hardware constraints is highly demanded.

## The main components of neural architecture search (NAS)
Generally NAS methods can be categorized according to three dimensions: Search space, Search strategy, and Performance estimation strategy.
### Search Space
For the search space, it mean that the **all candidate neural architectures can be selected** (kernel sizes, channel size, layer size, and etc.). 
### Search Strategy
Given a search space, the objective of neural architecture search is to **search the best architecture under the specific hardware constraint**. To achieve the goal, a search strategy is needed to be adopted to search the best architecture from the search space. The common search strategies are RL, random search, evolution algorithm, and differentiable.
### Performance estimation strategy
With the search strategy, **how do we evaluate the architectures in the search space? In other words, how do we know the performace of each architectures?**. To evaluate each architectures in the search space, the performance estimation strategy has to been adopted. The most intuitive estimation strategy is that we train each architecture few epochs (e.g., 10, 5, or 20) to get the approximated perofmrnace of each architecture.



