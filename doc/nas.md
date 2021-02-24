* [What is neural architecture search (NAS)](./nas.md)
* [What is one-shot NAS](./one_shot_nas.md)

# What is neural architecture search (NAS)?
By now, deep learning networks has brought surprising advances in various research field (e.g., image classification, semantic segmentation, and object detection). However, it is very difficult to design a good neural architecture for a specific hardware constraint (or devices with limited hardware resource), which requires much domain knowledge and lots of time. To reduce the effort of designing neural architecutre and get the better neural architecture, neural architecture search (NAS) has recevied much attention in recent years.

## The main components of neural architecture search (NAS)
Generally NAS methods can be categorized according to three dimensions: search space, search strategy, and performance estimation strategy.
### Search Space
There are lots of configuration can be selected to design a good neural architecture (e.g., kernel size, layer size, and channel size). For the search space, it means that the **all candidate neural architectures composed by all configurations**.
For example, for VGG16, the search space size is 3\*\*16 if the kernel size in each layer can be implemented by {3, 5, 7}.

The common search spaces in recent years are constructed by the inverted residual blocks with linear bottlenecks (MobilenetV2) and the DAG search space (DATRS).

### Search Strategy
Given a search space, the objective of neural architecture search is to **search the best architecture under the specific hardware constraint**. 
To achieve the goal, a search strategy is demanded to be adopted to search the best architecture from the search space. The common search strategies are RL, random search, evolution algorithm, and gradient descent (differentiable NAS).
### Performance estimation strategy
With the search strategy, **how can we evaluate the architectures in the search space? In other words, how do we know the performace of each architectures?**. To evaluate each architectures in the search space, the performance estimation strategy has to been adopted. The most intuitive estimation strategy is training each architecture few epochs (e.g., 10, 5, or 20) to get the approximated perofmrnace of each architecture or training a performance predictor to predict the accuracy of each architecture.
However, performance estimation is still very time-consuming because tons of network training are required, which makes NAS is not approachable. One-shot NAS and zero-shot NAS have been proposed to save large amount of time.

In this repo, we focus on one-shot NAS. You can implement each baseline of one-shot NAS easily by our repo.


