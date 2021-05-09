# How to customize your training strategy
In OSNASLib, we cover serveral basic training strategy (e.g., differentiable training strategy, uniform training strategy and fairstrictness training strategy) to train the supernet. Besides, we also provide flexible API interface to customize for any your specific tasks or the novel training strategy. In this document, we will briefly introduce how to customize the training strategy in `./training_strategy/` easily.

## Create `customize.py` File
First of all, you should create a new file for your training strategy.
```
touch ./training_strategy/customize.py
```

## Training Strategy Class
For customizing the training strategy, you can utilize the API following in your training strategy file.
In OSNASLib, all training strategy inherited by `BaseSampler` in `./training_strategy/base.py`.

```python
# ./training_strategy/customize.py
from .base import BaseSampler

class CustomizerSampler(BaseSampler):
    def generate_training_architecture(self):
        """
            In this function, the sampler should generate an architecture and return. 
            With the generated architecture, the sampler can set activate architecture 
            in the supernet to forward and update.
            
        """
        return architecture
    
    def step(self):
        """
            During training supernet, this function will be called before each supernet 
            training iteration to set the activate architecture in the supernet.
        """
        architecture = self.generate_training_architecture()
```

## Import Your Training Strategy
After creating your training strategy, the class should be imported to in `./training_strategy/__init__.py`.


```python
# ./training_strategy/__init__.py
from .customizer import CustomizerSampler
```

## Setting The Training Strategy Agent In Config File
To utilize the customizing training strategy, you should set the agent in the config file as following.

```python
agent:
    training_strategy_agent: "CustomizerSampler"
```
