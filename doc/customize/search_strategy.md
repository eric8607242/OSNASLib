# How to customize the search strategy
In OSNASLib, we cover serveral basic search strategy (e.g., differentiable search strategy, random search, evolution algorithm and architecture generator). Besides, we also provide flexible API interface to customize for any your specific tasks or the novel search strategy. In this document, we will briefly introduce how to customize the search strategy in `./search_strategy/` easily.


## Create `customize.py` File
First of all, you should create a new file for your search strategy.
```
touch ./search_strategy/customize.py
```

## Search Strategy Class
For customizing the search strategy, you can utilize the API following in your search strategy file.
In OSNASLib, all search strategy inherited by `BaseSearcher` in `./search_strategy/base.py`.

```python
# ./search_strategy/customize.py
class CustomizerSearcher(BaseSearcher):
    def search(self):
        """
            After training supernet, this function will be called to search the best neural architecture.
        """
        return best_architecture, best_architecture_hc, best_architecture_top1
    
    def step(self):
        """
            During training supernet, this function will be called after each supernet training iteration.
        """
        pass
```


## Import Your Search Strategy
After creating your search strategy, the class should be imported to in `./search_strategy/__init__.py`.


```python
# ./search_strategy/__init__.py
from .customizer import CustomizerSearcher
```

## Setting The Search Strategy Agent In Config File
To utilize the customizing search strategy, you should set the agent in the config file as following.

```python
agent:
    search_strategy_agent: CustomizerSearcher
```
