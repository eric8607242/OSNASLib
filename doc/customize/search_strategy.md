# How to customize the search strategy
In OSNASLib, we cover serveral basic search strategy (e.g., differentiable search strategy, random search, evolution algorithm and architecture generator). Besides, we also provide flexible API interface to customize for any your specific tasks or the novel search strategy. In this document, we will briefly introduce how to customize the search strategy in `./search_strategy/` easily.


## Generate Interface
```
python3 build_interface.py -t search_strategy --customize-name [CUSTOMIZE NAME] --customize-class [CUSTOMIZE CLASS]
```

After generating the search strategy interface, the file `[CUSTOMIZE NAME].py` will be created in `searchstrategy/` and the corresponding import will be create automatically.

### Interface Struture
```
- search_strategy/
        |- [CUSTOMIZE NAME].py
            ...
```


## Search Strategy Interface
For customizing search strategy, the interface class `[CUSTOMIZE CLASS]Searcher` in `[CUSTOMIZE NAME.py` should inherited the base class `BaseSearcher`. You should implement `step()` and `search()` for searching the architectures. Besides, in `BaseSearcher`, we provide evaluating methods for evaluating the performance of the architectures.

```python3
# ./search_strategy/[CUSTOMIZE NAME].py
from .base import BaseSearcher

class {{customize_class}}Searcher(BaseSearcher):
    def __init__(self, config, supernet, val_loader, lookup_table, training_strategy, device, criterion, logger):
        super({{customize_class}}, self).__init__(config, supernet, val_loader, lookup_table, training_strategy, device, criterion, logger)

    def step(self):
        """ The searcher step before each iteration. 
        """
        pass

    def search(self):
        """ Searching the best architecture based on the hardware constraints and the supernet.

        Return:
            best_architecture (np.ndarray)
            best_architecture_hc (float)
            best_architecture_top1 (float)
        """
        return best_architecture, best_architecture_hc, best_architecture_top1


```

## Setting Config File
After customizing for your search space, you can utilize your search strategy by setting the search strategy in the config file easily.

```python
agent:
    search_strategy_agent: "[CUSTOMIZE CLASS]Searcher"
```
