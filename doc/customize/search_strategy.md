# How to customize the search strategy

## API
```python
class CustomizerSearcher(BaseSearcher):
    def search(self):
        return best_architecture, best_architecture_hc, best_architecture_top1
```
