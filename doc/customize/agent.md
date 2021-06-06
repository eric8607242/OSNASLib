# How to customize the agent
To supprot searching architectures on various tasks (e.g., classification, face recognition, and object detection), OSNASLib allow users to customize the agent for different training pipeline. In this document, will briefly introduce how to customize the agent for your task in `agnet/` easily.


## Generate Template
```
python3 build_template.py -t agent --customize-name [CUSTOMIZE NAME] --customize-class [CUSTOMIZE CLASS]
```

After generating the agent template, the directory `[CUSTOMIZE NAME]/` will be created in `agent/`, and the corresponding files (`__init__.py`, `base_agent.py`, `search_agent.py`, and `evaluate_agent.py`) are created in the directory `[CUSTOMIZE NAME]/`.


## Agent Interface
For customizing agent, you should implement three interface class, `[CUSTOMIZE CLASS]MetaAgent`, `[CUSTOMIZE CLASS]EvaluateAgent`, and `[CUSTOMIZE CLASS]SearchAgent`, respectively. We describe the three interface classes in detail as follows:

### Base Agent Interface
`[CUSTOMIZE CLASS]MetaAgent` is the abstract class for search agent and evaluate agent. In `[CUSTOMIZE CLASS]MetaAgent`, you shoud implement the `_training_step()`, `_validate()`, and `searching_evaluate()` for your task.

```python3
from utils import AverageMeter, accuracy, save

from ..base_agent import MetaAgent

class [CUSTOMIZE CLASS]MetaAgent(MetaAgent):
    """[CUSTOMIZE CLASS] meta agent
    """
    evaluate_metric = ""
    def _training_step(
            self,
            model,
            train_loader,
            epoch,
            print_freq=100):
        losses = AverageMeter()

        model.train()
        start_time = time.time()

        for step, datas in enumerate(train_loader):
            self._iteration_preprocess()

            # Write your code here

            # ====================

            losses.update(loss.item(), N)
            if (step > 1 and step % print_freq == 0) or (step == len(train_loader) - 1):
                self.logger.info(f"Train : [{(epoch+1):3d}/{self.epochs}] "
                                 f"Step {step:3d}/{len(train_loader)-1:3d} Loss {losses.get_avg():.3f} ")

        self.writer.add_scalar("Train/_loss/", losses.get_avg(), epoch)
        self.logger.info(
            f"Train: [{epoch+1:3d}/{self.epochs}] Final Loss {losses.get_avg():.3f} " 
            f"Time {time.time() - start_time:.2f} ")

    def _validate(self, model, val_loader, epoch):
        model.eval()
        start_time = time.time()

        # Writer your code here

        # =====================

        self.writer.add_scalar("Valid/_losses/", -minus_losses_avg, epoch)

        self.logger.info(
            f"Valid : [{epoch+1:3d}/{self.epochs}]" 
            f"Final Losses: {-minus_losses_avg:.2f}"
            f"Time {time.time() - start_time:.2f}")


    @staticmethod
    def searching_evaluate(model, val_loader, device, criterion):
        pass
```

### Evaluate Agent Interface
After searching the architecture, `[CUSTOMIZE CLASS]EvaluateAgent` can evaluate the searched architecture by training from scratch in the full dataset. Therefore, in `[CUSTOMIZE CLASS]EvaluateAgent`, you shoud implement the `_evaluate()`, and `_inference()` for your task.

```python3
from .base_agent import [CUSTOMIZE CLASS]MetaAgent

class [CUSTOMIZE CLASS]EvaluateAgent([CUSTOMIZE CLASS]MetaAgent):
    """[CUSTOMIZE CLASS] evaluate agent
    """
    agent_state = "evaluate_agent"
    def fit(self):
        self._evaluate()
        self._inference()

    def _iteration_preprocess(self):
        pass

    def _evaluate(self):
        pass

    def _inference(self):
        pass
```

### Search Agent Interface
`[CUSTOMIZE CLASS]SearchAgent` can search the architecture for your task. For searching the architecture, you should implement the `_iteration_preprocess()` and `_search()` for your task.

```python3
from model import save_architecture

from .base_agent import [CUSTOMIZE CLASS]MetaAgent

class [CUSTOMIZE CLASS]SearchAgent([CUSTOMIZE CLASS]MetaAgent):
    """ [CUSTOMIZE CLASS] search agent
    """
    agent_state = "search_agent"
    def fit(self):
        self._search()

    def _iteration_preprocess(self):
        pass

    def _search(self):
        pass
```
