# How to customize the agent
To supprot searching architectures on various tasks (e.g., classification, face recognition, and object detection), OSNASLib allow users to customize the training agent for different training pipeline. With the the customizing training agent, the search agent and evaluate agent can incorporate different training pipeline to search the architectures and evaluate the searched architecture by scratch. In this document, will briefly introduce how to customize the agent for your task in `agnet/` easily.

## Generate Template
```
python3 build_template.py -t agent --customize-name [CUSTOMIZE NAME] --customize-class [CUSTOMIZE CLASS]
```

After generating the agent template, the directory `[CUSTOMIZE NAME]/` will be created in `agent/`, and the corresponding files (`__init__.py`, `training_agent.py`, and `agents.py`) are created in the directory `[CUSTOMIZE NAME]/`.


## Agent Interface
For customizing agent for various tasks, you should implement the training interface `[CUSTOMIZE CLASS]TrainingAgent` with the training pipeline. We describe the three interface classes in detail as follows:

### Training Agent Interface
`[CUSTOMIZE CLASS]TrainingAgent` is the training agent implemented the training pipeline specific for customizing task. In `[CUSTOMIZE CLASS]TrainingAgent`, you shoud implement the `_search_training_step()`, `_search_validate_step()`, `_evaluate_training_step()`, `_evaluate_validate_step()`, and `searching_evaluate()` for your task.
Note that we provide the example training pipeline in the template. You shoud modify the training pipeline to specific for your task.

```python3
import os
import time

import torch

from utils import AverageMeter, save

class {{customize_class}}TrainingAgent:
    """The training agent to train the supernet and the searched architecture.

    By implementing TrainingAgent class, users can adapt the searching and evaluating agent into
    various tasks easily.
    """
    def train_loop(self, model, train_loader, val_loader, agent):
        """ The main training loop.

        Args:
            model (nn.Module)
            train_loader (torch.utils.data.DataLoader)
            val_loader (torch.utils.data.DataLoader)
            agent (Object)
        """
        # Utilize different step method based on differet agent state
        training_step = getattr(self, f"_{agent.agent_state}_training_step")
        validate_step = getattr(self, f"_{agent.agent_state}_validate_step")

        best_val_performance = -float("inf")
        for epoch in range(agent.start_epochs, agent.epochs):
            agent.logger.info(f"Start to train for epoch {epoch}")
            agent.logger.info(f"Learning Rate : {agent.optimizer.param_groups[0]['lr']:.8f}")

            training_step(
                model,
                train_loader,
                agent,
                epoch)
            val_performance = validate_step(
                model,
                val_loader,
                agent,
                epoch)

            if val_performance > best_val_performance:
                agent.logger.info(f"Best validation performance : {val_performance}. Save model!")
                best_val_performance = val_performance
                save(
                    model,
                    agent.config["experiment_path"]["best_checkpoint_path"],
                    agent.optimizer,
                    agent.lr_scheduler,
                    epoch + 1)

            save(
                model,
                os.path.join(
                    agent.config["experiment_path"]["checkpoint_root_path"],
                    f"{agent.agent_state}_{epoch}.pth"),
                agent.optimizer,
                agent.lr_scheduler,
                epoch + 1)


    def _search_training_step(self, model, train_loader, agent, epoch):
        """ The training step for searching process. Users should step the sampler
        to decide how to train supernet and step the search strategy to search the architecture.

        Args:
            model (nn.Module)
            train_loader (torch.utils.data.DataLoader)
            agent (Object): The search agent.
            epoch (int)
        """
        pass

    def _search_validate_step(self, model, val_loader, agent, epoch):
        """ The validate step for searching process.

        Args:
            model (nn.Module)
            val_loader (torch.utils.data.DataLoader)
            agent (Object): The search agent.
            epoch (int)

        Return:
            evaluate_metric (float): The performance of the supernet
        """
        return evaluate_metric

    def _evaluate_training_step(self, model, train_loader, agent, epoch):
        """ The training step for evaluating process (training from scratch).

        Args:
            model (nn.Module)
            train_loader (torch.utils.data.DataLoader)
            agent (Object): The evaluate agent
            epoch (int)
        """
        pass

    def _evaluate_validate_step(self, model, val_loader, agent, epoch):
        """ The training step for evaluating process (training from scratch).

        Args:
            model (nn.Module)
            val_loader (torch.utils.data.DataLoader)
            agent (Object): The evaluate agent
            epoch (int)

        Return:
            evaluate_metric (float): The performance of the searched model.
        """
        return evaluate_metric

    @staticmethod
    def searching_evaluate(model, val_loader, device, criterion):
        """ Evaluating the performance of the supernet. The search strategy will evaluate
        the architectures by this static method to search.

        Args:
            model (nn.Module)
            val_loader (torch.utils.data.DataLoader)
            device (torch.device)
            criterion (nn.Module)

        Return:
            evaluate_metric (float): The performance of the supernet.
        """
        return evaluate_metric
```

### Agents Interface
After constructing the train agent, in the `agents.py`, the `[CUSTOMIZE CLASS]SearchAgent` and the `[CUSTOMIZE CLASS]EvaluateAgent` are created by inherited the abstract class `MetaSearchAgent` and `MetaEvaluateAgent`. And in the customizing agents, the train agent is set as the property to incorporate into the searching pipeline and evaluating pipeline. 
> Note that this file and the corresponding codes are created automatically.

```python3
from .training_agent import {{customize_class}}TrainingAgent
from ..base_search_agent import MetaSearchAgent
from ..base_evaluate_agent import MetaEvaluateAgent

class {{customize_class}}SearchAgent(MetaSearchAgent):
    agent_state = "search"
    training_agent = {{customize_class}}TrainingAgent()

class {{customize_class}}EvaluateAgent(MetaEvaluateAgent):
    agent_state = "evaluate"
    training_agent = {{customize_class}}TrainingAgent()
```

## Setting Config File
After constructing agent for your task, you can utilize your agent by setting the agent into the config file easily.
### Search Mode
```
agent:
    main_agent: "[CUSTOMIZE CLASS]SearchAgent"
```

### Evlauate Mode
```
agent:
    main_agent: "[CUSTOMIZE CLASS]EvaluateAgent"
```
