import time

import torch.nn as nn

from .base_agent import MetaAgent

from model import get_search_space_class, save_architecture

from search_strategy import get_search_strategy
from training_strategy import get_training_strategy

class MetaSearchAgent(MetaAgent):
    def _init_agent_state(self):
        """ Initialize for searching process.
        """
        supernet_class, lookup_table_class = get_search_space_class(self.config["agent"]["search_space_agent"])
        # Construct model and correspond optimizer ======================================
        self.supernet = supernet_class(
            self.config["dataset"]["classes"],
            self.config["dataset"]["dataset"],
            bn_momentum=self.config["train"]["bn_momentum"],
            bn_track_running_stats=self.config["train"]["bn_track_running_stats"])
        self.macro_cfg, self.micro_cfg = self.supernet.get_model_cfg(self.config["dataset"]["classes"])
        
        self._resume(self.supernet)
        
        self.supernet.to(self.device)
        self.supernet = self._parallel_process(self.supernet)

        self._optimizer_init(self.supernet, self.criterion)

        # Construct search utility ========================================================
        training_strategy_class = get_training_strategy(self.config["agent"]["training_strategy_agent"])
        self.training_strategy = training_strategy_class(self.supernet)

        macro_len, micro_len = self.supernet.module.get_model_cfg_shape() \
                                    if isinstance(self.supernet, nn.DataParallel) else self.model.get_model_cfg_shape()
        self.lookup_table = lookup_table_class(
            self.macro_cfg,
            self.micro_cfg,
            macro_len,
            micro_len,
            self.config["experiment_path"]["lookup_table_path"],
            self.config["dataset"]["input_size"],
            info_metric=self.config["search_utility"]["info_metric"])

        search_strategy_class = get_search_strategy(self.config["agent"]["search_strategy_agent"])
        self.search_strategy = search_strategy_class(
                self.config, 
                self.supernet, 
                self.val_loader, 
                self.lookup_table, 
                self.training_strategy, 
                self.device, self.criterion, self.logger)

        # Resume checkpoint ===============================================================
        
    def fit(self):
        """ Fit searching process.
        Training the supernet and searching the architecture by the search strategy.
        """
        start_time = time.time()
        self.logger.info("Searching process start!")

        if not self.config["search_utility"]["directly_search"]:
            self.training_agent.train_loop(
                self.supernet,
                self.train_loader,
                self.val_loader,
                self)

        best_architecture, best_architecture_hc, best_architecture_performance = self.search_strategy.search()

        self.logger.info(f"Best architectrue : {best_architecture}")
        self.logger.info(f"Best architectrue performance : {best_architecture_performance:.3f}")
        self.logger.info(f"Best architectrue hc : {best_architecture_hc}")

        save_architecture(self.config["experiment_path"]["searched_model_path"], best_architecture)
        self.logger.info(f"Total search time : {time.time()-start_time:.2f}")

    def _iteration_preprocess(self):
        """ Process at the begin of each iteration.
        Step the search strategy and training strategy (e.g., set activate path in supernet).
        """
        self.search_strategy.step()
        self.training_strategy.step()
