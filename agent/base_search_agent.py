import time

from .base_agent import MetaAgent

from model import get_supernet_class, LookUpTable, save_architecture

from search_strategy import get_search_strategy
from training_strategy import get_training_strategy

class MetaSearchAgent(MetaAgent):
    def _init_agent_state(self):
        """ Initialize for searching process.
        """
        # Construct model and correspond optimizer ======================================
        supernet = self._construct_supernet()
        self.macro_cfg, self.micro_cfg = supernet.get_model_cfg(self.config["dataset"]["classes"])
        
        self.supernet = supernet.to(self.device)
        self.supernet = self._parallel_process(self.supernet)

        self._optimizer_init(self.supernet)

        # Construct search utility ========================================================
        training_strategy_class = get_training_strategy(self.config["agent"]["training_strategy_agent"])
        self.training_strategy = training_strategy_class(len(self.micro_cfg), len(self.macro_cfg["search"]), self.supernet)

        self.lookup_table = LookUpTable(
            self.macro_cfg,
            self.micro_cfg,
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
        self._resume(self.supernet)

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

        best_architecture, best_architecture_hc, best_architecture_top1 = self.search_strategy.search()

        self.logger.info(f"Best architectrue : {best_architecture}")
        self.logger.info(f"Best architectrue top1 : {best_architecture_top1*100:.3f}")
        self.logger.info(f"Best architectrue hc : {best_architecture_hc}")

        save_architecture(self.config["experiment_path"]["searched_model_path"], best_architecture)
        self.logger.info(f"Total search time : {time.time()-start_time:.2f}")

    def _iteration_preprocess(self):
        """ Process at the begin of each iteration.
        Step the search strategy and training strategy (e.g., set activate path in supernet).
        """
        self.search_strategy.step()
        self.training_strategy.step()
