import time

from .base_agent import MetaAgent

from model import Model, get_search_space_class, LookUpTable, load_architecture, calculate_model_efficient


class MetaEvaluateAgent(MetaAgent):
    def _init_agent_state(self):
        """ Initialize for evaluating agent (Train from scratch).
        """
        # Construct model and correspond optimizer ======================================
        architecture = load_architecture(self.config["experiment_path"]["searched_model_path"])

        supernet_class, _ = get_search_space_class(self.config["agent"]["supernet_agent"])
        self.macro_cfg, self.micro_cfg = supernet_class.get_model_cfg(self.config["dataset"]["classes"])

        model = Model(
            self.macro_cfg,
            self.micro_cfg,
            architecture,
            self.config["dataset"]["classes"],
            self.config["dataset"]["dataset"])

        calculate_model_efficient(model, 3, self.config["dataset"]["input_size"], self.logger)

        self.model = model.to(self.device)
        self.model = self._parallel_process(self.model)

        self._optimizer_init(self.model)
        # =================================================================================

        # Resume checkpoint ===============================================================
        self._resume(self.model)


    def fit(self):
        """ Fit evalating process.
        """
        start_time = time.time()
        self.logger.info("Evaluating process start!")

        self.training_agent.train_loop(
            self.model,
            self.train_loader,
            self.test_loader,
            self)
        self.logger.info(f"Total search time : {time.time()-start_time:.2f}")


