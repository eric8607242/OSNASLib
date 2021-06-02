import time

from model import save_architecture

from .base_agent import CFMetaAgent

class CFSearchAgent(CFMetaAgent):
    """ Classification search agent
    """
    agent_state = "search_agent"
    def fit(self):
        self.search()

    def _iteration_preprocess(self):
        self.search_strategy.step()
        self.training_strategy.step()

    def search(self):
        start_time = time.time()
        self.logger.info("Searching process start!")

        if not self.config["search_utility"]["directly_search"]:
            self.train_loop(
                self.supernet,
                self.train_loader,
                self.val_loader,
                self.optimizer,
                self.lr_scheduler)

        best_architecture, best_architecture_hc, best_architecture_top1 = self.search_strategy.search()

        self.logger.info("Best architectrue : {}".format(best_architecture))
        self.logger.info(
            "Best architectrue top1 : {:.3f}".format(
                best_architecture_top1 * 100))
        self.logger.info("Best architectrue hc : {}".format(best_architecture_hc))

        save_architecture(self.config["experiment_path"]["searched_model_path"], best_architecture)
        self.logger.info(
            "Total search time : {:.2f}".format(
                time.time() - start_time))


