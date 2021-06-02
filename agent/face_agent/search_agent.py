import time

from model import save_architecture

from .base_agent import FRMetaAgent

class FRSearchAgent(FRMetaAgent):
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
                self.val_loader)

        best_architecture, best_architecture_hc, best_architecture_top1 = self.search_strategy.search()

        self.logger.info(f"Best architectrue : {best_architecture}")
        self.logger.info(f"Best architectrue top1 : {best_architecture_top1*100:.3f}")
        self.logger.info(f"Best architectrue hc : {best_architecture_hc}")

        save_architecture(self.config["experiment_path"]["searched_model_path"], best_architecture)
        self.logger.info(f"Total search time : {time.time()-start_time:.2f}")


    def validate(self, model, val_loader, epoch):
        model.eval()
        start_time = time.time()

        minus_losses_avg = self.searching_evaluate(model, val_loader, self.device, self.criterion)

        self.writer.add_scalar("Valid/_losses/", -minus_losses_avg, epoch)

        self.logger.info(
            f"Valid : [{epoch+1:3d}/{self.epochs}]" 
            f"Final Losses: {-minus_losses_avg:.2f}"
            f"Time {time.time() - start_time:.2f}")

        return minus_losses_avg
