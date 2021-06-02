import time

from .base_agent import CFMetaAgent

class CFEvaluateAgent(CFMetaAgent):
    """Classification evaluate agent
    """
    agent_state = "evaluate_agent"
    def fit(self):
        self.evaluate()
        self.inference()

    def _iteration_preprocess(self):
        pass

    def evaluate(self):
        start_time = time.time()
        self.logger.info("Evaluating process start!")

        self.train_loop(
            self.model,
            self.train_loader,
            self.test_loader)
        self.logger.info(f"Total search time : {time.time()-start_time:.2f}")

    def inference(self):
        start_time = time.time()
        top1_acc = self.validate(self.model, self.test_loader, 0)
        self.logger.info(f"Final Top1 accuracy : {top1_acc}")
        self.logger.info(f"Total inference time : {time.time()-start_time:.2f}")
