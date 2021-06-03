import time
import numpy as np

import torch

from .base_agent import FRMetaAgent
from .face_evaluate import evaluate

class FREvaluateAgent(FRMetaAgent):
    """Face recognition agent
    """
    agent_state = "evaluate_agent"
    def _iteration_preprocess(self):
        pass

    def fit(self):
        self.evaluate()
        self.inference()

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
        acc_avg = self.validate(self.model, self.test_loader, 0)
        self.logger.info(f"Final Acc : {acc_avg}")
        self.logger.info(f"Total inference time : {time.time()-start_time:.2f}")

    def validate(self, model, val_loader, epoch):
        model.eval()
        start_time = time.time()

        acc_avg, thresh_avg = self._validate_step(model, val_loader)

        self.writer.add_scalar("Valid/_acc/", acc_avg, epoch)
        self.writer.add_scalar("Valid/_thresh/", thresh_avg, epoch)

        self.logger.info(
            f"Valid : [{epoch+1:3d}/{self.epochs}] " 
            f"Final Acc : {acc_avg:.2f} Final Thresh : {thresh_avg:.2f} "
            f"Time {time.time() - start_time:.2f}")

        return acc_avg

    def _validate_step(self, model, val_loader):
        all_labels = []
        all_embeds1, all_embeds2 = [], []

        with torch.no_grad():
            for idx, ((imgs1, imgs2), labels) in enumerate(val_loader):
                # Move data sample
                batch_size = labels.size(0)
                imgs1 = imgs1.to(self.device)
                imgs2 = imgs2.to(self.device)
                labels = labels.to(self.device)
                # Extract embeddings
                embeds1 = model(imgs1)
                embeds2 = model(imgs2)
                # Accumulates
                all_labels.append(labels.detach().cpu().numpy())
                all_embeds1.append(embeds1.detach().cpu().numpy())
                all_embeds2.append(embeds2.detach().cpu().numpy())

        # Evaluate
        labels = np.concatenate(all_labels)
        embeds1 = np.concatenate(all_embeds1)
        embeds2 = np.concatenate(all_embeds2)
        TP_ratio, FP_ratio, accs, best_thresholds = evaluate(embeds1, embeds2, labels)
        # Save Checkpoint
        acc_avg = accs.mean()
        thresh_avg = best_thresholds.mean()

        return acc_avg, thresh_avg
