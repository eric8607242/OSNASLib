from utils import AverageMeter, accuracy, save

from ..base_agent import MetaAgent

class {{customize_class}}MetaAgent(MetaAgent):
    """{{customize_class}} meta agent
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

