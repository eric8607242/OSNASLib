from .base_agent import MetaAgent

from utils import get_optimizer, get_lr_scheduler, resume_checkpoint
from model import Supernet, save_architecture

class SearchAgent(MetaAgent):
    def __init__(self, args):
        super(SearchAgent, self).__init__(args, "search")

        # Construct model and correspond optimizer ======================================
        supernet = Supernet(
            self.macro_cfg,
            self.micro_cfg,
            self.args.classes,
            self.args.dataset,
            self.args.search_strategy,
            bn_momentum=self.args.bn_momentum,
            bn_track_running_stats=self.args.bn_track_running_stats)
        self.supernet = supernet.to(self.device)

        self.optimizer = get_optimizer(
            self.supernet.parameters(),
            self.args.optimizer,
            learning_rate=self.args.lr,
            weight_decay=self.args.weight_decay,
            logger=self.logger,
            momentum=self.args.momentum,
            alpha=self.args.alpha,
            beta=self.args.beta)

        self.lr_scheduler = get_lr_scheduler(
            self.optimizer,
            self.args.lr_scheduler,
            self.logger,
            step_per_epoch=len(
                self.train_loader),
            step_size=self.args.decay_step,
            decay_ratio=self.args.decay_ratio,
            total_epochs=self.args.epochs)
        # =================================================================================


        # Construct search utility ========================================================
        self.training_strategy = TrainingStrategy(
            self.args.sample_strategy, len(micro_cfg), len(
                macro_cfg["search"]), self.supernet)

        self.lookup_table = LookUpTable(
            macro_cfg,
            micro_cfg,
            self.args.lookup_table_path,
            self.args.input_size,
            info_metric=[
                "flops",
                "param"])
        # =================================================================================


        # Resume checkpoint ===============================================================
        start_epoch = 0
        if self.args.resume:
            start_epoch = resume_checkpoint(
                self.supernet,
                self.args.resume,
                self.optimizer,
                self.lr_scheduler)
            logger.info(
                "Resume training from {} at epoch {}".format(
                    self.args.resume, start_epoch))
        # =================================================================================

        if self.device.type == "cuda" and self.args.ngpu >= 1:
            self.supernet = nn.DataParallel(
                self.supernet, list(range(self.args.ngpu)))

    def search(self):
        start_time = time.time()
        self.logger.info("Searching process start!")

        if not self.args.directly_search:
            self.train_loop(
                self.supernet,
                self.train_loader,
                self.val_loader,
                self.optimizer,
                self.lr_scheduler,
                start_epoch)

        best_architecture, best_architecture_hc, best_architecture_top1 = self.search_strategy.search(
            self.trainer, self.training_strategy)

        logger.info("Best architectrue : {}".format(best_architecture))
        logger.info(
            "Best architectrue top1 : {:.3f}".format(
                best_architecture_top1 * 100))
        logger.info("Best architectrue hc : {}".format(best_architecture_hc))

        save_architecture(self.args.searched_model_path, best_architecture)
        logger.info(
            "Total search time : {:.2f}".format(
                time.time() - start_time))
