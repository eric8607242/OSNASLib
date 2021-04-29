from .base_agent import MetaAgent

from utils import get_optimizer, get_lr_scheduler, resume_checkpoint
from model import Model, load_architecture

class EvaluateAgent(MetaAgent):
    def __init__(self, args):
        super(EvaluateAgent, self).__init__(args, "evaluate")

        # Construct model and correspond optimizer ======================================
        architecture = load_architecture(self.args.searched_model_path)

        self.model = Model(
            self.macro_cfg,
            self.micro_cfg,
            architecture,
            self.args.classes,
            self.args.dataset)

        self.optimizer = get_optimizer(
            self.model.parameters(),
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

        # Resume checkpoint ===============================================================
        self.start_epoch = 0
        if self.args.resume:
            self.start_epoch = resume_checkpoint(
                self.model,
                self.args.resume,
                self.optimizer,
                self.lr_scheduler)
            logger.info(
                "Resume training from {} at epoch {}".format(
                    self.args.resume, start_epoch))
        # =================================================================================

        if device.type == "cuda" and self.args.ngpu >= 1:
            self.model = self.model.to(self.device)
            self.model = nn.DataParallel(
                self.model, list(range(self.args.ngpu)))

    def evaluate(self):
        start_time = time.time()
        self.train_loop(
            self.model,
            self.train_loader,
            self.test_loader,
            self.optimizer,
            self.lr_scheduler,
            self.start_epoch)
        logger.info(
            "Total search time : {:.2f}".format(
                time.time() - start_time))

    def inference(self):
        start_time = time.time()
        top1_acc = self.validate(self.model, self.test_loader, 0)
        logger.info("Final Top1 accuracy : {}".format(top1_acc))
        logger.info(
            "Total search time : {:.2f}".format(
                time.time() - start_time))
