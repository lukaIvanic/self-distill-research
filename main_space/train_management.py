
import torch
import torch.nn as nn
import torch.optim as optim
import contextlib

from main_space.settings import Config
from main_space.model import MyTransformerLM
from main_space.utils.train_utils import make_train_step
from main_space.utils.dataloader_utils import get_dataloader, get_next_batch
from main_space.utils.log_utils import setup_wandb_watch, initialize_wandb, get_profiler_context


_train_manager = None


def get_global_train_manager():
    global _train_manager
    if _train_manager is None:
        _train_manager = TrainManager()

    return _train_manager


class TrainManager:

    def __init__(self):

        self.projectConfig = Config()

        self.device = None

        self.model = None
        self.optimizer = None
        self.criterion = None

        self.train_dataloader = None
        self.train_iter = None

        self.current_train_step = None
        self.input_ids = None
        self.target_ids = None

        self.wandb_run = None
        self.profiler_or_null_context = None

    def setup_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_wandb_run(self):
        self.wandb_run = initialize_wandb()

    def setup_train_dataloader(self):
        self.train_dataloader = get_dataloader(split='train')

    def setup_model(self):

        if self.device is None:
            raise BrokenPipeError("trainManage.setup_model was called, but self.device wasn't initialized yet. "
                                  "Please call trainManage.setup_device first.")

        hyperParamConfig = self.projectConfig.trainingConfig.hyperParamConfig

        # left explicit passing of params for clarity
        self.model = MyTransformerLM(
            vocab_size=hyperParamConfig.vocab_size,
            d_model=hyperParamConfig.d_model,
            n_heads=hyperParamConfig.num_heads,
            n_layers=hyperParamConfig.num_layers,
            ctx_size=hyperParamConfig.ctx_len,
            p_dropout=hyperParamConfig.dropout_rate
        ).to(self.device)

    def setup_wandb_watch(self):

        if (not self.projectConfig.wandbConfig.is_wandb_enabled) or (self.projectConfig.wandbConfig.wandb_watch_level == "none"):
            return


        if self.model is None:
            raise BrokenPipeError("trainManage.set_wandb_watch was called, but self.model wasn't initialized yet. "
                                  "Please call trainManage.setup_model first.")

        if self.wandb_run is None:
            raise BrokenPipeError(
                "trainManage.set_wandb_watch was called, but self.wandb_run wasn't initialized yet. "
                "Please call trainManage.setup_wandb_run first.")

        setup_wandb_watch(model=self.model,
                          wandb_run=self.wandb_run)

    def get_profiler_context(self):
        wandbConfig = self.projectConfig.wandbConfig
        if not (wandbConfig.is_wandb_enabled and wandbConfig.is_profiler_enabled):
            return contextlib.nullcontext()

        if self.device is None:
            raise BrokenPipeError("trainManage.get_profiler_context was called, but self.device wasn't initialized yet. "
                                  "Please call trainManage.setup_device first.")
        if self.wandb_run is None:
            raise BrokenPipeError(
                "trainManage.get_profiler_context was called, but self.wandb_run wasn't initialized yet. "
                "Please call trainManage.setup_wandb_run first.")

        return get_profiler_context(wandb_run=self.wandb_run, device=self.device)

    def setup_optimizer(self):
        if self.model is None:
            raise BrokenPipeError("trainManage.setup_optimizer was called, but self.model wasn't initialized yet. "
                                  "Please call trainManage.setup_model first.")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.projectConfig.trainingConfig.peak_lr)

    def setup_criterion(self):
        # For LM, CrossEntropyLoss ignores index -100 by default, which our DataLoader uses for label padding.
        # TODO: check out the pad token something about it being index -100 by default or something, which the dataloader uses also by default
        self.criterion = nn.CrossEntropyLoss()

    def init_train_iterator(self):

        if self.train_dataloader is None:
            raise BrokenPipeError(
                "trainManage.init_train_iterator was called, but self.train_dataloader wasn't initialized yet. "
                "Please call trainManage.setup_train_dataloader first.")

        self.train_iter = iter(self.train_dataloader)

    def init_curr_step_counter(self):
        self.current_train_step = 0

    def next_batch(self):

        if self.train_iter is None:
            raise BrokenPipeError(
                "trainManage.next_batch was called, but self.train_iter wasn't initialized yet. "
                "Please call trainManage.init_train_iterator first.")

        if self.train_dataloader is None:
            raise BrokenPipeError(
                "trainManage.next_batch was called, but self.train_dataloader wasn't initialized yet. "
                "Please call trainManage.setup_train_dataloader first.")


        if self.current_train_step is None:
            raise BrokenPipeError(
                "trainManage.next_batch was called, but self.current_train_step wasn't initialized yet. "
                "Please call trainManage.init_curr_step_counter first.")

        if self.device is None:
            raise BrokenPipeError("trainManage.next_batch was called, but self.device wasn't initialized yet. "
                                  "Please call trainManage.setup_device first.")

        self.input_ids, self.target_ids, self.train_iter = get_next_batch(train_iter=self.train_iter,
                                                           train_dataloader=self.train_dataloader,
                                                           step_num=self.current_train_step,
                                                           device=self.device)


    def make_train_step(self):

        if self.current_train_step is None:
            raise BrokenPipeError(
                "trainManage.make_train_step was called, but self.current_train_step wasn't initialized yet. "
                "Please call trainManage.init_curr_step_counter first.")

        if self.model is None:
            raise BrokenPipeError(
                "trainManage.make_train_step was called, but self.model wasn't initialized yet. "
                "Please call trainManage.setup_model first.")

        if self.criterion is None:
            raise BrokenPipeError(
                "trainManage.make_train_step was called, but self.criterion wasn't initialized yet. "
                "Please call trainManage.setup_criterion first.")


        if self.optimizer is None:
            raise BrokenPipeError(
                "trainManage.make_train_step was called, but self.optimizer wasn't initialized yet. "
                "Please call trainManage.setup_optimizer first.")

        if self.device is None:
            raise BrokenPipeError("trainManage.make_train_step was called, but self.device wasn't initialized yet. "
                                  "Please call trainManage.setup_device first.")

        if self.input_ids is None:
            raise BrokenPipeError("trainManage.make_train_step was called, but self.input_ids is None. "
                                  "\nMake sure you called TrainManage.next_batch() before the call to TrainManage.make_train_step."
                                  "\nYou can't call TrainManage.make_train_step multiple times in a row, "
                                  "\nyou need to call TrainManage.next_batch() every time before TrainManage.make_train_step().")
        if self.target_ids is None:
            raise BrokenPipeError("trainManage.make_train_step was called, but self.target_ids is None."
                                  f"\nIf self.input_ids {self.input_ids} is not None, then something has seriously gone wrong with the iterative train pipeline."
                                  f"\nPlease check TrainManage.next_batch() and TrainManage.make_train_step() thoroughly.")

        if self.wandb_run is None and self.projectConfig.wandbConfig.is_wandb_enabled:
            raise BrokenPipeError(
                "trainManage.make_train_step was called and wandbConfig.is_wandb_enabled == True. But self.wandb_run is None. "
                "Please try calling trainManage.setup_wandb_run first. However, this hints at a *major* pipeline issue. "
                "Please pay attention to TrainManage.setup_wandb_run() and maybe TrainManage.setup_wandb_watch().")

        if self.current_train_step is None:
            raise BrokenPipeError(
                "trainManage.make_train_step was called, but self.current_train_step wasn't initialized yet. "
                "Please call trainManage.init_curr_step_counter first.")

        make_train_step(
            step_num=self.current_train_step,
            model=self.model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=self.device,
            batch_input_ids=self.input_ids,
            batch_target_ids=self.target_ids,
            wandb_run_obj=self.wandb_run,
            profiler_obj=self.profiler_or_null_context
        )


        self.current_train_step += 1

        # Making sure these variables don't sneak their
        # way into training steps they weren't meant for.
        self.input_ids = None
        self.target_ids = None
