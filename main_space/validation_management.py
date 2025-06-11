import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import contextlib

import wandb

from main_space.settings import Config
from main_space.model import MyTransformerLM
from main_space.utils.validation_utils import make_val_step
from main_space.utils.dataloader_utils import get_dataloader, get_next_batch
from main_space.utils.log_utils import setup_wandb_watch, initialize_wandb, get_profiler_context, log_validation_step

_validation_manager = None


def get_global_validation_manager() -> 'ValidationManager':
    global _validation_manager
    if _validation_manager is None:
        _validation_manager = ValidationManager()

    return _validation_manager


class ValidationManager:

    def __init__(self):

        self.projectConfig = Config()

        self.device = None

        self.model = None
        self.criterion = None

        self.validation_dataloader = None
        self.validation_iter = None

        self.current_validation_step = None  # TODO: this step represents total steps, as opposed to the training step in the current epoch

        self.total_ce_loss = None
        self.total_ce_losses_bucketed = None

        self.input_ids = None
        self.target_ids = None

        self.wandb_run = None
        self.profiler_or_null_context = None

    def setup_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_wandb_run(self):
        self.wandb_run = initialize_wandb()

    def setup_validation_dataloader(self):
        self.validation_dataloader = get_dataloader(split='validation')

    def setup_model(self):

        if self.device is None:
            raise BrokenPipeError("validationManage.setup_model was called, but self.device wasn't initialized yet. "
                                  "Please call validationManage.setup_device first.")

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

        if (not self.projectConfig.wandbConfig.is_wandb_enabled) or (
                self.projectConfig.wandbConfig.wandb_watch_level == "none"):
            return

        if self.model is None:
            raise BrokenPipeError("validationManage.set_wandb_watch was called, but self.model wasn't initialized yet. "
                                  "Please call validationManage.setup_model first.")

        if self.wandb_run is None:
            raise BrokenPipeError(
                "validationManage.set_wandb_watch was called, but self.wandb_run wasn't initialized yet. "
                "Please call validationManage.setup_wandb_run first.")

        setup_wandb_watch(model=self.model,
                          wandb_run=self.wandb_run)

    def get_profiler_context(self):
        wandbConfig = self.projectConfig.wandbConfig
        if not (wandbConfig.is_wandb_enabled and wandbConfig.is_profiler_enabled):
            return contextlib.nullcontext()

        if self.device is None:
            raise BrokenPipeError(
                "validationManage.get_profiler_context was called, but self.device wasn't initialized yet. "
                "Please call validationManage.setup_device first.")
        if self.wandb_run is None:
            raise BrokenPipeError(
                "validationManage.get_profiler_context was called, but self.wandb_run wasn't initialized yet. "
                "Please call validationManage.setup_wandb_run first.")

        return get_profiler_context(wandb_run=self.wandb_run, device=self.device)

    def setup_criterion(self):
        # For LM, CrossEntropyLoss ignores index -100 by default, which our DataLoader uses for label padding.
        # TODO: check out the pad token something about it being index -100 by default or something, which the dataloader uses also by default
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def init_validation_iterator(self):

        if self.validation_dataloader is None:
            raise BrokenPipeError(
                "validationManage.init_validation_iterator was called, but self.validation_dataloader wasn't initialized yet. "
                "Please call validationManage.setup_validation_dataloader first.")

        self.validation_iter = iter(self.validation_dataloader)

    def init_curr_step_counter(self):
        if self.current_validation_step is None:
            self.current_validation_step = 0

    def setup_for_loss_calculation(self):
        self.total_ce_loss = 0.0
        self.total_ce_losses_bucketed = [0.0 for _ in
                                         range(self.projectConfig.trainingConfig.hyperParamConfig.ctx_len // 64)]

    def next_batch(self):

        if self.validation_iter is None:
            raise BrokenPipeError(
                "validationManage.next_batch was called, but self.validation_iter wasn't initialized yet. "
                "Please call validationManage.init_validation_iterator first.")

        if self.validation_dataloader is None:
            raise BrokenPipeError(
                "validationManage.next_batch was called, but self.validation_dataloader wasn't initialized yet. "
                "Please call validationManage.setup_validation_dataloader first.")

        if self.current_validation_step is None:
            raise BrokenPipeError(
                "validationManage.next_batch was called, but self.current_validation_step wasn't initialized yet. "
                "Please call validationManage.init_curr_step_counter first.")

        if self.device is None:
            raise BrokenPipeError("validationManage.next_batch was called, but self.device wasn't initialized yet. "
                                  "Please call validationManage.setup_device first.")

        self.input_ids, self.target_ids, self.validation_iter, new_epoch = get_next_batch(
            dataset_iter=self.validation_iter,
            dataloader=self.validation_dataloader,
            step_num=self.current_validation_step,
            device=self.device)

        return not new_epoch

    def make_validation_step(self):

        if self.current_validation_step is None:
            raise BrokenPipeError(
                "validationManage.make_validation_step was called, but self.current_validation_step wasn't initialized yet. "
                "Please call validationManage.init_curr_step_counter first.")

        if self.model is None:
            raise BrokenPipeError(
                "validationManage.make_validation_step was called, but self.model wasn't initialized yet. "
                "Please call validationManage.setup_model first.")

        if self.criterion is None:
            raise BrokenPipeError(
                "validationManage.make_validation_step was called, but self.criterion wasn't initialized yet. "
                "Please call validationManage.setup_criterion first.")

        if self.device is None:
            raise BrokenPipeError(
                "validationManage.make_validation_step was called, but self.device wasn't initialized yet. "
                "Please call validationManage.setup_device first.")

        if self.input_ids is None:
            raise BrokenPipeError("validationManage.make_validation_step was called, but self.input_ids is None. "
                                  "\nMake sure you called validationManage.next_batch() before the call to validationManage.make_validation_step."
                                  "\nYou can't call validationManage.make_validation_step multiple times in a row, "
                                  "\nyou need to call validationManage.next_batch() every time before validationManage.make_validation_step().")
        if self.target_ids is None:
            raise BrokenPipeError("validationManage.make_validation_step was called, but self.target_ids is None."
                                  f"\nIf self.input_ids {self.input_ids} is not None, then something has seriously gone wrong with the iterative validation pipeline."
                                  f"\nPlease check validationManage.next_batch() and validationManage.make_validation_step() thoroughly.")

        if self.wandb_run is None and self.projectConfig.wandbConfig.is_wandb_enabled:
            raise BrokenPipeError(
                "validationManage.make_validation_step was called and wandbConfig.is_wandb_enabled == True. But self.wandb_run is None. "
                "Please try calling validationManage.setup_wandb_run first. However, this hints at a *major* pipeline issue. "
                "Please pay attention to validationManage.setup_wandb_run() and maybe validationManage.setup_wandb_watch().")

        if self.current_validation_step is None:
            raise BrokenPipeError(
                "validationManage.make_validation_step was called, but self.current_validation_step wasn't initialized yet. "
                "Please call validationManage.init_curr_step_counter first.")

        loss, bucket_losses = make_val_step(
            model=self.model,
            criterion=self.criterion,
            device=self.device,
            batch_input_ids=self.input_ids,
            batch_target_ids=self.target_ids,
            # wandb_run_obj=self.wandb_run,
            # profiler_obj=self.profiler_or_null_context
        )

        self.current_validation_step += 1

        self.total_ce_loss += loss.item()

        for i in range(self.projectConfig.trainingConfig.hyperParamConfig.ctx_len // 64):
            self.total_ce_losses_bucketed[i] += bucket_losses[i].item()

        print(f"Loss: {loss}, bucket_losses: {bucket_losses}")

        avg_ce_loss = self.total_ce_loss / self.current_validation_step
        avg_periodic_losses = [x / self.current_validation_step for x in self.total_ce_losses_bucketed]

        log_validation_step(step_num=self.current_validation_step,
                            curr_avg_ce_loss=avg_ce_loss,
                            curr_avg_periodic_losses=avg_periodic_losses,
                            device=self.device,
                            profiler_obj=self.profiler_or_null_context,
                            wandb_run_obj=self.wandb_run
                            )

        # Making sure these variables don't sneak their
        # way into training steps they weren't meant for.
        self.input_ids = None
        self.target_ids = None

    def attempt_load_checkpoint_if_exists(self):

        checkpointConfig = self.projectConfig.checkpointConfig

        if not checkpointConfig.attempt_load_checkpoint_if_exists:
            return

        if self.wandb_run is None:
            raise BrokenPipeError(
                "validationManage.load_checkpoint was called, but self.wandb_run wasn't initialized yet. "
                "Please call validationManage.setup_wandb_run first.")

        if self.device is None:
            raise BrokenPipeError(
                "validationManage.load_checkpoint was called, but self.device wasn't initialized yet. "
                "Please call validationManage.setup_device first.")

        if self.model is None:
            raise BrokenPipeError("validationManage.load_checkpoint was called, but self.model wasn't initialized yet. "
                                  "Please call validationManage.setup_model first.")

        if self.validation_dataloader is None:
            raise BrokenPipeError(
                "validationManage.load_checkpoint was called, but self.validation_dataloader wasn't initialized yet. "
                "Please call validationManage.setup_validation_dataloader first.")

        if self.validation_iter is None:
            raise BrokenPipeError(
                "validationManage.load_checkpoint was called, but self.validation_iter wasn't initialized yet. "
                "Please call validationManage.init_validation_iterator first.")

        artifact_name = self.projectConfig.checkpointConfig.artifact_base_name  # TODO: currently only test implementation
        artifact_alias_to_load = self.projectConfig.checkpointConfig.alias_to_load  # This would come from self.projectConfig...

        artifact_full_path = f"{self.wandb_run.entity}/{self.wandb_run.project}/{artifact_name}:{artifact_alias_to_load}"

        try:
            artifact = self.wandb_run.use_artifact(artifact_full_path, type="model-checkpoint")
        except wandb.errors.CommError as e:
            raise ValueError(f"Couldn't find saved model. {e}")

        artifact_dir = artifact.download()
        print(f"Artifact  downloaded to: {artifact_dir}")

        checkpoint_file_name_in_artifact = "checkpoint.pt"
        checkpoint_path = os.path.join(artifact_dir, checkpoint_file_name_in_artifact)

        if not os.path.exists(checkpoint_path):
            raise BrokenPipeError(
                f"Error: Checkpoint file '{checkpoint_file_name_in_artifact}' not found in downloaded artifact at {artifact_dir}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        is_strict = checkpointConfig.strict_state_dict_loading
        load_result = self.model.load_state_dict(checkpoint['model_state_dict'], strict=is_strict)

        print("Missing keys:", load_result.missing_keys)
        print("Unexpected keys:", load_result.unexpected_keys)
        print("Model successfully loaded from checkpoint.")
