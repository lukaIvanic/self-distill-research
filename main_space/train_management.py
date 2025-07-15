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
from main_space.utils.train_utils import make_train_step
from main_space.utils.validation_utils import do_validation_set
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
        self.teacher_model = None
        self.optimizer = None
        self.criterion = None

        self.train_dataloader = None
        self.train_iter = None

        self.validation_dataloader = None

        self.current_train_step = None  # TODO: this step represents total steps, as opposed to the training step in the current epoch
        self.current_train_epoch = None
        self.input_ids = None
        self.target_ids = None

        self.wandb_run = None
        self.profiler_or_null_context = None

    def setup_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_wandb_run(self):
        self.wandb_run = initialize_wandb()

    def setup_train_dataloader(self):
        # self.train_dataloader = get_dataloader(split='train')
        print()
        print("*"*180)
        print(f"###### USING VALIDATION DATALOADER, CHANGE BACK AFTER DEVELOPMENT ######")
        print("*"*180)
        self.train_dataloader = get_dataloader(split='validation')

    def setup_validation_dataloader(self):
        self.validation_dataloader = get_dataloader(split='validation')

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
            p_dropout=hyperParamConfig.dropout_rate,
            attach_aux_heads=hyperParamConfig.attach_aux_heads
            # needs_adapters=True,
            # teacher_d_model=512
        )

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)


        self.model.to(self.device)
        self.model.eval()

    def setup_wandb_watch(self):

        if (not self.projectConfig.wandbConfig.is_wandb_enabled) or (
                self.projectConfig.wandbConfig.wandb_watch_level == "none"):
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
            raise BrokenPipeError(
                "trainManage.get_profiler_context was called, but self.device wasn't initialized yet. "
                "Please call trainManage.setup_device first.")
        if self.wandb_run is None:
            raise BrokenPipeError(
                "trainManage.get_profiler_context was called, but self.wandb_run wasn't initialized yet. "
                "Please call trainManage.setup_wandb_run first.")


        self.profiler_or_null_context = get_profiler_context(wandb_run=self.wandb_run, device=self.device)
        print(f"Properly returning profiler: {self.profiler_or_null_context}")
        return self.profiler_or_null_context


    def setup_optimizer(self):
        if self.model is None:
            raise BrokenPipeError("trainManage.setup_optimizer was called, but self.model wasn't initialized yet. "
                                  "Please call trainManage.setup_model first.")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.projectConfig.trainingConfig.peak_lr, weight_decay=0.1)  # TODO: make weight_decay a hyperparam

    def setup_criterion(self):
        # For LM, CrossEntropyLoss ignores index -100 by default, which our DataLoader uses for label padding.
        # TODO: check out the pad token something about it being index -100 by default or something, which the dataloader uses also by default
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def init_train_iterator(self):

        if self.train_dataloader is None:
            raise BrokenPipeError(
                "trainManage.init_train_iterator was called, but self.train_dataloader wasn't initialized yet. "
                "Please call trainManage.setup_train_dataloader first.")

        self.train_iter = iter(self.train_dataloader)

    def init_curr_step_counter(self):
        if self.current_train_step is None:
            self.current_train_step = 0

        if self.current_train_epoch is None:
            self.current_train_epoch = 0

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

        self.input_ids, self.target_ids, self.train_iter, new_epoch_happened = get_next_batch(
            dataset_iter=self.train_iter,
            dataloader=self.train_dataloader,
            step_num=self.current_train_step,
            device=self.device)

        if new_epoch_happened:
            self.current_train_epoch += 1

    def make_train_step(self):

        if self.current_train_step is None:
            raise BrokenPipeError(
                "trainManage.make_train_step was called, but self.current_train_step wasn't initialized yet. "
                "Please call trainManage.init_curr_step_counter first.")

        if self.model is None:
            raise BrokenPipeError(
                "trainManage.make_train_step was called, but self.model wasn't initialized yet. "
                "Please call trainManage.setup_model first.")


        if self.projectConfig.trainingConfig.distill_enabled and self.teacher_model is None:
            raise BrokenPipeError(
                "trainManage.make_train_step was called and self.projectConfig.trainingConfig.distill_enabled is True, but self.teacher_model wasn't initialized yet. "
                "Please call trainManage.setup_teahcer_model first.")

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

        if self.profiler_or_null_context:
            profiler: torch.profiler.profile = self.profiler_or_null_context
            profiler.step()


        curr_step = self.current_train_step
        log_freq = self.projectConfig.wandbConfig.wandb_log_freq_metrics

        avg_val_loss = None
        if (curr_step == 0 or ((curr_step+1) % log_freq == 0)) and self.projectConfig.wandbConfig.log_validation:
            print(f"Doing validation set for step_num: {curr_step}")
            avg_val_loss = do_validation_set(model=self.model,
                              criterion=self.criterion,
                              dataloader=self.validation_dataloader,
                              device=self.device)

        make_train_step(
            step_num=self.current_train_step,
            model=self.model,
            teacher_model=self.teacher_model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=self.device,
            batch_input_ids=self.input_ids,
            batch_target_ids=self.target_ids,
            wandb_run_obj=self.wandb_run,
            avg_val_loss=avg_val_loss
        )

        self.current_train_step += 1

        # Making sure these variables don't sneak their
        # way into training steps they weren't meant for.
        self.input_ids = None
        self.target_ids = None

        self.save_checkpoint()

    def save_checkpoint(self):

        trainingConfig = self.projectConfig.trainingConfig

        if self.current_train_step is None:
            raise BrokenPipeError(
                "trainManage.save_checkpoint was called, but self.current_train_step wasn't initialized yet. "
                "Please call trainManage.init_curr_step_counter first.")

        if (self.current_train_step + 1) % self.projectConfig.checkpointConfig.checkpoint_frequency != 0:
            return

        if self.model is None:
            raise BrokenPipeError("trainManage.save_checkpoint was called, but self.model wasn't initialized yet. "
                                  "Please call trainManage.setup_model first.")

        if self.optimizer is None:
            raise BrokenPipeError(
                "trainManage.save_checkpoint was called, but self.optimizer wasn't initialized yet. "
                "Please call trainManage.setup_optimizer first.")

        if self.device is None:
            raise BrokenPipeError("trainManage.save_checkpoint was called, but self.device wasn't initialized yet. "
                                  "Please call trainManage.setup_device first.")

        if self.wandb_run is None:
            raise BrokenPipeError(
                "trainManage.save_checkpoint was called, but self.wandb_run wasn't initialized yet. "
                "Please call trainManage.setup_wandb_run first.")

        checkpoint_state = {
            'resume_training_step': self.current_train_step,
            'resume_training_epoch': self.current_train_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'torch_rng_state': torch.get_rng_state(),
            # TODO: implement scaler saving and loading
        }

        if torch.cuda.is_available() and self.device.type == 'cuda':
            checkpoint_state['cuda_rng_state'] = torch.cuda.get_rng_state()

        local_checkpoint_filename = f"checkpoint_{self.current_train_step}.pt"
        local_checkpoint_path = os.path.join(self.wandb_run.dir, local_checkpoint_filename)

        try:
            torch.save(checkpoint_state, local_checkpoint_path)
            print(f"Checkpoint saved locally to {local_checkpoint_path}")
        except Exception as e:
            raise OSError(f"Error saving checkpoint to W&B: {e}")

        artifact_name = self.projectConfig.checkpointConfig.artifact_base_name  # TODO: currently only test implementation
        artifact_type_custom = "model-checkpoint"

        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type_custom,
            description=f"Checkpoint at step {self.current_train_step} for run {self.wandb_run.name}",
            metadata={
                'step': self.current_train_step,
                'project': self.projectConfig.wandbConfig.wandb_project_name,
                'run_id': self.wandb_run.id,
            }
        )

        artifact.add_file(local_checkpoint_path, name="checkpoint.pt")  # name here is how it appears IN the artifact


        latest_alias = "latest"
        id_step_alias = f"run_{self.wandb_run.id}_step_{self.current_train_step}"
        lr_alias = f"lr {trainingConfig.peak_lr:.1e}"
        ctx_alias = f"ctx 2**{math.log2(trainingConfig.hyperParamConfig.ctx_len):.1f}"
        batch_tokens_alias = f"ba_toks {trainingConfig.batch_size*trainingConfig.hyperParamConfig.ctx_len}"

        aliases_to_log = [latest_alias,
                          id_step_alias,
                          lr_alias,
                          ctx_alias,
                          batch_tokens_alias]  # TODO: investigate what these aliases actually mean
        self.wandb_run.log_artifact(artifact, aliases=aliases_to_log)
        print(f"Checkpoint artifact '{artifact_name}' logged to W&B with aliases: {aliases_to_log}")

        print(f"After saving curr step: {self.current_train_step}")


    def setup_teacher_model(self):
        if not self.projectConfig.trainingConfig.distill_enabled:
            return

        projectName = "self-distill-research"
        entity = "luka_newbie"
        alias_tag = "30M_classic:run_e39zl7c0_step_11999"
        artifactName = alias_tag.split(':')[0]
        alias = alias_tag.split(':')[1]

        api = wandb.Api()

        artifact_path = f"{entity}/{projectName}/{artifactName}:{alias}"
        print(f"[INFO] Loading artifact '{artifact_path}'")
        artifact = api.artifact(artifact_path, type="model-checkpoint")
        download_dir = artifact.download()
        print(f"[INFO] Artifact downloaded to: {download_dir}")

        # 1) Grab the run that produced this artifact:
        creator = artifact.logged_by()  # a <wandb.apis.public.Run> stub
        run_ref = f"{creator.entity}/{creator.project}/{creator.id}"
        print(f"[INFO] Loading run that produced artifact: {run_ref}")
        run = api.run(run_ref)

        # 2) Extract and print the run.config
        config = run.config or {}
        print("[INFO] Loaded run.config:")
        for k, v in sorted(config.items()):
            print(f"  - {k}: {v!r}")

        # Read metadata for model hyperparameters
        metadata = artifact.metadata or {}
        print("[INFO] Loaded artifact metadata:")
        for k, v in metadata.items():
            print(f"  - {k}: {v}")

        # Extract required hyperparameters (with defaults or errors)
        try:
            vocab_size = int(config["vocab_size"])
            d_model = int(config["d_model"])
            n_heads = int(config["num_heads"])
            n_layers = int(config["num_layers"])
            ctx_size = int(config["ctx_len"])
            p_dropout = float(config["dropout_rate"])
        except KeyError as e:
            print(f"[ERROR] Missing hyperparameter in metadata: {e}")
            return

        # Instantiate model
        print("[INFO] Instantiating teacher model with loaded hyperparameters...")
        self.teacher_model = MyTransformerLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ctx_size=ctx_size,
            p_dropout=p_dropout,
        )
        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        print("[INFO] Model architecture:")
        print(self.teacher_model)

        # Load checkpoint
        checkpoint_file = os.path.join(download_dir, "checkpoint.pt")
        if not os.path.exists(checkpoint_file):
            print(f"[ERROR] checkpoint.pt not found in {download_dir}")
            return

        print(f"[INFO] Loading state dict from {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        result = self.teacher_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(f"[INFO] Missing keys: {result.missing_keys}")
        print(f"[INFO] Unexpected keys: {result.unexpected_keys}")
        print("[INFO] Model weights loaded successfully.")



    def attempt_load_checkpoint_if_exists(self):

        checkpointConfig = self.projectConfig.checkpointConfig

        if not checkpointConfig.attempt_load_checkpoint_if_exists:
            return

        if self.wandb_run is None:
            raise BrokenPipeError(
                "trainManage.load<<_checkpoint was called, but self.wandb_run wasn't initialized yet. "
                "Please call trainManage.setup_wandb_run first.")

        if self.device is None:
            raise BrokenPipeError("trainManage.load_checkpoint was called, but self.device wasn't initialized yet. "
                                  "Please call trainManage.setup_device first.")

        if self.model is None:
            raise BrokenPipeError("trainManage.load_checkpoint was called, but self.model wasn't initialized yet. "
                                  "Please call trainManage.setup_model first.")

        if self.optimizer is None:
            raise BrokenPipeError(
                "trainManage.load_checkpoint was called, but self.optimizer wasn't initialized yet. "
                "Please call trainManage.setup_optimizer first.")

        if self.train_dataloader is None:
            raise BrokenPipeError(
                "trainManage.load_checkpoint was called, but self.train_dataloader wasn't initialized yet. "
                "Please call trainManage.setup_train_dataloader first.")

        if self.train_iter is None:
            raise BrokenPipeError(
                "trainManage.load_checkpoint was called, but self.train_iter wasn't initialized yet. "
                "Please call trainManage.init_train_iterator first.")

        artifact_name = self.projectConfig.checkpointConfig.artifact_base_name  # TODO: currently only test implementation
        artifact_alias_to_load = "run_dl0n7uc3_step_9999"  # TODO: fix, this should come from checkpointConfig

        artifact_full_path = f"{self.wandb_run.entity}/{self.wandb_run.project}/{artifact_name}:{artifact_alias_to_load}"

        try:
            artifact = self.wandb_run.use_artifact(artifact_full_path, type="model-checkpoint")
        except wandb.errors.CommError as e:
            print(f"wandb CommError, likely this is the first training run: {e}")
            return

        artifact_dir = artifact.download()
        print(f"Artifact  downloaded to: {artifact_dir}")

        checkpoint_file_name_in_artifact = "checkpoint.pt"
        checkpoint_path = os.path.join(artifact_dir, checkpoint_file_name_in_artifact)

        if not os.path.exists(checkpoint_path):
            print(
                f"Error: Checkpoint file '{checkpoint_file_name_in_artifact}' not found in downloaded artifact at {artifact_dir}")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.current_train_step = checkpoint['resume_training_step']

        is_strict = checkpointConfig.strict_state_dict_loading
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=is_strict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # TODO: add same for optimizer and dataset, but be careful and add PipeLine errors
        #       at the start of the function.
        #       also need to add all the step settings, and any changeable configuration as well
        #       and torch rng state

        print("Re-initializing training data iterator with restored RNG state...")

        print("Data iterator re-initialized.")

        # TODO: I guess this works because the rng state sets it correctly?
        #       But if the current rng state is for the iteration that's at
        #       a certain point with the dataloader, then we will be
        #       advancing the torch rng until we hit the previous iteration.
        num_batches_to_skip_in_current_epoch = self.current_train_step

        self.setup_train_dataloader()   # These are redundant, as they were already called in
        self.init_train_iterator()      # the train.py script, but I'm re-initializing
                                        # just to be a little more future-safe.


        print(f"Advancing data iterator by {num_batches_to_skip_in_current_epoch} batches to match step {self.current_train_step}.")
        for i in range(num_batches_to_skip_in_current_epoch):
            try:
                next(self.train_iter)
            except StopIteration:
                self.current_train_epoch += 1
                print(f"Did {self.current_train_epoch} epochs so far!")

                print(f"Epoch finished at step {i}. Resetting train_loader for continued iteration.")

                self.train_iter = iter(self.train_dataloader)
                next(self.train_iter)
                # raise BrokenPipeError(
                #     f"in TrainManage.attempt_load_checkpoint_if_exists, "
                #     f"StopIteration was hit while advancing dataloader by {i + 1} "
                #     f"batches (target: {num_batches_to_skip_in_current_epoch})."
                #     f"This might indicate an issue with the loaded step number or dataloader length.")

        torch.set_rng_state(checkpoint['torch_rng_state'])
        if torch.cuda.is_available() and self.device.type == 'cuda':
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])

        print("Torch RNG states restored.")

        print("Data iterator position updated.")

        print(f"After loading curr step: {self.current_train_step}")

