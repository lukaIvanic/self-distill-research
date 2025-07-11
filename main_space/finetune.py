import torch
from torch.profiler import record_function

# Uvozimo sve potrebne managere i pomoćne funkcije iz vašeg postojećeg koda
from main_space.train_management import get_global_train_manager
import main_space.utils.settings_utils as settings_utils
from main_space.utils.log_utils import print_model_params
from main_space.utils.checker_utils import validate_config

# Dohvaćanje globalnih objekata za upravljanje
trainManage = get_global_train_manager()
trainingConfig = settings_utils.get_training_config()
checkpointConfig = settings_utils.get_checkpoint_config()
hyperParamConfig = settings_utils.get_hyperparameter_config()


def setup_finetune_config():
    """
    Pomoćna funkcija za postavljanje specifičnih parametara za fino podešavanje.
    Ovo čini namjeru koda jasnijom.
    """
    print("--- Konfiguriranje postavki za fino podešavanje ---")

    # 1. Aktiviramo logiku za fino podešavanje u cijelom projektu
    trainingConfig.is_finetuning = True

    # 2. Postavljamo nižu stopu učenja i manji broj koraka, što je tipično za fino podešavanje
    trainingConfig.peak_lr = 1e-5  # Značajno niža stopa učenja
    trainingConfig.warmup_steps = 100
    trainingConfig.train_steps = 1500  # Manji broj koraka je dovoljan za prilagodbu

    # 3. Specificiramo koji pre-trenirani model želimo učitati kao polaznu točku
    checkpointConfig.attempt_load_checkpoint_if_exists = True
    # VAŽNO: Ovdje unesite točan alias modela koji želite fino podesiti!
    # Primjer: "100M_teacher:run_9ttfmtet_step_49994"
    checkpointConfig.alias_to_load = "100M_teacher:latest"

    # 4. Onemogućavamo strogo učitavanje jer ćemo imati novu veličinu vokabulara
    checkpointConfig.strict_state_dict_loading = False

    # 5. Definiramo ime pod kojim će se spremiti novi, fino podešeni model
    checkpointConfig.artifact_base_name = "100M_finetuned_slim-orca"

    print(f"Model za fino podešavanje: {checkpointConfig.alias_to_load}")
    print(f"Nova stopa učenja: {trainingConfig.peak_lr}, Broj koraka: {trainingConfig.train_steps}")
    print("--------------------------------------------------")


def main():
    """Glavna funkcija za pokretanje procesa finog podešavanja."""

    # Inicijalizacija i validacija
    torch.manual_seed(trainingConfig.seed)
    validate_config()

    # Postavljanje specifičnih konfiguracija za fino podešavanje
    setup_finetune_config()

    # Standardno postavljanje uređaja i W&B praćenja
    trainManage.setup_device()
    trainManage.setup_wandb_run()

    # Učitavanje podataka za fino podešavanje
    # get_dataloader će sada interno pozvati logiku za SlimOrca i vratiti ažurirani tokenizer
    print("Učitavanje podataka za fino podešavanje (SlimOrca)...")
    train_dataloader, tokenizer = trainManage.setup_train_dataloader()
    trainManage.train_dataloader = train_dataloader
    print(f"Podaci učitani. Veličina novog vokabulara: {len(tokenizer)}")

    # Postavljanje modela
    # Ključan korak: ažuriramo veličinu vokabulara u konfiguraciji PRIJE inicijalizacije modela
    hyperParamConfig.vocab_size = len(tokenizer)
    trainManage.setup_model()

    # Učitavanje težina pre-treniranog modela
    print(f"Učitavanje pre-treniranog modela: {checkpointConfig.alias_to_load}...")
    trainManage.attempt_load_checkpoint_if_exists()

    # Ključan korak: Prilagodba embedding sloja modela novoj veličini vokabulara
    # Ovo je nužno jer smo dodali specijalne tokene ('<|im_start|>', '<|im_end|>')
    print("Prilagodba embedding sloja modela novoj veličini vokabulara...")
    trainManage.model.resize_token_embeddings(len(tokenizer))

    print_model_params(trainManage.model)

    # Postavljanje optimizatora, kriterija i iteratora
    trainManage.setup_optimizer()  # Koristit će nižu stopu učenja iz finetune postavki
    trainManage.setup_criterion()
    trainManage.init_train_iterator()
    trainManage.init_curr_step_counter()  # Počinjemo od nule za fino podešavanje

    # Postavljanje W&B watch nakon što je sve postavljeno
    trainManage.setup_wandb_watch()

    print(f"\nZapočinje fino podešavanje na {trainingConfig.train_steps} koraka...")

    # Glavna petlja za fino podešavanje
    for step_num in range(trainingConfig.train_steps):
        trainManage.next_batch()
        trainManage.make_train_step()  # Ova funkcija radi bez izmjena!
        print(f"{step_num} / {trainingConfig.train_steps}")

    # Spremanje konačnog, fino podešenog modela kao novi W&B artefakt
    print("Spremanje konačnog, fino podešenog modela...")
    trainManage.save_checkpoint(force_save=True)

    print("\nFino podešavanje je završeno.")
    if trainManage.wandb_run:
        print(f"Rezultati su spremljeni u W&B. Link: {trainManage.wandb_run.url}")
        trainManage.wandb_run.finish()


if __name__ == "__main__":
    main()
