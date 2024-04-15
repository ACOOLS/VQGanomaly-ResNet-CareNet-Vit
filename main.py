import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset, Subset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only
import shutil
import yaml
import requests
import logging
logging.basicConfig(level=logging.INFO)

class TelegramLogger:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}/sendMessage"

    def send_message(self, message):
        data = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }
        response = requests.post(self.base_url, data=data)
        return response.json()

class TelegramProgressLogger(pl.Callback):
    def __init__(self, telegram_logger, config_filenames):
        self.telegram_logger = telegram_logger
        if isinstance(config_filenames, list) and config_filenames:
            config_filename = config_filenames[0]  # Prenez le premier fichier si c'est une liste
        else:
            config_filename = config_filenames  # Sinon, utilisez la variable telle quelle

        # Extraire les segments du nom du fichier
        base_name = os.path.basename(config_filename)  # Extrait le nom du fichier, par exemple 'custom_vqgan_1CH_breast_classique.yaml'
        name_without_ext = os.path.splitext(base_name)[0]  # Retire l'extension, donne 'custom_vqgan_1CH_breast_classique'
        parts = name_without_ext.split('_')  # Divise la chaîne en parties basées sur '_'
        self.dataset_type = parts[3]  # 'breast'
        self.model_type = parts[4]  # 'classique'

    def on_train_start(self, trainer, pl_module):
        start_message = f"Début de l'entraînement pour {self.dataset_type} avec le modèle {self.model_type}..."
        self.telegram_logger.send_message(start_message)

    def on_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get('train/total_loss', 'N/A')
        val_loss = trainer.callback_metrics.get('val/aeloss_epoch', 'N/A')
        if isinstance(train_loss, float) and isinstance(val_loss, float):
            message = f"Époque {trainer.current_epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
        else:
            message = f"Époque {trainer.current_epoch}: Train Loss = {train_loss}, Val Loss = {val_loss}"
        self.telegram_logger.send_message(message)

    def on_train_end(self, trainer, pl_module):
        end_message = f"Entraînement terminé pour {self.dataset_type} avec le modèle {self.model_type}."
        self.telegram_logger.send_message(end_message)

        
torch.set_default_dtype(torch.float32)

def find_latest_logdir(logs_directory="logs"):
    # Trouvez le dossier le plus récent dans le répertoire des logs
    # Exclude directories named '.ipynb_checkpoints'
    list_of_dirs = [
        os.path.join(logs_directory, d) for d in os.listdir(logs_directory)
        if os.path.isdir(os.path.join(logs_directory, d)) and d != ".ipynb_checkpoints"
    ]
    if not list_of_dirs:
        return None
    latest_dir = max(list_of_dirs, key=os.path.getmtime)
    return latest_dir


def read_status_file(logdir):
    status_file = os.path.join(logdir, "status.txt")
    if os.path.exists(status_file):
        with open(status_file, "r") as file:
            status = file.read().strip()
        return status
    return None

def write_status_file(logdir, status):
    status_file = os.path.join(logdir, "status.txt")
    os.makedirs(os.path.dirname(status_file), exist_ok=True)
    with open(status_file, "w") as file:
        file.write(status)

def get_obj_from_str(string, reload=False):
    # package_name = None
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    # if module[0] == '.':
    #     file = str(Path(__file__).parent)
    #     package_name = file
    return getattr(importlib.import_module(module, package=None), cls)

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)

    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument('--paperspace', action='store_true', help="Use the latest checkpoint if available")
    
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    

    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )

    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )

    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )

    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )

    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))

def options():
    opt, _ = Trainer.add_argparse_args(get_parser()).parse_known_args()
    return opt

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None, subset_size=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        self.subset_size = subset_size  # Ajout de la taille du sous-ensemble
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader

        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader

        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader

        self.wrap = wrap
    
    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs
        )
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])


    def get_subset_loader(self, dataset, shuffle=True, **loader_args):
        if self.subset_size is None:
            return DataLoader(dataset, shuffle=shuffle, **loader_args)

        # Sélectionner aléatoirement un sous-ensemble du dataset
        subset_indices = torch.randperm(len(dataset))[:self.subset_size].tolist()
        subset = Subset(dataset, subset_indices)

        # DataLoader pour le sous-ensemble
        return DataLoader(subset, shuffle=shuffle, **loader_args)

    # Redéfinition des méthodes DataLoader pour utiliser `get_subset_loader`
    def _train_dataloader(self):
        return self.get_subset_loader(self.datasets["train"], batch_size=self.batch_size,
                                      num_workers=self.num_workers, shuffle=True, pin_memory = True)

    def _val_dataloader(self):
        return self.get_subset_loader(self.datasets["validation"], batch_size=self.batch_size,
                                      num_workers=self.num_workers, shuffle=False, pin_memory = True)

    def _test_dataloader(self):
        return self.get_subset_loader(self.datasets["test"], batch_size=self.batch_size,
                                      num_workers=self.num_workers, shuffle=False, pin_memory = True)

# class DataModuleFromConfig(pl.LightningDataModule):
#     def __init__(self, batch_size, train=None, validation=None, test=None,
#                  wrap=False, num_workers=None):
#         super().__init__()
#         self.batch_size = batch_size
#         self.dataset_configs = dict()
#         self.num_workers = num_workers if num_workers is not None else batch_size*2
#         if train is not None:
#             self.dataset_configs["train"] = train
#             self.train_dataloader = self._train_dataloader
#         if validation is not None:
#             self.dataset_configs["validation"] = validation
#             self.val_dataloader = self._val_dataloader
#         if test is not None:
#             self.dataset_configs["test"] = test
#             self.test_dataloader = self._test_dataloader
#         self.wrap = wrap

#     def prepare_data(self):
#         for data_cfg in self.dataset_configs.values():
#             instantiate_from_config(data_cfg)

#     def setup(self, stage=None):
#         self.datasets = dict(
#             (k, instantiate_from_config(self.dataset_configs[k]))
#             for k in self.dataset_configs)
#         if self.wrap:
#             for k in self.datasets:
#                 self.datasets[k] = WrappedDataset(self.datasets[k])

#     def _train_dataloader(self):
#         return DataLoader(self.datasets["train"], batch_size=self.batch_size,
#                           num_workers=self.num_workers, shuffle=True, collate_fn=custom_collate)

#     def _val_dataloader(self):
#         return DataLoader(self.datasets["validation"],
#                           batch_size=self.batch_size,
#                           num_workers=self.num_workers, collate_fn=custom_collate)

#     def _test_dataloader(self):
#         return DataLoader(self.datasets["test"], batch_size=self.batch_size,
#                           num_workers=self.num_workers, collate_fn=custom_collate)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_pretrain_routine_start(self, trainer, pl_module):
        # Créez toujours le répertoire configs au niveau du répertoire principal
        
        if trainer.global_rank == 0:
            # Créer les répertoires logdir et ckptdir uniquement pour le processus principal
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print("Project config")
            config_dict = OmegaConf.to_container(self.config, resolve=True)
            print(yaml.dump(config_dict))
            #print(OmegaConf.to_yaml(self.config))  # Utilisez to_yaml pour l'affichage
            print("cfgdir", self.cfgdir)
            try:
                os.makedirs(self.cfgdir, exist_ok=True)
            except FileExistsError:
                print(f"Le répertoire {self.cfgdir} existe déjà.")
            except Exception as e:
                print(f"Erreur lors de la création du répertoire {self.cfgdir}: {e}")
            # Sauvegardez la configuration principale et lightning
            OmegaConf.save(self.config, os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            # Créer un objet OmegaConf à partir du dictionnaire
            config_to_save = OmegaConf.create({"lightning": OmegaConf.to_container(self.lightning_config, resolve=True)})

            # Ensuite, sauvegarder cet objet OmegaConf
            OmegaConf.save(config_to_save, os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))
            #OmegaConf.save({"lightning": OmegaConf.to_container(self.lightning_config, resolve=True)}, 
            #            os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # Pour les processus/rangs non principaux, gérez les répertoires de log
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)

                try:
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            # pl.loggers.WandbLogger: self._wandb,
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        pass
        raise ValueError("No way wandb")
        grids = dict()
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grids[f"{split}/{k}"] = wandb.Image(grid)
        pl_module.logger.experiment.log(grids)

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            #grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=16)

            #grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid*255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, pl_module=pl_module)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()
   



    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")

# Configuration du logger Telegram
apiToken = '6312332443:AAGxzgoSAz8ls2Cxl1IX2uKTo1KnFZQnGeM'
chatID = '5272107001'
logger = TelegramLogger(apiToken, chatID)


def main():
    torch.cuda.empty_cache()
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    
    telegram_progress_logger = TelegramProgressLogger(logger, opt.base)
    logdir = None
    if opt.paperspace:
        logdir = find_latest_logdir()
        print(logdir)
        if logdir and read_status_file(logdir) == "finished":
            print("Training already completed, nothing to do.")
            return 
        elif logdir:
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
            opt.resume_from_checkpoint = ckpt
            if os.path.exists(ckpt):
                print(f"Resuming from checkpoint: {ckpt}")
            else:
                print("No checkpoint found, starting a new training session")
                ckpt = None
        else:
            print("No previous log directory found, starting a new training session")
            if opt.name:
                name = "_"+opt.name
            elif opt.base:
                cfg_fname = os.path.split(opt.base[0])[-1]
                cfg_name = os.path.splitext(cfg_fname)[0]
                name = "_"+cfg_name
            else:
                name = ""
            nowname = now+name+opt.postfix
            logdir = os.path.join("logs", nowname)
            os.makedirs(logdir, exist_ok=True)
            ckpt = None
            
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            idx = len(paths)-paths[::-1].index("logs")+1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs+opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[_tmp.index("logs")+1]
    else:
        if opt.name:
            name = "_"+opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_"+cfg_name
        else:
            name = ""
        nowname = now+name+opt.postfix
    if logdir is None :
        logdir = os.path.join("logs", nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)
    
    write_status_file(logdir, "in progress")
    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)

        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        
        # default to ddp
        #trainer_config["distributed_backend"] = 'ddp'
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not "gpus" in trainer_config:
            del trainer_config["distributed_backend"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)
        
        
        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        # NOTE wandb < 0.10.0 interferes with shutdown
        # wandb >= 0.10.0 seems to fix it but still interferes with pudb
        # debugging (wrongly sized pudb ui)
        # thus prefer testtube for now
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["wandb"]
        logger_cfg = lightning_config.logger or OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }
        # Configuration du ModelCheckpoint
        model_checkpoint_callback = ModelCheckpoint(
            monitor='val/aeloss_epoch', 
            dirpath=ckptdir,
            filename='{epoch:02d}-{val_aeloss_epoch:.2f}',
            save_top_k=3,  # Changez ce nombre selon vos besoins
            save_last=True,
            verbose=True,
            mode='min'
        )
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3

        modelckpt_cfg = lightning_config.modelcheckpoint or OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "batch_frequency": 50,
                    "max_images": 256,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    #"log_momentum": True
                }
            },
        }
        callbacks_cfg = lightning_config.callbacks or OmegaConf.create()
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        
        # Instanciation des callbacks configurés via OmegaConf et ajout du callback Telegram
        callbacks = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
        callbacks.append(telegram_progress_logger)
        callbacks.append(model_checkpoint_callback)
        
        trainer_kwargs["resume_from_checkpoint"] = ckpt
        #trainer_kwargs["max_epochs"] = config.training.epochs
        trainer_kwargs["callbacks"] = callbacks
        
        #trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
        trainer_kwargs["gpus"] = -1
        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)

        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        # if not cpu:
        #     ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        # else:
        #    ngpu = 1
        ngpu = 1

        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches or 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
            model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb; pudb.set_trace()

        # import signal
        # signal.signal(signal.SIGUSR1, melk)
        # signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                trainer.fit(model, data)
                write_status_file(logdir, "finished")
            except Exception:
                if model is not None:
                    melk()
                    raise
        #if not opt.no_test and not trainer.interrupted:
        #    trainer.test(model, data)
    except Exception:
        if opt.debug and trainer.global_rank==0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank==0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)

if __name__ == "__main__":
    main()
