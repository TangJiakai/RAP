import random
import numpy as np
import torch
from colorama import init
import os
import sys
import shutil
import hashlib
import datetime
import time
import logging
import colorlog
import re
import importlib
from torch.utils.tensorboard import SummaryWriter

log_colors_config = {
    "DEBUG": "cyan",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "red",
}

os.chdir(sys.path[0])


def init_data_path(config):
    model = config['model']
    dataset = config['dataset']
    resume_id = config['resume_id']
    id = config['id']
    id = config['id'] = id if id is not None else int(time.time())

    dataset_type = 'clean' if config['noise_ratio'] == 0. else config['noise_ratio']

    config['train_data_path'] = f"../data/pro-{dataset}/{dataset}-train.{dataset_type}"
    config['valid_data_path'] = f"../data/pro-{dataset}/{dataset}-valid.{dataset_type}"
    config['test_data_path'] = f"../data/pro-{dataset}/{dataset}-test.{dataset_type}"

    if resume_id:
        id = config['id'] = resume_id
        resume_dir = f"../model_cpt/{dataset}_{dataset_type}/{model}/{resume_id}/"
        if not os.path.isdir(resume_dir):
            raise ValueError('You must specified valid resume_id!')
        config['resume_file'] = os.path.join(resume_dir, os.listdir(resume_dir)[0])
    
    saved_model_dir = f"../model_cpt/{dataset}_{dataset_type}/{model}/{id}/"

    if config['saved']:
        ensure_dir(saved_model_dir)
        config['saved_model_path_pre'] = saved_model_dir
    
    if config['use_tensorboard']:
        tensorboard_dir = f"../tensorboard/{dataset}_{dataset_type}/{model}/{id}/"
        ensure_dir(saved_model_dir)
        config['tensorboard_dir'] = tensorboard_dir

    return config


def init_seed(seed, reproducibility=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    
def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")
    return cur


class RemoveColorFilter(logging.Filter):
    def filter(self, record):
        if record:
            ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
            record.msg = ansi_escape.sub("", str(record.msg))
        return True


def init_logger(config):
    init(autoreset=True)

    sfmt = "%(log_color)s%(asctime)-15s %(levelname)s  %(message)s"
    sdatefmt = "%d %b %H:%M:%S"
    sformatter = colorlog.ColoredFormatter(sfmt, sdatefmt, log_colors=log_colors_config)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(sformatter)

    if config['log']:
        dataset_type = 'clean' if config['noise_ratio'] == 0. else config['noise_ratio']
        LOGROOT = f"../log/{config['dataset']}_{dataset_type}/{config['model']}/{config['id']}/"
        dir_name = os.path.dirname(LOGROOT)
        ensure_dir(dir_name)
        config_str = "".join([str(key) for key in config.values()])
        md5 = hashlib.md5(config_str.encode(encoding="utf-8")).hexdigest()[:6]
        logfilename = "{}-{}-{}-{}-{}-{}.log".format(
            config["dataset"], config['noise_ratio'], config["model"], config['id'], get_local_time(), md5
        )

        logfilepath = os.path.join(LOGROOT, logfilename)
        config['log_file_path'] = logfilepath

        filefmt = "%(asctime)-15s %(levelname)s  %(message)s"
        fileformatter = logging.Formatter(filefmt)
        fh = logging.FileHandler(logfilepath)
        fh.setLevel(logging.INFO)
        fh.setFormatter(fileformatter)
        remove_color_filter = RemoveColorFilter()
        fh.addFilter(remove_color_filter)
        logging.basicConfig(level=logging.INFO, handlers=[sh, fh])
    else:
        logging.basicConfig(level=logging.INFO, handlers=[sh])


def init_device(config):
    gpu_id = str(config['gpu_id'])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    config["device"] = torch.device("cpu") if len(gpu_id) == 0 or not torch.cuda.is_available() or not config['use_gpu'] else torch.device("cuda")

    
def set_color(log, color, highlight=True):
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"


def early_stopping(value, best, cur_step, max_step):
    stop_flag = False
    update_flag = False
    if value >= best:
        cur_step = 0
        best = value
        update_flag = True
    else:
        cur_step += 1
        if cur_step > max_step:
            stop_flag = True
    if max_step <= 0:
        stop_flag = False
    return best, cur_step, stop_flag, update_flag 


def dict2str(result_dict):
    return "\t".join(
        [str(metric) + " : " + str(value) for metric, value in result_dict.items()]
    )


def clear_dir(dir_path):
    shutil.rmtree(dir_path)
    os.mkdir(dir_path)

def get_model(model):
    module = importlib.import_module(f'model.{model}')
    return getattr(module, model)

def get_tensorboard(tensorboard_dir, purge_step=0):
    if tensorboard_dir is None:
        return None
    writer = SummaryWriter(tensorboard_dir, purge_step=purge_step)
    return writer

def remove_tensorboard_dir(tensorboard_dir):
    files_and_dirs = os.listdir(tensorboard_dir)
    for name in files_and_dirs:
        file_or_dir_name = os.path.join(tensorboard_dir, name)
        if os.path.isdir(file_or_dir_name):
            shutil.rmtree(file_or_dir_name)

def output_config(config):
    args_info = "\n"
    args_info += set_color("Parameters:\n", "pink")
    for key, value in config.items():
        args_info += set_color(f"{key}", "cyan") + " =" + set_color(f" {value}", "yellow") + '\n' 
    args_info += "\n\n"
    
    return args_info