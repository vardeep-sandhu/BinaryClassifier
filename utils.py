import os
import yaml
from easydict import EasyDict
import copy 
import logging
import functools
import sys

def load_cfg_from_cfg_file(file):
    assert os.path.isfile(file) and file.endswith('.yaml'), \
        '{} is not a yaml file'.format(file)

    with open(file, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    return EasyDict(cfg_from_file)

def merge_cfg_with_args(cfg, args_list):
    new_cfg = copy.deepcopy(cfg)
    assert len(args_list) % 2 == 0
    for key, v in zip(args_list[0::2], args_list[1::2]):
        subkey = key.split('.')[-1]
        new_cfg[subkey] = v
    print(new_cfg)
    return new_cfg

# so that calling setup_logger multiple times won't add many handlers
@functools.lru_cache()
def get_logger(output):
    logger = logging.getLogger("main-logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(plain_formatter)
    logger.addHandler(ch)

    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)
    return logger

# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return open(filename, "a")