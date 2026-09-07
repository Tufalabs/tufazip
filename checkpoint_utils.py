"""Deterministic, atomic checkpoint helpers for TufaZip."""

import json
import os
import random

import numpy as np
import torch


CHECKPOINT_VERSION = 1
REQUEUE_EXIT_CODE = 75
LATEST_NAME = "latest.pt"
DONE_NAME = "DONE"
STOP_NAME = "STOP_REQUESTED"


def checkpoint_path(directory):
    return os.path.join(directory, LATEST_NAME)


def done_path(directory):
    return os.path.join(directory, DONE_NAME)


def stop_path(directory):
    return os.path.join(directory, STOP_NAME)


def atomic_torch_save(value, path):
    """Write a torch checkpoint without ever exposing a partial latest.pt."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = "{}.tmp.{}".format(path, os.getpid())
    try:
        with open(temporary, "wb") as checkpoint_file:
            torch.save(value, checkpoint_file)
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_text(value, path):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = "{}.tmp.{}".format(path, os.getpid())
    try:
        with open(temporary, "w") as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def config_signature(args, excluded=()):
    """JSON-stable configuration used to reject an incompatible resume."""
    excluded = set(excluded)
    values = {
        key: value for key, value in vars(args).items()
        if key not in excluded
    }
    return json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)


def flush_and_sync(file_object):
    if file_object is None or file_object.closed:
        return
    file_object.flush()
    os.fsync(file_object.fileno())


def load_checkpoint(directory, device):
    path = checkpoint_path(directory)
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location=device, weights_only=False)


def mark_done(directory):
    atomic_text("complete\n", done_path(directory))


def stop_requested(directory):
    return bool(directory) and os.path.exists(stop_path(directory))
