"""YAML config + `--set key=value` overrides for the dataclass configs."""
from __future__ import annotations

import argparse
import dataclasses

import yaml


def _coerce(v: str):
    return yaml.safe_load(v)


def load_config(cls, argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML file")
    ap.add_argument("--set", nargs="*", default=[], help="overrides key=value (YAML-parsed)")
    args = ap.parse_args(argv)
    kw = {}
    if args.config:
        with open(args.config) as f:
            kw.update(yaml.safe_load(f) or {})
    for kv in args.set:
        k, v = kv.split("=", 1)
        kw[k] = _coerce(v)
    names = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(kw) - set(names)
    if unknown:
        raise SystemExit(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    for k, v in kw.items():
        if isinstance(v, list) and isinstance(names[k].default, tuple):
            kw[k] = tuple(v)
    return cls(**kw)
