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
        if "." in k:  # nested: curriculum.window=10
            outer, inner = k.split(".", 1)
            kw.setdefault(outer, {})[inner] = _coerce(v)
        else:
            kw[k] = _coerce(v)
    return build(cls, kw)


def build(cls, kw: dict):
    names = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(kw) - set(names)
    if unknown:
        raise SystemExit(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    for k, v in list(kw.items()):
        f = names[k]
        sub = f.default_factory() if f.default_factory is not dataclasses.MISSING else None
        if isinstance(v, dict) and dataclasses.is_dataclass(sub):
            kw[k] = build(type(sub), v)
        elif isinstance(v, list) and isinstance(f.default, tuple):
            kw[k] = tuple(v)
    return cls(**kw)
