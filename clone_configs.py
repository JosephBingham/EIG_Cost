#!/usr/bin/env python
"""Clone all mimic_iv Hydra configs to mimic_iv_5class equivalents.

For each mimic_iv.yaml found under extra/conf/, creates a sibling
mimic_iv_5class.yaml with the following substitutions applied in order:

  MimicIVDataset          -> MimicIV5ClassDataset
  n_features: 55          -> n_features: 54
  n_classes: 21           -> n_classes: 5
  n_groups: 30            -> n_groups: 29
  mimic_iv_split_1        -> mimic_iv_5class_split_1   (wandb artifact name)
  extra/data/mimic_iv"    -> extra/data/mimic_iv_5class"  (quoted path)
  extra/data/mimic_iv/    -> extra/data/mimic_iv_5class/  (slash-terminated path)
  dataset: mimic_iv       -> dataset: mimic_iv_5class
  dataset=mimic_iv        -> dataset=mimic_iv_5class
  name: mimic_iv          -> name: mimic_iv_5class

A safety pass runs last to undo any accidental double-replacements.

Run with --dry-run first to review all changes line-by-line.
"""
import argparse
import sys
from pathlib import Path

# Tested against all 11 actual config files. Order matters.
SUBSTITUTIONS = [
    ("MimicIVDataset",              "MimicIV5ClassDataset"),
    ("n_features: 55",              "n_features: 54"),
    ("n_features: 55,",             "n_features: 54,"),
    ("n_classes: 21",               "n_classes: 5"),
    ("n_classes: 21,",              "n_classes: 5,"),
    ("n_groups: 30",                "n_groups: 29"),
    ("n_groups: 30,",               "n_groups: 29,"),
    ("mimic_iv_split_1",            "mimic_iv_5class_split_1"),
    ("extra/data/mimic_iv/",        "extra/data/mimic_iv_5class/"),
    ("extra/data/mimic_iv\"",       "extra/data/mimic_iv_5class\""),
    ("dataset: mimic_iv",           "dataset: mimic_iv_5class"),
    ("dataset=mimic_iv",            "dataset=mimic_iv_5class"),
    ("name: mimic_iv",              "name: mimic_iv_5class"),
    # Safety guards against accidental double-replacement
    ("mimic_iv_5class_5class",      "mimic_iv_5class"),
    ("MimicIV5Class5ClassDataset",  "MimicIV5ClassDataset"),
]


def apply_substitutions(text):
    for old, new in SUBSTITUTIONS:
        text = text.replace(old, new)
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--conf-root", type=Path,
        default=Path("extra/conf"),
        help="Root of the Hydra conf tree (default: extra/conf)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be created without writing files",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite existing mimic_iv_5class.yaml files",
    )
    args = p.parse_args()

    sources = sorted(args.conf_root.rglob("mimic_iv.yaml"))
    if not sources:
        print("No mimic_iv.yaml files found under " + str(args.conf_root))
        return 1

    print("Found {} mimic_iv.yaml file(s).\n".format(len(sources)))
    created = 0
    skipped = 0

    for src in sources:
        dest = src.parent / "mimic_iv_5class.yaml"
        original = src.read_text()
        new_text = apply_substitutions(original)

        if args.dry_run:
            print("=== Would create: {} ===".format(dest))
            orig_lines = original.splitlines()
            new_lines = new_text.splitlines()
            changed = False
            for i in range(max(len(orig_lines), len(new_lines))):
                o = orig_lines[i] if i < len(orig_lines) else "(missing)"
                n = new_lines[i] if i < len(new_lines) else "(missing)"
                if o != n:
                    print("  line {:3d}  - {}".format(i + 1, o))
                    print("  line {:3d}  + {}".format(i + 1, n))
                    changed = True
            if not changed:
                print("  (no changes from source)")
            print()
        else:
            if dest.exists() and not args.force:
                print("[skip]    {} (already exists)".format(dest))
                skipped += 1
            else:
                dest.write_text(new_text)
                print("[created] {}".format(dest))
                created += 1

    if args.dry_run:
        print("Dry run complete. Re-run without --dry-run to create the files.")
    else:
        print("\nCreated: {}  Skipped: {}".format(created, skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
