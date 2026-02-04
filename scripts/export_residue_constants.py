import argparse
import numpy as np

from openfold.np import residue_constants as rc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="residue_constants_ref.npz")
    args = parser.parse_args()

    np.savez(
        args.output,
        restype_rigid_group_default_frame=rc.restype_rigid_group_default_frame,
        restype_atom14_to_rigid_group=rc.restype_atom14_to_rigid_group,
        restype_atom14_mask=rc.restype_atom14_mask,
        restype_atom14_rigid_group_positions=rc.restype_atom14_rigid_group_positions,
    )


if __name__ == "__main__":
    main()
