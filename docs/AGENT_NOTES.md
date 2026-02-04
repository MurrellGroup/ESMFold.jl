# Agent Notes

- Before running any command or editing any file in this project, write a short two-sentence summary of what I’m about to do so the user can follow along, and then proceed without waiting for confirmation.

- Current issue: StructureModule parity now matches through intermediates; `positions`, `frames`, `angles`, and `states` are within ~1e-5 of Python after fixing AngleResnet reshape (C-order) and rigid rotation layout. The remaining large mismatch is isolated to `lddt_head`/`plddt` (and `categorical_lddt`), while other heads are close; next step is to inspect and instrument the LDDT head path in Julia vs Python.
