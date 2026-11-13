"""Cross-validation of Lumen's ImageJ-compatible commands against ImageJ 1.54p itself (see docs/CROSSVALIDATION.md).

``harness.py`` runs every macro in ``macros/`` twice – in a headless-ish ImageJ 1.x batch process (the real ``ij.jar``)
and through Lumen's macro layer (``lumen.core.ij_compat.IJ.macro``) – and compares pixels, Results rows and printed
values.  ``test_imagej_crossval.py`` asserts the recorded tolerances; it is opt-in (``-m crossval``).
"""
