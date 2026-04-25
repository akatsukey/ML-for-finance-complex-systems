"""
benchmarking.paths
------------------
Output directory helpers for benchmark figures.

Two environment variables are honored, in the following priority order:

1. ``LIQUIDATION_BENCHMARK_PNG_DIR`` -- legacy variable, preserved for
   backwards compatibility with code that already sets it.
2. ``BENCHMARK_PNG_DIR`` -- generic variable introduced with the
   :mod:`benchmarking` subpackage.

If neither is set the directory defaults to ``<package_parent>/results_benchmark/``
where ``<package_parent>`` is the directory containing the ``benchmarking``
package (i.e. the ``jfb-for-implicit-oc`` folder).  Callers (notably the
:mod:`liquidation_benchmark` shim) can override the default via the
``default_dir`` argument of :func:`benchmark_png_dir`.
"""

from __future__ import annotations

import os
from typing import Optional


_PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_BENCHMARK_PNG_DIR = os.path.join(_PACKAGE_PARENT, "results", "benchmark")


def benchmark_png_dir(default_dir: Optional[str] = None) -> str:
    """Return the benchmark PNG output directory.

    Parameters
    ----------
    default_dir : str, optional
        Fallback directory to use when neither environment variable is set.
        When ``None`` (default) the package-level default
        ``<package_parent>/results_benchmark`` is used.

    Returns
    -------
    str
        Absolute path to the output directory.  The directory is **not**
        created by this function; use :func:`benchmark_png_path` to ensure
        the directory exists before writing a file.

    Notes
    -----
    Resolution order:

    1. ``LIQUIDATION_BENCHMARK_PNG_DIR`` environment variable.
    2. ``BENCHMARK_PNG_DIR`` environment variable.
    3. ``default_dir`` argument if provided.
    4. Package-level default ``<package_parent>/results_benchmark``.
    """
    legacy = os.environ.get("LIQUIDATION_BENCHMARK_PNG_DIR", "").strip()
    if legacy:
        return os.path.abspath(legacy)

    generic = os.environ.get("BENCHMARK_PNG_DIR", "").strip()
    if generic:
        return os.path.abspath(generic)

    if default_dir is not None:
        return os.path.abspath(default_dir)
    return os.path.abspath(_DEFAULT_BENCHMARK_PNG_DIR)


def benchmark_png_path(
    filename: str,
    subdir: Optional[str] = None,
    default_dir: Optional[str] = None,
) -> str:
    """Return the full path for a PNG under :func:`benchmark_png_dir`.

    Creates the parent directory on demand so the caller can pass the result
    straight to ``fig.savefig``.

    Parameters
    ----------
    filename : str
        The file name (not a full path) of the image to write.
    subdir : str, optional
        Optional sub-directory under :func:`benchmark_png_dir` -- e.g.
        ``"almgren_single"`` or ``"almgren_multi"`` -- to keep outputs
        from different problems organised.
    default_dir : str, optional
        Forwarded to :func:`benchmark_png_dir`.
    """
    root = benchmark_png_dir(default_dir=default_dir)
    folder = os.path.join(root, subdir) if subdir else root
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, filename)
