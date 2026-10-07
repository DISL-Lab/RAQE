"""Point pyjnius at a usable JDK before Pyserini imports it.

Anserini needs the ``jdk.incubator.vector`` module. A JVM without it — a
JetBrains "nomod" runtime shipped inside a conda environment is the common case —
aborts the whole process at import time with

    Error occurred during initialization of boot layer
    java.lang.module.FindException: Module jdk.incubator.vector not found

which is fatal and gives no hint about the cause. Importing this module before
``pyserini`` picks the first candidate JDK that actually resolves that module and
exports ``JVM_PATH``/``JAVA_HOME`` for it. An existing ``JVM_PATH`` is never
overridden, so an explicit choice always wins. See docs/ENVIRONMENT.md.
"""

import os
import subprocess
from pathlib import Path

_SYSTEM_JDKS = (
    "/usr/lib/jvm/java-21-openjdk-amd64",
    "/usr/lib/jvm/java-21-openjdk",
    "/usr/lib/jvm/default-java",
)


def _supports_vector_module(java_home: Path) -> bool:
    """True if `java --add-modules jdk.incubator.vector -version` succeeds."""
    java = java_home / "bin" / "java"
    if not java.exists():
        # No launcher next to libjvm.so (a bare JRE layout): fall back to trusting it.
        return True
    try:
        proc = subprocess.run([str(java), "--add-modules", "jdk.incubator.vector", "-version"],
                              capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _candidate_java_homes():
    """System JDKs first: a conda env often ships a JVM that lacks the module."""
    for path in _SYSTEM_JDKS:
        yield Path(path)
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        yield Path(java_home)
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        yield Path(conda_prefix)
        yield Path(conda_prefix) / "lib" / "jvm"


def ensure_jvm() -> str | None:
    """Set JVM_PATH/JAVA_HOME to a JDK that can load Anserini, if one is found."""
    current = os.environ.get("JVM_PATH")
    if current and Path(current).exists():
        return current

    fallback = None
    for java_home in _candidate_java_homes():
        libjvm = java_home / "lib" / "server" / "libjvm.so"
        if not libjvm.exists():
            continue
        if _supports_vector_module(java_home):
            os.environ["JVM_PATH"] = str(libjvm)
            os.environ["JAVA_HOME"] = str(java_home)
            return str(libjvm)
        if fallback is None:
            fallback = (libjvm, java_home)

    if fallback is not None:
        # Nothing validated; use what we have and let Pyserini report the failure.
        libjvm, java_home = fallback
        os.environ["JVM_PATH"] = str(libjvm)
        os.environ.setdefault("JAVA_HOME", str(java_home))
        return str(libjvm)
    return None


ensure_jvm()
