"""Best-effort network block for the sandbox subprocess.

`install()` makes it impossible to create a socket through the `socket` module.
Also used as a pytest plugin (`-p trivec.lab.netguard`) when agent tests run.

The blocked socket must stay a CLASS: libraries such as `ssl` run
`class SSLSocket(socket.socket)` when imported, which breaks if socket.socket is a function.

Limit: this is a Python-level block. On Linux the runner adds `unshare -n` (a real network
namespace). On Windows this block plus the static check is all there is; see docs/design.md 7.3.
"""

import os
import socket
from typing import Any


class NetworkBlockedError(RuntimeError):
    pass


def _blocked(*_args: Any, **_kwargs: Any) -> Any:
    raise NetworkBlockedError("network access is disabled inside the TriVec sandbox")


class BlockedSocket(socket.socket):
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        _blocked()


def install() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TRIVEC_SANDBOX"] = "1"
    socket.socket = BlockedSocket  # type: ignore[misc]
    for name in ("create_connection", "getaddrinfo", "socketpair"):
        setattr(socket, name, _blocked)


def pytest_configure(config: Any) -> None:  # pytest plugin hook
    install()
