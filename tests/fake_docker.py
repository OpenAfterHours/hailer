"""A scripted ``docker`` CLI with state (test helper, standard library only).

:class:`FakeDocker` implements the :class:`~hailer.kernel_docker.DockerRunner` protocol and keeps
containers and networks the way Docker 29 does: each gets an id when it is created (``run -d``,
``create`` and ``network create`` print it), names are unique, ``--filter label=...`` narrows
``ps`` and ``network ls``, and removing something that is not there answers like Docker (``rm -f``
exits 0 with "No such container" on stderr; ``network rm`` exits 1 with "not found"). A container
in state ``removing`` answers ``rm -f`` with "removal ... is already in progress" (and is then
gone); ``network_busy`` makes the next N ``network rm`` calls answer "has active endpoints"
(containers still detaching), as when ``kernel stop`` races another terminal's cleanup.

``fail`` maps an argv prefix to (exit code, stderr); ``timeout`` holds argv prefixes that time out;
``stream_hook`` is called with the argv inside :meth:`stream` (e.g. to raise KeyboardInterrupt).
Nothing is ever run.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field

from hailer import __version__
from hailer.kernel_docker import LABEL_ROLE, LABEL_WORKSPACE, docker_not_installed
from hailer.kernel_image import VERSION_LABEL

TOKEN = "unit-test-token-0123456789"


@dataclass
class Container:
    id: str
    name: str
    labels: dict[str, str]
    state: str = "running"  # created | running | exited | removing
    exit_code: int = 0
    oom: bool = False
    networks: list[str] = field(default_factory=list)
    argv: list[str] = field(default_factory=list)


@dataclass
class Network:
    id: str
    name: str
    labels: dict[str, str]


def _labels_of(args: list[str]) -> dict[str, str]:
    return dict(args[i + 1].split("=", 1) for i, a in enumerate(args) if a == "--label")


class FakeDocker:
    def __init__(
        self,
        *,
        installed: bool = True,
        engine: str | None = "29.4.3 linux",
        ncpu: int = 16,
        image_version: str | None = __version__,
        pulled_version: str = __version__,
        pull_code: int = 0,
        pull_error: str = "",
        build_code: int = 0,
    ) -> None:
        self.installed = installed
        self.engine = engine  # None: the engine is down
        self.ncpu = ncpu
        self.image_version = image_version  # None: the image is not on this machine
        self.pulled_version = pulled_version
        self.pull_code = pull_code
        self.pull_error = pull_error
        self.build_code = build_code
        self.containers: dict[str, Container] = {}
        self.networks: dict[str, Network] = {}
        self.logs = ["marimo is starting", f"URL: http://0.0.0.0:2718?access_token={TOKEN}"]
        self.fail: dict[tuple[str, ...], tuple[int, str]] = {}
        self.timeout: set[tuple[str, ...]] = set()
        self.stream_hook = None
        self.network_busy = 0
        self.calls: list[list[str]] = []
        self.streams: list[list[str]] = []
        self._created = 0

    # -- state helpers for tests --------------------------------------------- #

    def _new_id(self, kind: str) -> str:
        self._created += 1
        return hashlib.sha256(f"{kind}-{self._created}".encode()).hexdigest()

    def add_container(self, name: str, *, workspace: str | None = None, role: str = "kernel", state: str = "running", **labels: str) -> Container:
        """A container that exists before the test's command (a leftover, or another workspace's)."""
        all_labels = dict(labels)
        if workspace is not None:
            all_labels[LABEL_WORKSPACE] = workspace
            all_labels[LABEL_ROLE] = role
        container = Container(self._new_id("container"), name, all_labels, state=state)
        self.containers[container.id] = container
        return container

    def add_network(self, name: str, *, workspace: str | None = None) -> Network:
        labels = {LABEL_WORKSPACE: workspace, LABEL_ROLE: "network"} if workspace is not None else {}
        network = Network(self._new_id("network"), name, labels)
        self.networks[network.id] = network
        return network

    def container(self, ref: str) -> Container | None:
        """By id, id prefix (12+ characters) or name."""
        for container in self.containers.values():
            if ref == container.name or ref == container.id or (len(ref) >= 12 and container.id.startswith(ref)):
                return container
        return None

    def network(self, ref: str) -> Network | None:
        for network in self.networks.values():
            if ref == network.name or ref == network.id or (len(ref) >= 12 and network.id.startswith(ref)):
                return network
        return None

    def names(self, *, running: bool | None = None) -> list[str]:
        return [c.name for c in self.containers.values() if running is None or (c.state == "running") == running]

    def commands(self, *head: str) -> list[list[str]]:
        return [c for c in self.calls if c[: len(head)] == list(head)]

    # -- the DockerRunner protocol ------------------------------------------- #

    @staticmethod
    def _done(args: list[str], code: int = 0, out: str = "", err: str = "") -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["docker", *args], code, out, err)

    def run(self, args, *, timeout=None, check=False):
        args = list(args)
        self.calls.append(args)
        if not self.installed:
            raise docker_not_installed()
        for prefix in self.timeout:
            if tuple(args[: len(prefix)]) == prefix:
                raise subprocess.TimeoutExpired(["docker", *args], timeout)
        for prefix, (code, err) in self.fail.items():
            if tuple(args[: len(prefix)]) == prefix:
                return self._done(args, code, "", err)
        if args[0] != "version" and self.engine is None:
            return self._done(args, 1, "", "error during connect: open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.")
        result = self._answer(args)
        if check and result.returncode != 0:
            from hailer.errors import KernelRuntimeError

            raise KernelRuntimeError(f"docker {' '.join(args[:2])} failed (exit code {result.returncode}).", hint=result.stderr)
        return result

    def _answer(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        head = args[0]
        if head == "version":
            if self.engine is None:
                return self._done(args, 1, "", "error during connect: open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.")
            return self._done(args, 0, self.engine + "\n")
        if head == "info":
            return self._done(args, 0, f"{self.ncpu}\n")
        if args[:2] == ["image", "inspect"]:
            if self.image_version is None:
                return self._done(args, 1, "", f"Error response from daemon: No such image: {args[-1]}")
            return self._done(args, 0, json.dumps({VERSION_LABEL: self.image_version}) + "\n")
        if head == "ps":
            return self._ps(args)
        if args[:2] == ["network", "ls"]:
            rows = [n for n in self.networks.values() if self._matches(n.labels, n.name, args)]
            fmt = args[args.index("--format") + 1]
            return self._done(args, 0, "".join(fmt.replace("{{.ID}}", n.id[:12]).replace("{{.Name}}", n.name) + "\n" for n in rows))
        if args[:2] == ["network", "create"]:
            name = args[-1]
            if self.network(name) is not None:
                return self._done(args, 1, "", f"Error response from daemon: network with name {name} already exists")
            network = Network(self._new_id("network"), name, _labels_of(args))
            self.networks[network.id] = network
            return self._done(args, 0, network.id + "\n")
        if args[:2] == ["network", "connect"]:
            network, container = self.network(args[2]), self.container(args[3])
            if network is None or container is None:
                return self._done(args, 1, "", f"Error response from daemon: No such network: {args[2]}")
            container.networks.append(network.id)
            return self._done(args)
        if args[:2] == ["network", "rm"]:
            network = self.network(args[2])
            if network is None:
                return self._done(args, 1, "", f"Error response from daemon: network {args[2]} not found\nexit status 1")
            busy = any(network.id in c.networks and c.state == "running" for c in self.containers.values())
            if busy or self.network_busy > 0:
                self.network_busy = max(0, self.network_busy - 1)
                return self._done(args, 1, "", f"Error response from daemon: error while removing network: network {network.name} id {network.id} has active endpoints")
            del self.networks[network.id]
            return self._done(args, 0, args[2] + "\n")
        if head in ("run", "create"):
            name = args[args.index("--name") + 1]
            if self.container(name) is not None:
                return self._done(args, 125, "", f'docker: Error response from daemon: Conflict. The container name "/{name}" is already in use.')
            container = Container(self._new_id("container"), name, _labels_of(args), state="running" if head == "run" else "created", argv=args)
            if "--network" in args:
                network = self.network(args[args.index("--network") + 1])
                if network is None:
                    return self._done(args, 125, "", "docker: Error response from daemon: network not found.")
                container.networks.append(network.id)
            self.containers[container.id] = container
            return self._done(args, 0, container.id + "\n")
        if head == "start":
            container = self.container(args[1])
            if container is None:
                return self._done(args, 1, "", f"Error response from daemon: No such container: {args[1]}")
            container.state = "running"
            return self._done(args, 0, args[1] + "\n")
        if head == "rm":
            refs = [a for a in args[1:] if not a.startswith("-")]
            out, err = [], []
            code = 0
            for ref in refs:
                container = self.container(ref)
                if container is None:
                    err.append(f"Error response from daemon: No such container: {ref}")
                elif container.state == "removing":  # another command is removing it right now
                    del self.containers[container.id]
                    err.append(f"Error response from daemon: removal of container {ref} is already in progress")
                    code = 1
                else:
                    del self.containers[container.id]
                    out.append(ref)
            return self._done(args, code, "".join(f"{line}\n" for line in out), "\n".join(err))
        if head == "inspect":
            return self._inspect(args)
        if head == "logs":
            return self._done(args, 0, "\n".join(self.logs) + "\n")
        return self._done(args)

    @staticmethod
    def _filters(args: list[str]) -> list[str]:
        return [args[i + 1] for i, a in enumerate(args) if a == "--filter"]

    def _matches(self, labels: dict[str, str], name: str, args: list[str]) -> bool:
        for flt in self._filters(args):
            kind, _, value = flt.partition("=")
            if kind == "label":
                key, has_value, wanted = value.partition("=")
                if key not in labels or (has_value and labels[key] != wanted):
                    return False
            elif kind == "name" and not re.search(value, name):
                return False
        return True

    def _ps(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        rows = [c for c in self.containers.values() if self._matches(c.labels, c.name, args)]
        if "-a" not in args:
            rows = [c for c in rows if c.state == "running"]
        fmt = args[args.index("--format") + 1] if "--format" in args else "{{.Names}}"

        def render(c: Container) -> str:
            text = fmt.replace("{{.ID}}", c.id[:12]).replace("{{.Names}}", c.name).replace("{{.State}}", c.state)
            return re.sub(r'\{\{\.Label "([^"]+)"\}\}', lambda m: c.labels.get(m.group(1), ""), text)

        return self._done(args, 0, "".join(render(c) + "\n" for c in rows))

    def _inspect(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        rest = args[1:]
        kind = "container"
        if rest[:1] == ["--type"]:
            kind, rest = rest[1], rest[2:]
        fmt, refs = rest[1], rest[2:]
        out, missing = [], []
        if kind == "network":
            for ref in refs:
                network = self.network(ref)
                if network is None:
                    missing.append(ref)
                else:
                    out.append(fmt.replace("{{.Id}}", network.id))
            err = "\n".join(f"Error response from daemon: network {ref} not found" for ref in missing)
            return self._done(args, 1 if missing else 0, "".join(f"{line}\n" for line in out), err)
        for ref in refs:
            c = self.container(ref)
            if c is None:
                missing.append(ref)
                continue
            if fmt == "{{json .Config.Labels}}":
                out.append(json.dumps(c.labels))
                continue
            text = (
                fmt.replace("{{.Id}}", c.id)
                .replace("{{.Name}}", "/" + c.name)
                .replace("{{.State.Running}}", "true" if c.state == "running" else "false")
                .replace("{{.State.ExitCode}}", str(c.exit_code))
                .replace("{{.State.Status}}", c.state)
                .replace("{{.State.OOMKilled}}", "true" if c.oom else "false")
            )
            out.append(text)
        err = "\n".join(f"error: no such object: {ref}" for ref in missing)
        return self._done(args, 1 if missing else 0, "".join(f"{line}\n" for line in out), err)

    def stream(self, args, *, keep_errors=False):
        args = list(args)
        self.streams.append(args)
        if not self.installed:
            raise docker_not_installed()
        if self.stream_hook is not None:
            self.stream_hook(args)
        if args[0] == "pull":
            if self.pull_code == 0:
                self.image_version = self.pulled_version
            return self._done(args, self.pull_code, None, self.pull_error if keep_errors else "")
        if args[0] == "build":
            return self._done(args, self.build_code, None, "")
        return self._done(args, 0, None, "")
