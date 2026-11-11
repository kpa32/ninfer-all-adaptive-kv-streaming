#!/usr/bin/env python3
"""R0: the parity checklist for --kv-stream, expressed as checks that must pass.

The requirement is that a streamed run behaves like the resident one: the same
checkpoints, the same sessions, the same disk slots. Each check below states one
baseline capability and reads its answer out of a served run. It is a harness,
not a registered test: it starts servers, drives them over HTTP and prints a
table, so a red line is a fact about the tree rather than a broken build.

Usage (on the test host, with a built ninfer-serve and a model artifact):

    python3 tools/bench/kv_stream/parity.py --model ~/...ninfer-v3.ninfer \
        --bin build-verify/apps/ninfer-serve

Every check writes its server log to --logs (default /tmp/kvlogs/parity) and its
requests to the same directory, so a failure can be read afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


SERVE_FLAGS = [
    "--max-concurrency", "1",
    "--kv-dtype", "rk8v4",
    "--max-context", "131072",
    "--kv-capacity", "auto",
    "--spec", "mtp",
    "--draft-tokens", "3",
    "--adaptive-mtp",
    "--prefill-chunk", "256",
    "--gdn-state-fp16",
    "--embedding-residency", "host",
    "--log-level", "info",
]

# The startup line the inventory checks read:
#   context cache | 1 active + 0 cached device states | host 1 states, 0 B KV | private 1 | shared 0 | anchors 0
INVENTORY = re.compile(
    r"context cache \|.*\| host (\d+) states, (\d+) B KV \| private (\d+) \| shared (\d+) \| anchors (\d+)")
STREAM_START = re.compile(r"\[stream\] request starts: reuse=(\d+) base=(\d+) valid=(\d+)")
FATAL = re.compile(r"worker crash|FATAL engine|terminate called")


@dataclass
class Result:
    name: str
    ok: bool
    detail: str


class Server:
    def __init__(self, args: argparse.Namespace, name: str, extra: list[str] | None = None):
        self.port = args.port
        self.log = Path(args.logs) / f"serve_{name}.log"
        self.log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [args.bin, args.model, "--model-id", "test", "--host", "127.0.0.1",
               "--port", str(self.port), "--kv-stream", *SERVE_FLAGS, *(extra or [])]
        self.cmd = cmd
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> "Server":
        env = dict(os.environ, NINFER_STREAM_DEBUG="1", NINFER_ADMISSION_DEBUG="1")
        self.handle = self.log.open("w")
        self.proc = subprocess.Popen(self.cmd, stdout=self.handle, stderr=subprocess.STDOUT,
                                     env=env, preexec_fn=os.setsid)
        deadline = time.time() + 300
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"serve exited before becoming ready, see {self.log}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/v1/models", timeout=2):
                    return self
            except (urllib.error.URLError, OSError):
                time.sleep(2)
        raise RuntimeError(f"serve did not become ready, see {self.log}")

    def __exit__(self, *_exc) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=30)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
        self.handle.close()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def text(self) -> str:
        return self.log.read_text(errors="replace") if self.log.exists() else ""

    def post(self, path: str, body: dict | None, timeout: float) -> tuple[int, str]:
        data = json.dumps(body).encode() if body is not None else b""
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data,
                                         headers={"content-type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as answer:
                return answer.status, answer.read().decode(errors="replace")
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode(errors="replace")
        except (urllib.error.URLError, OSError) as error:
            return 0, str(error)

    def ask(self, prompt: str, max_tokens: int = 16, timeout: float = 600) -> tuple[int, str]:
        body = {"model": "test", "max_tokens": max_tokens, "temperature": 0, "stream": False,
                "messages": [{"role": "user", "content": prompt}]}
        return self.post("/v1/chat/completions", body, timeout)

    def stream_then_cancel(self, prompt: str, seconds: float) -> None:
        """Starts a streamed answer and disconnects, which is what a client cancel is."""
        body = {"model": "test", "max_tokens": 4096, "temperature": 0, "stream": True,
                "messages": [{"role": "user", "content": prompt}]}
        data = json.dumps(body).encode()
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                                         data=data, headers={"content-type": "application/json"},
                                         method="POST")
        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=seconds) as answer:
                while time.time() - started < seconds:
                    if not answer.readline():
                        break
        except (urllib.error.URLError, OSError):
            pass


def inventory(text: str) -> tuple[int, int, int, int, int] | None:
    match = INVENTORY.search(text)
    return tuple(int(group) for group in match.groups()) if match else None


def stream_starts(text: str) -> list[tuple[int, int, int]]:
    return [tuple(int(group) for group in match.groups())
            for match in STREAM_START.finditer(text)]


def hits(text: str) -> int:
    """Published continuation hits, as the serve's own cache line counts them."""
    return len(re.findall(r"cache (\d+) \(\d+\.\d+%,", text))


def check_inventory(server: Server) -> Result:
    """The checkpoint inventory a resident run has: endpoints, anchors, several owners."""
    found = inventory(server.text())
    if found is None:
        return Result("P1 inventory", False, "no context cache line in the server log")
    states, host_kv, private, shared, anchors = found
    detail = f"host states={states} host_kv={host_kv}B private={private} shared={shared} anchors={anchors}"
    return Result("P1 inventory", anchors >= 1 and private >= 2, detail)


def check_repeat(server: Server, prompt: str) -> Result:
    """The same prompt twice: the second one must reuse the prompt's own checkpoint."""
    first, _ = server.ask(prompt)
    before = len(stream_starts(server.text()))
    second, _ = server.ask(prompt)
    starts = stream_starts(server.text())[before:]
    reused = starts[0][0] if starts else -1
    return Result("P2 same prompt twice", first == 200 and second == 200 and reused == 1,
                  f"http {first}/{second} reuse={reused}")


def check_cancel(server: Server, prompt: str) -> Result:
    """A cancelled prefill keeps what it computed, and the next request can use it."""
    first, _ = server.ask(prompt)
    server.stream_then_cancel(prompt + "\n\nContinue the same report.", 3.0)
    alive = server.alive()
    before = len(stream_starts(server.text()))
    second, _ = server.ask(prompt)
    starts = stream_starts(server.text())[before:]
    reused = starts[0][0] if starts else -1
    return Result("P3 cancel then continue", first == 200 and second == 200 and alive and reused == 1,
                  f"http {first}/{second} alive={alive} reuse={reused}")


def check_two_conversations(server: Server, first: str, second: str) -> Result:
    """Two conversations keep their own contexts: A, B, A must reuse A the second time."""
    code_a, _ = server.ask(first)
    code_b, _ = server.ask(second)
    before = len(stream_starts(server.text()))
    code_a2, _ = server.ask(first)
    starts = stream_starts(server.text())[before:]
    reused = starts[0][0] if starts else -1
    return Result("P4 two conversations", min(code_a, code_b, code_a2) == 200 and reused == 1,
                  f"http {code_a}/{code_b}/{code_a2} reuse={reused}")


def check_slot(server: Server, prompt: str) -> Result:
    """The disk slot round trip: save, erase, restore, and continue the restored session."""
    code, _ = server.ask(prompt)
    save, save_text = server.post("/slots/1?action=save", None, 600)
    erase, _ = server.post("/slots/1?action=erase", None, 600)
    restore, restore_text = server.post("/slots/1?action=restore", None, 600)
    after, _ = server.ask(prompt)
    ok = min(code, after) == 200 and save == 200 and restore == 200
    detail = f"ask={code} save={save} restore={restore} after={after}"
    if save != 200:
        detail += " save_said=" + (save_text.strip().splitlines() or [""])[-1][:120]
    if restore != 200:
        detail += " restore_said=" + (restore_text.strip().splitlines() or [""])[-1][:120]
    return Result("P5 disk slot round trip", ok and erase in (200, 204), detail)


def check_fatal(server: Server) -> Result:
    """Whatever the checks above did, the process must still be serving."""
    found = FATAL.findall(server.text())
    return Result("P6 process survives", server.alive() and not found,
                  f"alive={server.alive()} fatal={found[:2]}")


def sentence(repeat: int, text: str) -> str:
    return (text + " ") * repeat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--bin", default="build-verify/apps/ninfer-serve")
    parser.add_argument("--logs", default="/tmp/kvlogs/parity")
    parser.add_argument("--port", type=int, default=8131)
    parser.add_argument("--only", default="", help="comma separated check prefixes, e.g. P1,P2")
    args = parser.parse_args()
    wanted = [item.strip() for item in args.only.split(",") if item.strip()]

    def selected(name: str) -> bool:
        return not wanted or any(name.startswith(prefix) for prefix in wanted)

    long_prompt = sentence(60, "NInfer keeps every KV page of the context in pinned host memory "
                               "and streams the newest window onto the device.")
    other_prompt = sentence(60, "A different subject, entirely other words, and no shared prefix.")
    results: list[Result] = []

    with Server(args, "parity") as server:
        if selected("P1"):
            results.append(check_inventory(server))
        if selected("P2"):
            results.append(check_repeat(server, long_prompt))
        if selected("P3"):
            results.append(check_cancel(server, long_prompt))
        if selected("P4"):
            results.append(check_two_conversations(server, long_prompt, other_prompt))
        if selected("P6"):
            results.append(check_fatal(server))

    if selected("P5"):
        slots = Path(args.logs) / "slots"
        slots.mkdir(parents=True, exist_ok=True)
        with Server(args, "slots", ["--slot-save-path", str(slots)]) as server:
            results.append(check_slot(server, long_prompt))

    width = max(len(result.name) for result in results) if results else 10
    failures = 0
    for result in results:
        mark = "PASS" if result.ok else "FAIL"
        failures += 0 if result.ok else 1
        print(f"{mark}  {result.name.ljust(width)}  {result.detail}")
    print(f"\n{len(results) - failures}/{len(results)} checks pass; logs in {args.logs}")
    return failures


if __name__ == "__main__":
    sys.exit(main())
