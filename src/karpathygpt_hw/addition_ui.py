"""Local browser interface for the trained 1,364-digit addition model."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import ast
import json
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import mlx.core as mx

from karpathygpt_hw.addition_inference import generate_addition
from karpathygpt_hw.kda_mlx import KDAAdditionModel
from karpathygpt_hw.long_addition import TOKENS

MAX_OPERAND_DIGITS = 1_364
MAX_EXPRESSION_CHARACTERS = 12_000


class ExpressionError(ValueError):
    """Raised when the input is not a safe, supported addition expression."""


def _bounded_integer(value: int) -> int:
    if len(str(abs(value))) > MAX_OPERAND_DIGITS + 1:
        raise ExpressionError("an intermediate value is larger than 10^1364")
    return value


def _evaluate_operand(node: ast.AST) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        if isinstance(node.value, bool):
            raise ExpressionError("boolean values are not supported")
        return _bounded_integer(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_operand(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if not isinstance(node, ast.BinOp):
        raise ExpressionError("use integers and the operators +, -, *, and ^")

    left = _evaluate_operand(node.left)
    right = _evaluate_operand(node.right)
    if isinstance(node.op, ast.Add):
        return _bounded_integer(left + right)
    if isinstance(node.op, ast.Sub):
        return _bounded_integer(left - right)
    if isinstance(node.op, ast.Mult):
        return _bounded_integer(left * right)
    if isinstance(node.op, ast.Pow):
        if left != 10 or not 0 <= right <= MAX_OPERAND_DIGITS:
            raise ExpressionError("powers are limited to 10^n with 0 <= n <= 1364")
        return _bounded_integer(10**right)
    raise ExpressionError("use integers and the operators +, -, *, and ^")


def parse_addition_expression(expression: str) -> tuple[str, str]:
    """Safely parse one top-level addition into two decimal operands."""

    expression = expression.strip()
    if not expression:
        raise ExpressionError("enter an addition expression")
    if len(expression) > MAX_EXPRESSION_CHARACTERS:
        raise ExpressionError("the expression is too long")
    normalized = expression.replace("^", "**")
    try:
        tree = ast.parse(normalized, mode="eval")
    except (SyntaxError, ValueError) as error:
        raise ExpressionError("that expression could not be parsed") from error
    if sum(1 for _ in ast.walk(tree)) > 100:
        raise ExpressionError("the expression is too complex")
    if not isinstance(tree.body, ast.BinOp) or not isinstance(tree.body.op, ast.Add):
        raise ExpressionError("enter exactly two operands joined by a top-level +")

    operands = (_evaluate_operand(tree.body.left), _evaluate_operand(tree.body.right))
    if any(value < 0 for value in operands):
        raise ExpressionError("operands must be non-negative")
    operand_text = tuple(str(value) for value in operands)
    if any(len(value) > MAX_OPERAND_DIGITS for value in operand_text):
        raise ExpressionError("each operand must contain at most 1,364 digits")
    return operand_text  # type: ignore[return-value]


PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KDA Long Addition</title>
  <style>
    :root { color-scheme: dark; --ink:#f5f7ff; --muted:#a7afc3; --line:#29314a;
      --card:#151a2a; --accent:#8b7cff; --good:#57d69b; --bad:#ff7184; }
    * { box-sizing:border-box; }
    body { margin:0; min-height:100vh; color:var(--ink); font:16px/1.5 Inter,ui-sans-serif,system-ui;
      background:radial-gradient(circle at 15% 5%,#26205b 0,transparent 33%),#0b0e18; }
    main { width:min(980px,calc(100% - 32px)); margin:0 auto; padding:56px 0 80px; }
    .eyebrow { color:#bdb5ff; font-size:13px; font-weight:750; letter-spacing:.13em; text-transform:uppercase; }
    h1 { margin:9px 0 10px; font-size:clamp(34px,6vw,62px); line-height:1.02; letter-spacing:-.045em; }
    .lead { max-width:720px; margin:0 0 30px; color:var(--muted); font-size:18px; }
    .panel { padding:22px; border:1px solid var(--line); border-radius:20px; background:rgba(21,26,42,.92);
      box-shadow:0 22px 60px rgba(0,0,0,.32); backdrop-filter:blur(14px); }
    label { display:block; margin-bottom:9px; font-weight:700; }
    textarea { width:100%; min-height:108px; resize:vertical; padding:16px; color:var(--ink); background:#0e1220;
      border:1px solid #343d5d; border-radius:13px; font:15px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace; }
    textarea:focus { outline:2px solid var(--accent); border-color:transparent; }
    .actions { display:flex; flex-wrap:wrap; gap:10px; margin-top:14px; }
    button { border:0; border-radius:12px; padding:11px 16px; color:white; background:#272e46; font-weight:750; cursor:pointer; }
    button.primary { background:linear-gradient(135deg,#7764f8,#9a72ef); box-shadow:0 8px 24px rgba(119,100,248,.3); }
    button:disabled { opacity:.55; cursor:wait; }
    .hint { margin:12px 0 0; color:var(--muted); font-size:13px; }
    #notice { display:none; margin-top:18px; padding:13px 15px; border-radius:12px; background:#20263a; color:var(--muted); }
    #notice.error { display:block; color:#ffd8de; background:#481f2b; }
    #results { display:none; margin-top:18px; gap:14px; }
    .verdict { display:flex; align-items:center; justify-content:space-between; gap:16px; padding:16px 18px;
      border:1px solid rgba(87,214,155,.35); border-radius:15px; background:rgba(35,92,70,.3); }
    .verdict.bad { border-color:rgba(255,113,132,.4); background:rgba(112,37,50,.35); }
    .verdict strong { font-size:20px; }
    .meta { color:var(--muted); font-size:13px; text-align:right; }
    .result-card { position:relative; padding:17px; border:1px solid var(--line); border-radius:15px; background:#0e1220; }
    .result-card h2 { margin:0 0 10px; font-size:14px; color:var(--muted); }
    pre { max-height:260px; overflow:auto; margin:0; padding:0 48px 0 0; white-space:pre-wrap; overflow-wrap:anywhere;
      color:#e8eaff; font:14px/1.65 ui-monospace,SFMono-Regular,Menlo,monospace; }
    .copy { position:absolute; top:12px; right:12px; padding:7px 10px; font-size:12px; }
    footer { margin-top:17px; color:#778098; font-size:12px; }
    @media(max-width:600px){ main{padding-top:34px}.panel{padding:16px}.verdict{align-items:flex-start;flex-direction:column}.meta{text-align:left} }
  </style>
</head>
<body><main>
  <div class="eyebrow">Fused Metal · Kimi Delta Attention</div>
  <h1>Add very, very large numbers.</h1>
  <p class="lead">Enter two non-negative operands—directly or as powers of ten. The trained model generates the answer, then an independent exact-arithmetic verifier checks every digit.</p>
  <section class="panel">
    <label for="expression">Addition expression</label>
    <textarea id="expression" spellcheck="false">(10^1364 - 1) + (10^1364 - 1)</textarea>
    <div class="actions">
      <button class="primary" id="run">Run the model</button>
      <button id="example">Restore 1,364-digit example</button>
      <button id="small">Use a smaller example</button>
    </div>
    <p class="hint">Supported: decimal integers, parentheses, +, −, × using *, and powers written as 10^n. Maximum: 1,364 digits per operand.</p>
    <div id="notice"></div>
    <div id="results">
      <div class="verdict" id="verdict"><strong id="verdictText"></strong><span class="meta" id="meta"></span></div>
      <div class="result-card"><h2>Model answer</h2><button class="copy" data-copy="modelAnswer">Copy</button><pre id="modelAnswer"></pre></div>
      <div class="result-card"><h2>Independent verifier answer</h2><button class="copy" data-copy="expectedAnswer">Copy</button><pre id="expectedAnswer"></pre></div>
    </div>
  </section>
  <footer>The verifier never supplies digits to the model. Cached recurrent decoding preserves strict causal generation.</footer>
</main>
<script>
const $ = id => document.getElementById(id);
const example = '(10^1364 - 1) + (10^1364 - 1)';
$('example').onclick = () => $('expression').value = example;
$('small').onclick = () => $('expression').value = '12345678901234567890 + 98765432109876543210';
document.querySelectorAll('.copy').forEach(button => button.onclick = async () => {
  await navigator.clipboard.writeText($(button.dataset.copy).textContent);
  const old = button.textContent; button.textContent = 'Copied'; setTimeout(() => button.textContent = old, 900);
});
$('run').onclick = async () => {
  const button = $('run'); button.disabled = true; $('results').style.display = 'none';
  $('notice').className = ''; $('notice').style.display = 'block';
  $('notice').textContent = 'Loading the checkpoint and generating the answer…';
  try {
    const response = await fetch('/api/add', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({expression:$('expression').value})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'The request failed.');
    $('modelAnswer').textContent = data.model_answer;
    $('expectedAnswer').textContent = data.expected_answer;
    const verdict = $('verdict'); verdict.className = data.verified ? 'verdict' : 'verdict bad';
    $('verdictText').textContent = data.verified ? '✓ Every digit is accurate' : '✕ The verifier found a mismatch';
    $('meta').textContent = `${data.operand_digits.toLocaleString()}-digit width · ${data.output_digits.toLocaleString()} output digits · ${data.model_seconds.toFixed(2)} s model time`;
    $('notice').style.display = 'none'; $('results').style.display = 'grid';
  } catch (error) {
    $('notice').className = 'error'; $('notice').textContent = error.message;
  } finally { button.disabled = false; }
};
</script></body></html>"""


class AdditionApplication:
    def __init__(self, checkpoint: Path):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        self.model = KDAAdditionModel(
            vocab_size=len(TOKENS),
            dims=64,
            num_heads=4,
            num_layers=4,
            use_fused_kernel=True,
            position_encoding="addition_column_context",
        )
        self.model.load_weights(str(checkpoint))
        mx.eval(self.model.parameters())
        self.model.eval()
        self.lock = threading.Lock()

    def add(self, expression: str) -> dict[str, Any]:
        operand_a, operand_b = parse_addition_expression(expression)
        started = time.perf_counter()
        with self.lock:
            generated = generate_addition(self.model, operand_a, operand_b)
        expected = str(int(operand_a) + int(operand_b))
        return {
            "model_answer": generated.answer,
            "expected_answer": expected,
            "verified": generated.answer == expected,
            "trailing_space_correct": generated.trailing_space_correct,
            "operand_digits": max(len(operand_a), len(operand_b)),
            "output_digits": len(generated.answer),
            "model_seconds": generated.elapsed_seconds,
            "total_seconds": time.perf_counter() - started,
        }


def _handler(application: AdditionApplication) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "KDAAddition/1.0"

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self._send(status, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                self._send(HTTPStatus.OK, PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/health":
                self._json(HTTPStatus.OK, {"ok": True})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/add":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 20_000:
                    raise ExpressionError("request body is empty or too large")
                payload = json.loads(self.rfile.read(length))
                expression = payload.get("expression")
                if not isinstance(expression, str):
                    raise ExpressionError("expression must be text")
                result = application.add(expression)
            except (ExpressionError, json.JSONDecodeError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            except Exception as error:  # pragma: no cover - server safety net
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"model inference failed: {error}"},
                )
                return
            self._json(HTTPStatus.OK, result)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    default_checkpoint = (
        project_root / "runs/kda-aligned-context-1364-final/checkpoint-00003200/model.safetensors"
    )
    parser = argparse.ArgumentParser(description="Open the local KDA addition interface.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument("--open", action="store_true", dest="open_browser")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    application = AdditionApplication(args.checkpoint)
    server = ThreadingHTTPServer((args.host, args.port), _handler(application))
    url = f"http://{args.host}:{args.port}"
    print(f"KDA addition interface ready at {url}", flush=True)
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
