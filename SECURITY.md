# Security Policy

## Supported Versions

SpatialCrafter is a research codebase. Security fixes are provided for the
latest release on the default branch only.

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

Please **do not** report security issues through public GitHub issues, pull
requests, discussions, or social media.

Use GitHub's private vulnerability reporting instead:

1. Go to the [Security tab](https://github.com/fangchuan/SpatialCrafter/security)
   of the repository.
2. Click **Report a vulnerability** to open a private advisory.

If you cannot use GitHub advisories, contact the maintainers at
**<cfangac@connect.ust.hk>** .

Please include:

- The affected version or commit hash.
- A description of the issue and its potential impact.
- Steps to reproduce, ideally with a minimal proof of concept.
- Any suggested fix or mitigation.

### What to expect

- Initial acknowledgement within 5 business days.
- An assessment and, where applicable, a remediation plan communicated after
  triage.
- Credit in the release notes for the fix, unless you ask to remain anonymous.

## Scope

In scope are vulnerabilities in this repository's own code, in particular:

- Unsafe deserialization of model weights or other checkpoint files.
- Path traversal or arbitrary file read/write reachable from configuration
  files, CLI arguments, or dataset split CSVs.
- Injection, SSRF, or arbitrary code execution through remote model downloads
  (`huggingface-cli`, `modelscope`, or URLs passed to loaders).
- Command injection through `subprocess` invocations.
- Exposure of the optional Gradio demo service. The demo binds to
  `127.0.0.1` by default; if you expose it on a network interface, put it behind
  your own authentication and TLS termination.

## Out of Scope

- Vulnerabilities in third-party dependencies and vendored upstream projects,
  including TRELLIS, DiffSynth-Studio, Wan2.1 / WanVideo, Depth Anything V3,
  nvdiffrast, diffoctreerast, and the RaDe-GS rasterizer. Please report those
  upstream.
- Loading checkpoints from untrusted sources. Model weights are deserialized
  with PyTorch; several code paths use `weights_only=False` and can therefore
  execute arbitrary code embedded in the file. **Only load checkpoints you
  trust.** This is documented behaviour, not a vulnerability.
- Denial of service through resource exhaustion, such as oversized inputs that
  exhaust GPU memory. Inference cost scales with input resolution and frame
  count by design.
- Issues that require an attacker to already control the host, the Python
  environment, or files the user voluntarily supplies.
- The absence of authentication, authorization, or multi-tenancy. This project
  is not designed to be run as a shared public service.

## Data and Privacy

The repository ships no user data collection, telemetry, or phone-home
behaviour. Inference reads only the inputs you provide, and model weights are
downloaded from the sources named in the README.
