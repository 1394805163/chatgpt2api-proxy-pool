# Running Image Baseline

This branch records the application content actually served by the production
container at export time.

- Image tag: `chatgpt2api:quota-units-local`
- Image digest: `sha256:3a914ad202704bca49d5cf7a84a527e13aef8270e85f98ba331a94f0813af58e`
- Export location: `runtime_snapshot/app/`
- File checksums: `RUNTIME_FILE_HASHES.sha256`

The snapshot includes the runtime Python application, tests, and the exact
compiled frontend under `runtime_snapshot/app/web_dist/`.

It deliberately excludes mounted runtime configuration and data:
`config.json`, `.env`, `data/`, virtual environments, and caches.
The production image does not contain `web/src/`, the Dockerfile, or
dependency lockfiles. The compiled `web_dist/` artifact is therefore the
authoritative record of the frontend that was actually served.

Use this branch as a comparison baseline only. Continue feature work from the
`codex/vps-snapshot-20260818` development branch after reconciling the
differences.
