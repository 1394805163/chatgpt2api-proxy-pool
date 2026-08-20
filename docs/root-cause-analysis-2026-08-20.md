# Image generation timeout root-cause analysis

## Scope and evidence

The pre-restart production log is retained at
`/tmp/chatgpt2api-pre-restart.log` on the host. Its SHA-256 is
`37691f92f6ac1571395382203b4582ed851010689929e0094c94639b42fda4ee`.
It covers 2026-08-19T13:10:50Z through 2026-08-19T17:36:12Z.

Observed facts:

- 284 image entry requests.
- 138 `image_stream_conn_timeout_retry` events: 87 first retries, 44 second
  retries, and 7 third retries.
- 134 of those retries followed a curl timeout after response bytes had
  already been received. The parsed average was about 761 KB and the maximum
  was 2.79 MB.
- 744 successful `/api/image-tasks` polls and 67 successful `/api/logs`
  requests added load during the incident.
- There was no `OOM`, `Killed`, or abnormal process termination record. The
  service ended with a normal shutdown sequence.

## Conclusion 1: false SSE timeout caused duplicate upstream submissions

`curl_cffi` converts a numeric timeout on a streaming request into a libcurl
low-speed timeout. Image SSE can remain silent while the upstream renderer is
working. The old implementation treated a resulting `curl(28)` as a failed
submission even after it had received SSE bytes, and retried the upstream POST.
The upstream image job may already exist at that point, so a retry can create a
second generation and consume another account/connection slot.

### Minimal correction

1. Disable `LOW_SPEED_LIMIT` and `LOW_SPEED_TIME` only while establishing the
   image SSE request. The existing task deadline remains the authoritative
   total timeout.
2. Retry at most once, and only for a connection failure before an HTTP body is
   received. A zero-byte timeout is the only body-timeout retry candidate.
3. When an SSE stream has already yielded a `conversation_id`, continue by
   polling that conversation instead of posting a new generation request.
4. When polling reaches its deadline, return the original `conversation_id`.
   Do not switch account and re-submit the generation.

Implemented in:

- `services/openai_backend_api.py`
- `services/protocol/conversation.py`

## Conclusion 2: local admission control and UI reads amplified the backlog

The incident does not show a user bypassing the configured concurrent request
limit. It shows the service counting work by request rather than image count,
releasing capacity before a timed-out worker actually exited, and accepting
repeated client submits without a durable request identity. The high-frequency
task and log reads added secondary pressure, especially because log listing was
synchronous and unbounded for date-filtered views.

### Minimal correction

1. Count a request with `n` images as `n` concurrent units.
2. Keep its unit lease until the worker thread has actually exited, even if its
   externally visible task has timed out.
3. Bound asynchronous queue capacity globally and per owner.
4. Accept `Idempotency-Key` or `client_task_id`; coalesce matching requests,
   reject key/payload conflicts, and cache a bounded non-stream response.
5. Page `/api/logs` and perform log listing in a worker thread.

Implemented in:

- `services/image_concurrency.py`
- `services/image_task_service.py`
- `services/image_idempotency.py`
- `api/ai.py`
- `api/image_tasks.py`
- `api/system.py`
- `services/log_service.py`

## Validation

The focused regression suite passed in an isolated data directory:

```text
78 passed in 20.33s
```

The tests cover partial-SSE timeout resume polling, zero-byte timeout retry,
poll-timeout no-resubmission, SSE curl options, idempotency replay/conflict,
image-count concurrency, queue capacity, and log pagination. A candidate image
was also started without access to production data; `/health` and `/version`
returned HTTP 200.

## Recommended deployment sequence

1. Deploy this minimal correction first and monitor `curl(28)`,
   `image_stream_timeout_no_resubmit`, task queue depth, image latency, and
   idempotency replay counts.
2. Require the frontend/client to send a stable `Idempotency-Key` for every
   non-stream image request. Previous incident logs showed most requests had
   no client task identifier, so server-side memory-only de-duplication is
   limited to one process lifetime.
3. For multiple application replicas or restart-resilient recovery, move task
   state, idempotency records, and concurrency leases to Redis; run image jobs
   through dedicated workers and use `202 Accepted` plus `task_id` for the
   client protocol. This is the durable architecture, not a prerequisite for
   the immediate fix.

## Container publishing boundary

The distributed image intentionally excludes the workstation `config.json`.
It starts only when a deployment mounts runtime configuration or supplies
`CHATGPT2API_AUTH_KEY`. Production Compose already mounts `./config.json` at
`/app/config.json`; no runtime credential needs to be baked into the image.
