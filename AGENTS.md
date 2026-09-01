# AGENTS.md — RiffDB MLflow adapter

## Mission

Implement a standalone Python tracking-store plugin for pinned MLflow releases
using a compiled RiffDB application. All MLflow-specific behavior belongs in
this repository, never in RiffDB.

## Authoritative inputs

- MLflow `v3.15.1`, especially `mlflow.store.tracking.AbstractStore` and the
  upstream SQL tracking-store tests.
- This repository's `DESIGN.md` and `PLAN.md`.
- The accepted RiffDB specification and ADRs at the revision recorded in the
  generated application lock and dependency receipt.

If these inputs conflict, stop and request maintainer review. Do not weaken an
MLflow semantic or a RiffDB safety guarantee to make a test pass.

## Architecture boundaries

1. MLflow mutations use generated, compiled RiffDB commands only.
2. Reads use generated named RiffQL queries only. Do not add generic CRUD,
   dynamically assembled RiffQL, client-side filtering, or full-store scans.
3. One MLflow API mutation that is atomic in the SQL store remains one atomic
   compiled command, including bounded metric/parameter/tag batches.
4. Search syntax is parsed and normalized according to MLflow. Every accepted
   shape dispatches to a finite named-query family; unsupported shapes fail
   before contacting RiffDB.
5. Pagination is bounded and continuation tokens are opaque, filter-bound, and
   rejected on cross-query reuse.
6. MLflow metric values preserve their exact IEEE-754 payload and MLflow's
   latest-metric ordering semantics, including NaN and infinities.
7. Artifact bytes remain in an MLflow artifact repository. This adapter stores
   tracking metadata and never turns RiffDB into an artifact blob store.
8. Public exceptions map deliberately to MLflow error codes and never expose
   RiffDB internals, credentials, protected values, or symbolic plan details.

## Stop conditions

Pause for a generic RiffDB capability or DX review if implementation would
require:

- splitting an atomic MLflow batch or lifecycle transition;
- an unbounded query, client-side filtering, sorting, or cursor walking used to
  repair semantics;
- process-local time, IDs, ordering, locks, or leases for durable correctness;
- arbitrary transaction callbacks, raw query text, or direct storage access;
- target-language reimplementation of RiffDB transport guarantees; or
- MLflow-specific behavior, vocabulary, routes, or schemas in RiffDB.

## Completion checks

Before calling the adapter usable:

```bash
make check
make conformance
```

Unit fakes may supplement upstream tests but never replace real RiffDB
loopback coverage.

