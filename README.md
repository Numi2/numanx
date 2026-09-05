# NumanX

Apple-native multiphysics runtime direction for the Numi suite.

This repository currently implements **native prepared-state persistence and recovery support**.
It does **not** yet contain the complete NumanX multiphysics solver. The recovery code is source
implementation, not a statement that Swift compilation, numerical validation or Apple GPU execution
has passed.

## Implemented here

`NumanXRecovery` contains:

- A checked prepared-state manifest binding transaction, generations, time, topology, configuration,
  authoritative metadata and exact GPU-image bytes.
- `FileNumanXPreparedStateStore`: an exclusive, descriptor-relative, checksum-verified persistent
  store with immutable prepare/commit/abort records, file/full synchronization and directory sync.
- `NumanXRecoveryCoordinator`: prepare-before-vote, decision-before-publication, and idempotent
  roll-forward through a native owner that supplies an actual publication receipt.
- A bounded `numanx-recovery inspect` executable and regression test source.

The persistence implementation stores real supplied metadata/GPU bytes, not merely a JSON digest.
No GPU pointer is a valid persistent identity. The native solver must serialize the complete
logical state and recreate private buffers, indirect arguments and residency after restart.

## Build and inspect on the target machine

```sh
swift build
swift test --filter NumanXRecoveryTests
swift run numanx-recovery status
swift run numanx-recovery inspect /path/to/existing-store 2a
```

The inspection command checks stored byte counts and SHA-256 values and reads the durable decision.
It requires the exclusive store lock. It never creates a commit decision, restores a solver or moves
physical hardware. No builds or tests were executed during this source-development increment.

## Native-owner integration

Implement `NumanXVerifiedRecoverableSolver` in the authoritative solver module. Its capture method
must wait for successful completion of every producer and include all restart-relevant state:
positions, velocities, rotations, material/internal variables, pressures, contact/friction histories,
transport/metabolic fields, constraints, topology, allocators, queued events, RNG and model identity.
Immutable assets may be referenced only when their exact bytes remain available and verified.

`makeDurablePreparedImage` returns the native manifest and bytes. `restorePreparedImage` validates and
reconstructs an unpublished candidate. `publishRestoredPrepared` performs idempotent publication, and
`committedPublicationReceipt` must report the actual native published identity, not echo an input
manifest before GPU completion. The coordinator rejects unrelated or later generations instead of
rewinding them to a stale candidate.

Only the joint NumiBrain/NumiTissue/NumanX transaction manager may authorize `commit`. A prepared file
without a decision stays undecided. After a commit decision, recovery completes publication; it
never converts the transaction to abort. A persisted `committed` marker does not imply that a newly
launched process has recreated its GPU buffers, so recovery verifies or restores the native owner
before declaring the runtime available.

## What is not claimed

No native NumanX numerical implementation was found in this standalone repository. NumiLab has a
separate Matter runtime and accepted-snapshot exporter; the inspected exporter does not establish a
complete unpublished NumanX/MyoSim image. This package neither fabricates that model nor substitutes
a toy solver while claiming full-suite integration.

The native state adapter, full owner-transaction insertion, authentic joint decision transport and
Apple Silicon crash/fault campaign remain integration work. Local durable records are not distributed
consensus or signatures. A physical actuator or living-culture stimulus is not rollback-capable and
must remain outside the reversible simulation transaction.
