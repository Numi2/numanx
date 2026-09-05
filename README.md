# NumanX

Apple-native monolithic multiphysics solver for the Numi suite.

This repository now contains an executable **solver core**, a Metal 4 hot-path target, and crash-safe
prepared-state recovery support. It is still under active integration: the physics-specific rigid,
ABA, FEM, MPM, DER, transport and native scene adapters are not all implemented here yet, and this
development session did not run Swift builds, Metal compilation, numerical validation or benchmarks.

## Numerical authority

`NumanXCore` implements the authoritative nonlinear solve contract:

- A single coupled state layout for primal variables, pressure, equality multipliers and exact
  three-component circular-Coulomb contact multipliers.
- Exact Euclidean projection onto the Lorentz cone and the natural contact residual
  `lambda_hat - Proj_L(lambda_hat - rho*u_hat)`, including compliance, restitution and stabilization.
- Matrix-free flexible restarted GMRES with explicit preconditioned basis storage, modified
  Gram-Schmidt, conditional reorthogonalization and numerical-breakdown rejection.
- Globally safeguarded semismooth Newton with safe-step certificates, bounded Armijo trials,
  deterministic training budgets and a higher-accuracy adaptive profile.
- A monolithic `NXCoupledSystem` in which rigid/articulated/FEM/MPM/rod/transport/contact terms add
  to one residual and one Jacobian-vector product instead of being committed by separate solvers.
- Tensor-Schur patch descriptors and cohorts, overlapping additive corrections, mixed-precision
  advisory solves with FP32 promotion, and a deterministic dense micropatch reference backend.
- `NXAuthoritativeSolver`: base generation -> unpublished shadow -> exact-token commit/abort, with
  deterministic substep fallback. Failed candidates never mutate the committed state.

The authoritative state and residual remain FP32. FP16/BF16 are permitted only inside approximate
local preconditioner work and must return a finite FP32 correction before reaching Krylov state.

## Metal 4 hot path

`NumanXMetal` contains GPU kernels for:

- AXPBY and vector scaling.
- Finite-value failure detection.
- Deterministic hierarchical dot-product reductions.
- Exact circular-Coulomb natural residuals and cone/complementarity diagnostics.

The runtime uses the Metal 4 ownership model already used by NumiBrain: one `MTL4CommandQueue`, a
reusable `MTL4CommandBuffer`, a resettable `MTL4CommandAllocator`, explicit residency and an outer
owner-controlled completion boundary. It deliberately does not create a hidden queue per solve or
perform authoritative host readbacks in the Newton/Krylov hot loop.

Metal 4 compute encoders unify compute and copy operations in one pass and bind resources through
argument tables. Production patch kernels should therefore cohort topology/material/shape/precision
classes once, retain their allocations, and launch indirect/batched work from the same GPU timeline.

Apple references:

- https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api
- https://developer.apple.com/documentation/metal/mtl4computecommandencoder
- https://developer.apple.com/documentation/metal/synchronizing-passes-with-consumer-barriers
- https://developer.apple.com/documentation/metal/running-a-machine-learning-model-on-the-gpu-timeline

## Recovery

`NumanXRecovery` contains:

- A checked prepared-state manifest binding transaction, generations, time, topology, configuration,
  authoritative metadata and exact GPU-image bytes.
- `FileNumanXPreparedStateStore`: exclusive descriptor-relative storage with checksum verification,
  immutable prepare/commit/abort records, full file synchronization and directory synchronization.
- `NumanXRecoveryCoordinator`: prepare-before-vote, decision-before-publication and idempotent
  roll-forward through a native owner that supplies an actual publication receipt.
- `numanx-recovery inspect`, which verifies persisted bytes and reports the durable decision without
  creating a decision or publishing a generation.

No GPU virtual address is a persistent identity. A native solver recovery image must contain or
content-address every logical state needed to rebuild private buffers, argument tables, indirect
commands, topology, contact histories, transport state, RNG and pending events.

## Build on the target Mac

```sh
swift build
swift test --filter NumanXCoreTests
swift test --filter NumanXRecoveryTests
swift run numanx-recovery status
```

The committed tests cover Lorentz-cone geometry, Coulomb residual behavior, nonsymmetric matrix-free
FGMRES, overlapping Tensor-Schur patches, semismooth Newton, state layout, shadow publication/abort,
prepared-image integrity and recovery decision irreversibility. They are source tests only until run
on the target toolchain.

## Required production work

The next implementation layer is physics-specific rather than another abstraction layer:

1. Rigid 6x6 inertia and articulated multi-RHS ABA contributions plus their patch inverses.
2. Implicit corotated/hyperelastic FEM residual/Jv and positive tangent surrogate patches.
3. MPM particle-grid transfer, constitutive updates and multigrid preconditioning.
4. DER rods/cables/sutures with banded or cyclic-reduction local solves.
5. Pressure/incompressibility, attachment and transport blocks.
6. Broadphase/CCD, exact Coulomb contact generation and persistent friction history.
7. GPU-side safe-step reduction for CCD, det(F), volume and rod-length admissibility.
8. Native scene serialization sufficient for `NumanXVerifiedRecoverableSolver`.
9. NumiBrain/NumiTissue joint-root wiring and retained crash/fault qualification on Apple Silicon.

The core is designed so these blocks add residuals/Jv/preconditioner patches to one authoritative
Newton solve. They must not create separate publication domains that can drift from the body root.

A physical actuator or living-culture stimulus remains outside the reversible simulation transaction.
Local recovery files are durable state, not distributed consensus, signatures or safety approval.
