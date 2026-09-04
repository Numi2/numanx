# NumanX

NumanX is the Apple-Silicon-native multiphysics solver boundary for the Numi suite.

The repository does not yet contain the production solver. It therefore does not claim a working
rigid/FEM/MPM/contact implementation or a recoverable GPU runtime. The first committed source defines
the durable prepared-state contract required by NumiBrain and NumiTissue so the eventual solver is
built around correct crash semantics from the beginning.

A native NumanX transaction must persist the accepted shadow generation—including authoritative
solver metadata, topology identity and all Metal buffers required for reconstruction—before it votes
prepared. A durable commit decision is irrevocable. Publication is idempotent roll-forward; rollback
is permitted only before the commit decision.

See `Sources/NumanXRecovery/NumanXDurablePreparedState.swift`.
