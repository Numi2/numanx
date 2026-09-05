import Foundation

/// Executes persistence/recovery around a native solver owner; it never simulates missing physics.
/// Exclusive ownership of solver and store is required. Only the joint transaction manager may
/// call commit(). Device/biological stimulation is never a rollback-capable participant here.
public actor NumanXRecoveryCoordinator {
    private let store: any NumanXDurablePreparedStateStore
    private let solver: any NumanXVerifiedRecoverableSolver
    private var busy = false
    private var fencedTransaction: UInt64?

    public init(store: any NumanXDurablePreparedStateStore, solver: any NumanXVerifiedRecoverableSolver) {
        self.store = store; self.solver = solver
    }

    public func prepare(transactionFingerprint: UInt64) async throws -> NumanXPreparedStateManifest {
        guard !busy, fencedTransaction == nil else { throw NumanXRecoveryError.busy }
        busy = true; defer { busy = false }
        try Task.checkCancellation()
        let image = try await solver.makeDurablePreparedImage(transactionFingerprint: transactionFingerprint)
        guard image.manifest.transactionFingerprint == transactionFingerprint else {
            throw NumanXRecoveryError.invalidManifest
        }
        try image.manifest.verify(authoritativeState: image.authoritativeState, gpuBufferImage: image.gpuBufferImage)
        fencedTransaction = transactionFingerprint
        // On ambiguous persistence failure, do not run another transaction against the same owner.
        try await store.prepareDurably(manifest: image.manifest,
            authoritativeState: image.authoritativeState, gpuBufferImage: image.gpuBufferImage)
        return image.manifest
    }

    /// A returned receipt means the native owner published the expected generation and the store
    /// durably recorded completion. On any failure the transaction remains fenced for recovery.
    public func commit(transactionFingerprint: UInt64) async throws -> NumanXPublicationReceipt {
        guard !busy, fencedTransaction == transactionFingerprint else { throw NumanXRecoveryError.busy }
        busy = true; defer { busy = false }
        try Task.checkCancellation()
        try await store.decideCommit(transactionFingerprint: transactionFingerprint)
        // Cancellation cannot turn a durable commit into an abort; complete or remain in doubt.
        return try await rollForward(transactionFingerprint)
    }

    /// Used with a newly reconstructed native owner after process death. A merely prepared record
    /// remains undecided and requires the SAME external transaction manager's decision.
    public func recover(transactionFingerprint: UInt64) async throws -> NumanXPublicationReceipt {
        guard !busy, fencedTransaction == nil || fencedTransaction == transactionFingerprint else {
            throw NumanXRecoveryError.busy
        }
        busy = true; defer { busy = false }
        fencedTransaction = transactionFingerprint
        switch try await store.decision(transactionFingerprint: transactionFingerprint) {
        case .prepared: throw NumanXRecoveryError.undecided
        case .aborted: throw NumanXRecoveryError.conflictingDecision
        case .commitDecided, .committed: return try await rollForward(transactionFingerprint)
        }
    }

    public func requiresRecovery() -> Bool { fencedTransaction != nil }

    private func rollForward(_ transaction: UInt64) async throws -> NumanXPublicationReceipt {
        let image = try await store.loadPrepared(transactionFingerprint: transaction)
        try image.manifest.verify(authoritativeState: image.authoritativeState, gpuBufferImage: image.gpuBufferImage)
        let expected = NumanXPublicationReceipt(manifest: image.manifest)
        if let published = try await solver.committedPublicationReceipt() {
            if published == expected {
                try await store.markCommitted(transactionFingerprint: transaction)
                fencedTransaction = nil
                return published
            }
            // Never overwrite a later or unrelated generation with an old prepared image.
            guard published.generation == image.manifest.baseGeneration,
                  published.timeNanoseconds == image.manifest.startTimeNanoseconds,
                  published.solverConfigurationFingerprint == image.manifest.solverConfigurationFingerprint,
                  published.topologyFingerprint == image.manifest.topologyFingerprint else {
                throw NumanXRecoveryError.publicationMismatch
            }
        }
        try await solver.restorePreparedImage(manifest: image.manifest,
            authoritativeState: image.authoritativeState, gpuBufferImage: image.gpuBufferImage)
        await solver.publishRestoredPrepared(transactionFingerprint: transaction)
        guard let published = try await solver.committedPublicationReceipt(), published == expected else {
            throw NumanXRecoveryError.publicationMismatch
        }
        try await store.markCommitted(transactionFingerprint: transaction)
        fencedTransaction = nil
        return published
    }
}
