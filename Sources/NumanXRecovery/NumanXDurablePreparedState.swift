import Foundation

/// NumanX currently has no committed solver implementation in this repository. This file establishes
/// the recovery ABI that the native Apple-Silicon solver must implement rather than inventing physics
/// state that does not exist yet.
public struct NumanXPreparedStateManifest: Codable, Equatable, Sendable {
    public static let formatVersion: UInt32 = 1
    public var formatVersion: UInt32 = Self.formatVersion
    public var transactionFingerprint: UInt64
    public var baseGeneration: UInt64
    public var shadowGeneration: UInt64
    public var startTimeNanoseconds: UInt64
    public var endTimeNanoseconds: UInt64
    public var solverConfigurationFingerprint: UInt64
    public var topologyFingerprint: UInt64
    public var authoritativeStateSHA256: String
    public var gpuBufferImageSHA256: String
    public var gpuBufferImageBytes: UInt64

    public func validated() throws -> Self {
        guard formatVersion == Self.formatVersion, transactionFingerprint > 0,
              shadowGeneration == baseGeneration + 1, endTimeNanoseconds > startTimeNanoseconds,
              solverConfigurationFingerprint > 0, topologyFingerprint > 0,
              authoritativeStateSHA256.count == 64, gpuBufferImageSHA256.count == 64,
              authoritativeStateSHA256.allSatisfy({ $0.isHexDigit }),
              gpuBufferImageSHA256.allSatisfy({ $0.isHexDigit }), gpuBufferImageBytes > 0 else {
            throw NumanXRecoveryError.invalidManifest
        }
        return self
    }
}

public enum NumanXRecoveryDecision: String, Codable, Sendable { case prepared, commitDecided, committed, aborted }

/// Native solver requirements for crash-safe integration with NumiTissue/NumiBrain.
///
/// prepareDurably must not return until authoritative CPU metadata and every GPU buffer needed to
/// reconstruct the shadow generation are persisted and fsynced. decideCommit must be durable before
/// publication. publishPrepared must be idempotent for the same transaction. After commitDecided,
/// rollback is forbidden: recovery is roll-forward only.
public protocol NumanXDurablePreparedStateStore: Sendable {
    func prepareDurably(manifest: NumanXPreparedStateManifest, authoritativeState: Data,
                        gpuBufferImage: Data) async throws
    func decideCommit(transactionFingerprint: UInt64) async throws
    func decision(transactionFingerprint: UInt64) async throws -> NumanXRecoveryDecision
    func loadPrepared(transactionFingerprint: UInt64) async throws
        -> (manifest: NumanXPreparedStateManifest, authoritativeState: Data, gpuBufferImage: Data)
    func markCommitted(transactionFingerprint: UInt64) async throws
    func abortPrepared(transactionFingerprint: UInt64) async throws
}

public protocol NumanXRecoverableSolver: Sendable {
    /// Synchronize the accepted shadow generation after all Metal command buffers complete.
    func makeDurablePreparedImage(transactionFingerprint: UInt64) async throws
        -> (manifest: NumanXPreparedStateManifest, authoritativeState: Data, gpuBufferImage: Data)
    /// Recreate private Metal buffers and all indirect-dispatch/topology tables from exact bytes.
    func restorePreparedImage(manifest: NumanXPreparedStateManifest, authoritativeState: Data,
                              gpuBufferImage: Data) async throws
    /// Nonthrowing authoritative pointer/generation publication after validation. Must be idempotent.
    func publishRestoredPrepared(transactionFingerprint: UInt64) async
}

public enum NumanXRecoveryError: Error, Sendable {
    case invalidManifest
    case corruptImage
    case conflictingDecision
    case unsupportedUntilNativeSolverExists
}
