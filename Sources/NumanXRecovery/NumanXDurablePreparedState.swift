import Foundation
import CryptoKit

/// Actual solver-owned metadata and GPU bytes are supplied by the native runtime. A manifest is
/// not a substitute for that runtime and does not establish that all numerical state was captured.
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

    public init(transactionFingerprint: UInt64, baseGeneration: UInt64, shadowGeneration: UInt64,
                startTimeNanoseconds: UInt64, endTimeNanoseconds: UInt64,
                solverConfigurationFingerprint: UInt64, topologyFingerprint: UInt64,
                authoritativeStateSHA256: String, gpuBufferImageSHA256: String, gpuBufferImageBytes: UInt64) {
        self.transactionFingerprint = transactionFingerprint; self.baseGeneration = baseGeneration
        self.shadowGeneration = shadowGeneration; self.startTimeNanoseconds = startTimeNanoseconds
        self.endTimeNanoseconds = endTimeNanoseconds; self.solverConfigurationFingerprint = solverConfigurationFingerprint
        self.topologyFingerprint = topologyFingerprint; self.authoritativeStateSHA256 = authoritativeStateSHA256
        self.gpuBufferImageSHA256 = gpuBufferImageSHA256; self.gpuBufferImageBytes = gpuBufferImageBytes
    }
    public func validated() throws -> Self {
        let generation = baseGeneration.addingReportingOverflow(1)
        guard formatVersion == Self.formatVersion, transactionFingerprint > 0,
              !generation.overflow, shadowGeneration == generation.partialValue,
              endTimeNanoseconds > startTimeNanoseconds, solverConfigurationFingerprint > 0, topologyFingerprint > 0,
              NumanXRecoveryHash.isSHA256(authoritativeStateSHA256),
              NumanXRecoveryHash.isSHA256(gpuBufferImageSHA256), gpuBufferImageBytes > 0 else {
            throw NumanXRecoveryError.invalidManifest
        }
        return self
    }
    public func verify(authoritativeState: Data, gpuBufferImage: Data) throws {
        _ = try validated()
        guard !authoritativeState.isEmpty, UInt64(gpuBufferImage.count) == gpuBufferImageBytes,
              NumanXRecoveryHash.sha256(authoritativeState) == authoritativeStateSHA256,
              NumanXRecoveryHash.sha256(gpuBufferImage) == gpuBufferImageSHA256 else {
            throw NumanXRecoveryError.corruptImage
        }
    }
}

public enum NumanXRecoveryDecision: String, Codable, Sendable {
    case prepared, commitDecided, committed, aborted
}

public protocol NumanXDurablePreparedStateStore: Sendable {
    func prepareDurably(manifest: NumanXPreparedStateManifest, authoritativeState: Data, gpuBufferImage: Data) async throws
    func decideCommit(transactionFingerprint: UInt64) async throws
    func decision(transactionFingerprint: UInt64) async throws -> NumanXRecoveryDecision
    func loadPrepared(transactionFingerprint: UInt64) async throws
        -> (manifest: NumanXPreparedStateManifest, authoritativeState: Data, gpuBufferImage: Data)
    func markCommitted(transactionFingerprint: UInt64) async throws
    func abortPrepared(transactionFingerprint: UInt64) async throws
}

public protocol NumanXRecoverableSolver: Sendable {
    /// Synchronize all native producers, including physics, contact history, transport, RNG and
    /// pending events. Never persist Metal virtual addresses as restart-stable references.
    func makeDurablePreparedImage(transactionFingerprint: UInt64) async throws
        -> (manifest: NumanXPreparedStateManifest, authoritativeState: Data, gpuBufferImage: Data)
    func restorePreparedImage(manifest: NumanXPreparedStateManifest, authoritativeState: Data, gpuBufferImage: Data) async throws
    /// Idempotent authoritative publication. Call only after a durable joint commit decision.
    func publishRestoredPrepared(transactionFingerprint: UInt64) async
}

public struct NumanXPublicationReceipt: Codable, Equatable, Sendable {
    public var transactionFingerprint: UInt64
    public var generation: UInt64
    public var timeNanoseconds: UInt64
    public var solverConfigurationFingerprint: UInt64
    public var topologyFingerprint: UInt64
    public var preparedStateSHA256: String
    public var preparedGPUImageSHA256: String
    public init(manifest: NumanXPreparedStateManifest) {
        transactionFingerprint = manifest.transactionFingerprint; generation = manifest.shadowGeneration
        timeNanoseconds = manifest.endTimeNanoseconds
        solverConfigurationFingerprint = manifest.solverConfigurationFingerprint
        topologyFingerprint = manifest.topologyFingerprint
        preparedStateSHA256 = manifest.authoritativeStateSHA256
        preparedGPUImageSHA256 = manifest.gpuBufferImageSHA256
    }
}

/// A nonthrowing publish function by itself cannot prove that the expected state became visible.
/// Native owners implementing recovery must return their actual published identity after completion.
public protocol NumanXVerifiedRecoverableSolver: NumanXRecoverableSolver {
    func committedPublicationReceipt() async throws -> NumanXPublicationReceipt?
}

public enum NumanXRecoveryError: Error, Sendable {
    case invalidManifest, corruptImage, conflictingDecision, unsupportedUntilNativeSolverExists
    case invalidStore(String), capacity, writerBusy, poisoned, undecided, publicationMismatch, busy
}

public enum NumanXRecoveryHash {
    public static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    public static func isSHA256(_ text: String) -> Bool {
        text.utf8.count == 64 && text.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }
    static func encode<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(value)
    }
}
