import Foundation

public struct NXTransactionToken: Codable, Equatable, Sendable {
    public var environmentIdentifier: UInt32
    public var baseGeneration: UInt64
    public var shadowGeneration: UInt64
    public var startTimeNanoseconds: UInt64
    public var targetTimeNanoseconds: UInt64
    public var configurationFingerprint: UInt64
    public var topologyFingerprint: UInt64
    public var fingerprint: UInt64

    public init(environmentIdentifier: UInt32, baseGeneration: UInt64,
                startTimeNanoseconds: UInt64, targetTimeNanoseconds: UInt64,
                configurationFingerprint: UInt64, topologyFingerprint: UInt64) throws {
        let next = baseGeneration.addingReportingOverflow(1)
        guard !next.overflow, targetTimeNanoseconds > startTimeNanoseconds,
              configurationFingerprint > 0, topologyFingerprint > 0 else {
            throw NXError.invalidConfiguration("transaction token")
        }
        self.environmentIdentifier = environmentIdentifier; self.baseGeneration = baseGeneration
        shadowGeneration = next.partialValue; self.startTimeNanoseconds = startTimeNanoseconds
        self.targetTimeNanoseconds = targetTimeNanoseconds; self.configurationFingerprint = configurationFingerprint
        self.topologyFingerprint = topologyFingerprint
        fingerprint = Self.hash(environmentIdentifier: environmentIdentifier, baseGeneration: baseGeneration,
            shadowGeneration: next.partialValue, start: startTimeNanoseconds, target: targetTimeNanoseconds,
            configuration: configurationFingerprint, topology: topologyFingerprint)
    }

    public func validate() throws {
        let next = baseGeneration.addingReportingOverflow(1)
        guard !next.overflow, shadowGeneration == next.partialValue,
              targetTimeNanoseconds > startTimeNanoseconds,
              configurationFingerprint > 0, topologyFingerprint > 0,
              fingerprint == Self.hash(environmentIdentifier: environmentIdentifier, baseGeneration: baseGeneration,
                shadowGeneration: shadowGeneration, start: startTimeNanoseconds, target: targetTimeNanoseconds,
                configuration: configurationFingerprint, topology: topologyFingerprint) else {
            throw NXError.invalidState("transaction token fingerprint")
        }
    }

    private static func hash(environmentIdentifier: UInt32, baseGeneration: UInt64, shadowGeneration: UInt64,
                             start: UInt64, target: UInt64, configuration: UInt64, topology: UInt64) -> UInt64 {
        var h: UInt64 = 14_695_981_039_346_656_037
        for raw in [UInt64(environmentIdentifier), baseGeneration, shadowGeneration, start, target, configuration, topology] {
            var value = raw.littleEndian
            withUnsafeBytes(of: &value) { bytes in for byte in bytes { h = (h ^ UInt64(byte)) &* 1_099_511_628_211 } }
        }
        return h == 0 ? 14_695_981_039_346_656_037 : h
    }
}

public struct NXPreparedShadow: Sendable {
    public var token: NXTransactionToken
    public var state: NXAuthoritativeState
    public var certificate: NXConvergenceCertificate
    public var attemptedSubsteps: Int
}

/// Single-owner authoritative state machine. It deliberately has no public setter for committed state.
/// A root solves into a shadow, may be retried with smaller time slices, then requires an exact token
/// to publish. Abort drops the shadow without modifying the committed generation.
public actor NXAuthoritativeSolver {
    public let environmentIdentifier: UInt32
    public let configurationFingerprint: UInt64
    public let topologyFingerprint: UInt64
    private var committed: NXAuthoritativeState
    private var activeToken: NXTransactionToken?
    private var prepared: NXPreparedShadow?

    public init(environmentIdentifier: UInt32, configurationFingerprint: UInt64,
                topologyFingerprint: UInt64, initialState: NXAuthoritativeState) throws {
        guard configurationFingerprint > 0, topologyFingerprint > 0 else {
            throw NXError.invalidConfiguration("authoritative solver identity")
        }
        self.environmentIdentifier = environmentIdentifier
        self.configurationFingerprint = configurationFingerprint
        self.topologyFingerprint = topologyFingerprint
        committed = initialState
    }

    public func committedState() -> NXAuthoritativeState { committed }

    public func begin(targetTimeNanoseconds: UInt64) throws -> NXTransactionToken {
        guard activeToken == nil, prepared == nil else { throw NXError.invalidState("transaction already active") }
        let token = try NXTransactionToken(environmentIdentifier: environmentIdentifier,
            baseGeneration: committed.generation, startTimeNanoseconds: committed.timeNanoseconds,
            targetTimeNanoseconds: targetTimeNanoseconds, configurationFingerprint: configurationFingerprint,
            topologyFingerprint: topologyFingerprint)
        activeToken = token
        return token
    }

    /// The owner supplies a problem builder because a nonlinear residual may depend on the substep
    /// interval. In bounded mode a failed full interval is bisected deterministically until the
    /// configured maximum count; successful substeps become the next local base but remain unpublished.
    public func solve(token: NXTransactionToken, profile: NXExecutionProfile,
                      maximumSubsteps: Int = 8,
                      problemBuilder: @Sendable (_ localState: NXAuthoritativeState,
                                                  _ endTimeNanoseconds: UInt64) throws -> any NXNonlinearProblem) throws -> NXPreparedShadow {
        try token.validate()
        guard activeToken == token, prepared == nil, maximumSubsteps > 0,
              token.baseGeneration == committed.generation,
              token.startTimeNanoseconds == committed.timeNanoseconds else {
            throw NXError.invalidState("stale solver token")
        }
        var local = committed
        var currentTime = token.startTimeNanoseconds
        var interval = token.targetTimeNanoseconds - token.startTimeNanoseconds
        var acceptedSubsteps = 0
        var attempts = 0
        var lastCertificate: NXConvergenceCertificate?

        while currentTime < token.targetTimeNanoseconds {
            guard attempts < maximumSubsteps * 4 else { throw NXError.lineSearchFailed }
            attempts += 1
            let remaining = token.targetTimeNanoseconds - currentTime
            let step = min(interval, remaining)
            guard step > 0 else { throw NXError.numericalBreakdown("zero substep") }
            let end = currentTime + step
            let problem = try problemBuilder(local, end)
            guard problem.dimension == local.scalars.count else { throw NXError.invalidConfiguration("problem/state dimension") }
            let result = try NXSemismoothNewton.solve(problem: problem, initialState: local.scalars, profile: profile)
            switch result.disposition {
            case .committed(let certificate):
                local.scalars = result.state
                local.timeNanoseconds = end
                currentTime = end
                acceptedSubsteps += 1
                lastCertificate = certificate
                if currentTime < token.targetTimeNanoseconds {
                    interval = min(token.targetTimeNanoseconds - currentTime, max(step, interval))
                }
            case .substepRequired(let certificate):
                lastCertificate = certificate
                guard acceptedSubsteps < maximumSubsteps, step > 1 else { throw NXError.lineSearchFailed }
                interval = max(1, step / 2)
            case .rejected(let certificate):
                lastCertificate = certificate
                throw NXError.numericalBreakdown("adaptive nonlinear solve rejected: \(String(describing: certificate))")
            case .invalidInput(let reason): throw NXError.invalidState(reason)
            }
        }
        guard let certificate = lastCertificate, local.timeNanoseconds == token.targetTimeNanoseconds else {
            throw NXError.numericalBreakdown("root did not reach target time")
        }
        local.generation = token.shadowGeneration
        let shadow = NXPreparedShadow(token: token, state: local, certificate: certificate, attemptedSubsteps: attempts)
        prepared = shadow
        return shadow
    }

    public func publish(_ token: NXTransactionToken) throws -> NXAuthoritativeState {
        try token.validate()
        guard activeToken == token, let prepared, prepared.token == token,
              prepared.state.generation == token.shadowGeneration,
              prepared.state.timeNanoseconds == token.targetTimeNanoseconds else {
            throw NXError.invalidState("prepared shadow mismatch")
        }
        committed = prepared.state
        self.prepared = nil; activeToken = nil
        return committed
    }

    public func abort(_ token: NXTransactionToken) throws {
        try token.validate()
        guard activeToken == token else { throw NXError.invalidState("abort token mismatch") }
        prepared = nil; activeToken = nil
    }
}
