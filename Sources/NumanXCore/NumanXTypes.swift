import Foundation

public enum NXPrecision: String, Codable, Sendable { case fp32, fp16, bfloat16 }
public enum NXExecutionMode: String, Codable, Sendable { case boundedDeterministic, adaptiveAccuracy }

public struct NXExecutionProfile: Codable, Equatable, Sendable {
    public var mode: NXExecutionMode
    public var maximumNewtonIterations: Int
    public var maximumKrylovIterations: Int
    public var krylovRestart: Int
    public var maximumLineSearchTrials: Int
    public var relativeResidualTolerance: Float
    public var absoluteResidualTolerance: Float
    public var coneTolerance: Float
    public var complementarityTolerance: Float
    public var minimumSafeStep: Float
    public var armijo: Float

    public init(mode: NXExecutionMode = .boundedDeterministic,
                maximumNewtonIterations: Int = 8,
                maximumKrylovIterations: Int = 48,
                krylovRestart: Int = 16,
                maximumLineSearchTrials: Int = 3,
                relativeResidualTolerance: Float = 1e-4,
                absoluteResidualTolerance: Float = 1e-6,
                coneTolerance: Float = 1e-5,
                complementarityTolerance: Float = 1e-5,
                minimumSafeStep: Float = 1e-6,
                armijo: Float = 1e-4) throws {
        guard maximumNewtonIterations > 0, maximumKrylovIterations > 0,
              krylovRestart > 0, krylovRestart <= maximumKrylovIterations,
              maximumLineSearchTrials > 0,
              relativeResidualTolerance > 0, absoluteResidualTolerance > 0,
              coneTolerance > 0, complementarityTolerance > 0,
              minimumSafeStep > 0, minimumSafeStep <= 1,
              armijo > 0, armijo < 1 else { throw NXError.invalidConfiguration("execution profile") }
        self.mode = mode
        self.maximumNewtonIterations = maximumNewtonIterations
        self.maximumKrylovIterations = maximumKrylovIterations
        self.krylovRestart = krylovRestart
        self.maximumLineSearchTrials = maximumLineSearchTrials
        self.relativeResidualTolerance = relativeResidualTolerance
        self.absoluteResidualTolerance = absoluteResidualTolerance
        self.coneTolerance = coneTolerance
        self.complementarityTolerance = complementarityTolerance
        self.minimumSafeStep = minimumSafeStep
        self.armijo = armijo
    }

    public static var training: NXExecutionProfile {
        try! NXExecutionProfile(mode: .boundedDeterministic, maximumNewtonIterations: 6,
            maximumKrylovIterations: 32, krylovRestart: 16, maximumLineSearchTrials: 2,
            relativeResidualTolerance: 2e-4, absoluteResidualTolerance: 2e-6,
            coneTolerance: 2e-5, complementarityTolerance: 2e-5)
    }

    public static var scientific: NXExecutionProfile {
        try! NXExecutionProfile(mode: .adaptiveAccuracy, maximumNewtonIterations: 18,
            maximumKrylovIterations: 160, krylovRestart: 32, maximumLineSearchTrials: 8,
            relativeResidualTolerance: 1e-6, absoluteResidualTolerance: 1e-8,
            coneTolerance: 1e-7, complementarityTolerance: 1e-7, minimumSafeStep: 1e-8)
    }
}

public struct NXStateLayout: Codable, Equatable, Sendable {
    public var primalRange: Range<Int>
    public var pressureRange: Range<Int>
    public var equalityMultiplierRange: Range<Int>
    public var contactMultiplierRange: Range<Int>

    public init(primalCount: Int, pressureCount: Int = 0,
                equalityMultiplierCount: Int = 0, contactCount: Int = 0) throws {
        guard primalCount > 0, pressureCount >= 0, equalityMultiplierCount >= 0, contactCount >= 0 else {
            throw NXError.invalidConfiguration("state layout")
        }
        var cursor = 0
        primalRange = cursor..<(cursor + primalCount); cursor += primalCount
        pressureRange = cursor..<(cursor + pressureCount); cursor += pressureCount
        equalityMultiplierRange = cursor..<(cursor + equalityMultiplierCount); cursor += equalityMultiplierCount
        let contactScalars = contactCount.multipliedReportingOverflow(by: 3)
        guard !contactScalars.overflow else { throw NXError.capacity("contact layout") }
        contactMultiplierRange = cursor..<(cursor + contactScalars.partialValue)
    }
    public var scalarCount: Int { contactMultiplierRange.upperBound }
    public var contactCount: Int { contactMultiplierRange.count / 3 }
}

public struct NXAuthoritativeState: Equatable, Sendable {
    public let layout: NXStateLayout
    public var scalars: [Float]
    public var generation: UInt64
    public var timeNanoseconds: UInt64

    public init(layout: NXStateLayout, scalars: [Float], generation: UInt64, timeNanoseconds: UInt64) throws {
        guard scalars.count == layout.scalarCount, scalars.allSatisfy(\.isFinite) else {
            throw NXError.invalidState("authoritative vector")
        }
        self.layout = layout; self.scalars = scalars; self.generation = generation; self.timeNanoseconds = timeNanoseconds
    }
}

public struct NXGeometricCertificate: Codable, Equatable, Sendable {
    public var minimumDistance: Float
    public var minimumDeterminantF: Float
    public var minimumVolume: Float
    public var minimumRodLength: Float
    public var finite: Bool
    public var safeStep: Float

    public init(minimumDistance: Float = .greatestFiniteMagnitude,
                minimumDeterminantF: Float = .greatestFiniteMagnitude,
                minimumVolume: Float = .greatestFiniteMagnitude,
                minimumRodLength: Float = .greatestFiniteMagnitude,
                finite: Bool = true, safeStep: Float = 1) {
        self.minimumDistance = minimumDistance; self.minimumDeterminantF = minimumDeterminantF
        self.minimumVolume = minimumVolume; self.minimumRodLength = minimumRodLength
        self.finite = finite; self.safeStep = safeStep
    }
    public func validated(minimumSafeStep: Float) throws -> Self {
        guard finite, minimumDistance.isFinite, minimumDeterminantF.isFinite,
              minimumVolume.isFinite, minimumRodLength.isFinite, safeStep.isFinite,
              safeStep >= minimumSafeStep, safeStep <= 1 else { throw NXError.geometryRejected }
        return self
    }
}

public struct NXConvergenceCertificate: Codable, Equatable, Sendable {
    public var residualNorm: Float
    public var relativeResidual: Float
    public var coneDistance: Float
    public var complementarity: Float
    public var pressureResidual: Float
    public var equalityResidual: Float
    public var momentumResidual: Float
    public var newtonIterations: Int
    public var krylovIterations: Int
    public var geometric: NXGeometricCertificate

    public func satisfies(_ profile: NXExecutionProfile) -> Bool {
        residualNorm.isFinite && relativeResidual.isFinite && coneDistance.isFinite && complementarity.isFinite &&
        residualNorm <= max(profile.absoluteResidualTolerance, profile.relativeResidualTolerance) &&
        relativeResidual <= profile.relativeResidualTolerance && coneDistance <= profile.coneTolerance &&
        complementarity <= profile.complementarityTolerance && geometric.finite
    }
}

public enum NXSolveDisposition: Equatable, Sendable {
    case committed(NXConvergenceCertificate)
    case substepRequired(NXConvergenceCertificate?)
    case rejected(NXConvergenceCertificate?)
    case invalidInput(String)
}

public enum NXError: Error, Equatable, Sendable {
    case invalidConfiguration(String)
    case invalidState(String)
    case capacity(String)
    case numericalBreakdown(String)
    case geometryRejected
    case lineSearchFailed
}
