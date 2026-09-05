import Foundation

public enum NXResidualBlock: String, Codable, CaseIterable, Sendable {
    case momentum
    case pressure
    case equality
    case transport
    case contact
}

public struct NXResidualRange: Codable, Equatable, Sendable {
    public var block: NXResidualBlock
    public var range: Range<Int>
    public var scale: Float

    public init(block: NXResidualBlock, range: Range<Int>, scale: Float = 1) throws {
        guard range.lowerBound >= 0, range.upperBound >= range.lowerBound,
              scale.isFinite, scale > 0 else { throw NXError.invalidConfiguration("residual range") }
        self.block = block; self.range = range; self.scale = scale
    }
}

/// One physics term contributes to the same monolithic residual and Jacobian-vector product.
/// Implementations may represent rigid inertia, ABA articulated dynamics, FEM/MPM stress,
/// incompressibility, rods, transport, attachments or contact. They do not own publication.
public protocol NXPhysicsContribution: Sendable {
    var name: String { get }
    var stateDimension: Int { get }
    func addResidual(state: [Float], to residual: inout [Float]) throws
    func addJacobianVector(state: [Float], vector: [Float], to product: inout [Float]) throws
    func admissibleStep(state: [Float], direction: [Float]) throws -> NXGeometricCertificate
}

public protocol NXPreconditionerFactory: Sendable {
    func make(state: [Float], newtonIteration: Int) throws -> any NXPreconditioner
}

public struct NXCoupledSystem: NXNonlinearProblem {
    public let dimension: Int
    public let contributions: [any NXPhysicsContribution]
    public let residualRanges: [NXResidualRange]
    public let preconditionerFactory: any NXPreconditionerFactory

    public init(dimension: Int, contributions: [any NXPhysicsContribution],
                residualRanges: [NXResidualRange], preconditionerFactory: any NXPreconditionerFactory) throws {
        guard dimension > 0, !contributions.isEmpty,
              contributions.allSatisfy({ $0.stateDimension == dimension }),
              !residualRanges.isEmpty,
              residualRanges.allSatisfy({ $0.range.upperBound <= dimension }) else {
            throw NXError.invalidConfiguration("coupled system")
        }
        var covered = Array(repeating: false, count: dimension)
        for item in residualRanges {
            for i in item.range {
                guard !covered[i] else { throw NXError.invalidConfiguration("overlapping residual ranges") }
                covered[i] = true
            }
        }
        guard covered.allSatisfy({ $0 }) else { throw NXError.invalidConfiguration("residual ranges leave holes") }
        self.dimension = dimension; self.contributions = contributions
        self.residualRanges = residualRanges; self.preconditionerFactory = preconditionerFactory
    }

    public func residual(at state: [Float], into residual: inout [Float]) throws {
        guard state.count == dimension, state.allSatisfy(\.isFinite) else { throw NXError.invalidState("coupled state") }
        residual = Array(repeating: 0, count: dimension)
        for term in contributions { try term.addResidual(state: state, to: &residual) }
        guard residual.allSatisfy(\.isFinite) else { throw NXError.numericalBreakdown("coupled residual") }
    }

    public func linearization(at state: [Float]) throws -> any NXLinearOperator {
        guard state.count == dimension, state.allSatisfy(\.isFinite) else { throw NXError.invalidState("linearization state") }
        return NXCoupledLinearization(state: state, contributions: contributions, dimension: dimension)
    }

    public func preconditioner(at state: [Float], newtonIteration: Int) throws -> any NXPreconditioner {
        try preconditionerFactory.make(state: state, newtonIteration: newtonIteration)
    }

    public func safeStep(from state: [Float], along direction: [Float]) throws -> NXGeometricCertificate {
        guard state.count == dimension, direction.count == dimension else { throw NXError.invalidState("safe-step dimension") }
        var answer = NXGeometricCertificate(minimumDistance: .greatestFiniteMagnitude,
            minimumDeterminantF: .greatestFiniteMagnitude, minimumVolume: .greatestFiniteMagnitude,
            minimumRodLength: .greatestFiniteMagnitude, finite: true, safeStep: 1)
        for term in contributions {
            let item = try term.admissibleStep(state: state, direction: direction)
            answer.minimumDistance = min(answer.minimumDistance, item.minimumDistance)
            answer.minimumDeterminantF = min(answer.minimumDeterminantF, item.minimumDeterminantF)
            answer.minimumVolume = min(answer.minimumVolume, item.minimumVolume)
            answer.minimumRodLength = min(answer.minimumRodLength, item.minimumRodLength)
            answer.safeStep = min(answer.safeStep, item.safeStep)
            answer.finite = answer.finite && item.finite
        }
        return answer
    }

    public func diagnostics(at state: [Float], residual: [Float], newtonIterations: Int,
                            krylovIterations: Int, geometric: NXGeometricCertificate) throws -> NXConvergenceCertificate {
        guard residual.count == dimension else { throw NXError.invalidState("diagnostic residual") }
        var norms: [NXResidualBlock: Float] = [:]
        for item in residualRanges {
            let scaled = item.range.map { residual[$0] / item.scale }
            norms[item.block] = try NXVectorMath.norm(scaled)
        }
        let total = try NXVectorMath.norm(residual)
        let reference = max(1, try NXVectorMath.norm(state))
        return NXConvergenceCertificate(residualNorm: total, relativeResidual: total / reference,
            coneDistance: norms[.contact] ?? 0,
            complementarity: norms[.contact] ?? 0,
            pressureResidual: norms[.pressure] ?? 0,
            equalityResidual: norms[.equality] ?? 0,
            momentumResidual: norms[.momentum] ?? 0,
            newtonIterations: newtonIterations, krylovIterations: krylovIterations, geometric: geometric)
    }
}

private struct NXCoupledLinearization: NXLinearOperator {
    let state: [Float]
    let contributions: [any NXPhysicsContribution]
    let dimension: Int
    func apply(_ x: [Float], into y: inout [Float]) throws {
        guard x.count == dimension, x.allSatisfy(\.isFinite) else { throw NXError.invalidState("Jv vector") }
        y = Array(repeating: 0, count: dimension)
        for term in contributions { try term.addJacobianVector(state: state, vector: x, to: &y) }
        guard y.allSatisfy(\.isFinite) else { throw NXError.numericalBreakdown("Jv result") }
    }
}

public struct NXFixedPreconditionerFactory: NXPreconditionerFactory {
    public let preconditioner: any NXPreconditioner
    public init(_ preconditioner: any NXPreconditioner) { self.preconditioner = preconditioner }
    public func make(state: [Float], newtonIteration: Int) throws -> any NXPreconditioner { preconditioner }
}
