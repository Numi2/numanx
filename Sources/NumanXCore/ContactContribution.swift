import Foundation

public struct NXJacobianEntry: Codable, Equatable, Sendable {
    public var index: Int
    public var coefficient: Float
    public init(index: Int, coefficient: Float) throws {
        guard index >= 0, coefficient.isFinite else { throw NXError.invalidConfiguration("contact Jacobian entry") }
        self.index = index; self.coefficient = coefficient
    }
}

public struct NXContactRow: Codable, Equatable, Sendable {
    public var multiplierOffset: Int
    public var normal: [NXJacobianEntry]
    public var tangent1: [NXJacobianEntry]
    public var tangent2: [NXJacobianEntry]
    public var velocityBias: SIMD3<Float>
    public var law: NXCoulombContact
    public var rho: Float

    public init(multiplierOffset: Int,
                normal: [NXJacobianEntry], tangent1: [NXJacobianEntry], tangent2: [NXJacobianEntry],
                velocityBias: SIMD3<Float> = .zero, law: NXCoulombContact, rho: Float = 1) throws {
        guard multiplierOffset >= 0, !normal.isEmpty, rho.isFinite, rho > 0,
              velocityBias.x.isFinite, velocityBias.y.isFinite, velocityBias.z.isFinite else {
            throw NXError.invalidConfiguration("contact row")
        }
        self.multiplierOffset = multiplierOffset
        self.normal = try Self.canonical(normal)
        self.tangent1 = try Self.canonical(tangent1)
        self.tangent2 = try Self.canonical(tangent2)
        self.velocityBias = velocityBias; self.law = law; self.rho = rho
    }

    private static func canonical(_ entries: [NXJacobianEntry]) throws -> [NXJacobianEntry] {
        let sorted = entries.sorted { $0.index < $1.index }
        guard Set(sorted.map(\.index)).count == sorted.count else {
            throw NXError.invalidConfiguration("duplicate contact Jacobian index")
        }
        return sorted
    }
}

/// Exact circular Coulomb contacts in the same KKT residual as body momentum. Each row contributes
/// Jᵀλ to generalized momentum and a projection-based natural residual to its three multiplier slots.
public struct NXContactContribution: NXPhysicsContribution {
    public let name = "exact-circular-coulomb-contact"
    public let stateDimension: Int
    public let rows: [NXContactRow]

    public init(stateDimension: Int, rows: [NXContactRow]) throws {
        guard stateDimension > 0, !rows.isEmpty else { throw NXError.invalidConfiguration("contact contribution") }
        var multipliers = Set<Int>()
        for row in rows {
            guard row.multiplierOffset + 3 <= stateDimension else { throw NXError.invalidConfiguration("contact multiplier range") }
            for index in row.multiplierOffset..<(row.multiplierOffset + 3) {
                guard multipliers.insert(index).inserted else { throw NXError.invalidConfiguration("overlapping contact multipliers") }
            }
            for entry in row.normal + row.tangent1 + row.tangent2 {
                guard entry.index < stateDimension, !multipliers.contains(entry.index) else {
                    throw NXError.invalidConfiguration("contact Jacobian references invalid/generalized multiplier state")
                }
            }
        }
        self.stateDimension = stateDimension; self.rows = rows
    }

    public func addResidual(state: [Float], to residual: inout [Float]) throws {
        guard state.count == stateDimension, residual.count == stateDimension, state.allSatisfy(\.isFinite) else {
            throw NXError.invalidState("contact residual dimension")
        }
        for row in rows {
            let lambda = SIMD3<Float>(state[row.multiplierOffset], state[row.multiplierOffset + 1], state[row.multiplierOffset + 2])
            let velocity = SIMD3<Float>(evaluate(row.normal, state) + row.velocityBias.x,
                                        evaluate(row.tangent1, state) + row.velocityBias.y,
                                        evaluate(row.tangent2, state) + row.velocityBias.z)
            scatter(row.normal, multiplier: lambda.x, into: &residual)
            scatter(row.tangent1, multiplier: lambda.y, into: &residual)
            scatter(row.tangent2, multiplier: lambda.z, into: &residual)
            let contactResidual = try NXCoulombConeResidual.naturalResidual(lambda: lambda,
                relativeVelocity: velocity, law: row.law, rho: row.rho)
            residual[row.multiplierOffset] += contactResidual.x
            residual[row.multiplierOffset + 1] += contactResidual.y
            residual[row.multiplierOffset + 2] += contactResidual.z
        }
        guard residual.allSatisfy(\.isFinite) else { throw NXError.numericalBreakdown("contact residual") }
    }

    public func addJacobianVector(state: [Float], vector: [Float], to product: inout [Float]) throws {
        guard state.count == stateDimension, vector.count == stateDimension, product.count == stateDimension,
              state.allSatisfy(\.isFinite), vector.allSatisfy(\.isFinite) else {
            throw NXError.invalidState("contact Jv dimension")
        }
        for row in rows {
            let lambda = SIMD3<Float>(state[row.multiplierOffset], state[row.multiplierOffset + 1], state[row.multiplierOffset + 2])
            let dLambda = SIMD3<Float>(vector[row.multiplierOffset], vector[row.multiplierOffset + 1], vector[row.multiplierOffset + 2])
            let velocity = SIMD3<Float>(evaluate(row.normal, state) + row.velocityBias.x,
                                        evaluate(row.tangent1, state) + row.velocityBias.y,
                                        evaluate(row.tangent2, state) + row.velocityBias.z)
            let dVelocity = SIMD3<Float>(evaluate(row.normal, vector), evaluate(row.tangent1, vector), evaluate(row.tangent2, vector))
            scatter(row.normal, multiplier: dLambda.x, into: &product)
            scatter(row.tangent1, multiplier: dLambda.y, into: &product)
            scatter(row.tangent2, multiplier: dLambda.z, into: &product)
            let dr = try NXCoulombConeResidual.directionalDerivative(lambda: lambda, relativeVelocity: velocity,
                dLambda: dLambda, dVelocity: dVelocity, law: row.law, rho: row.rho)
            product[row.multiplierOffset] += dr.x
            product[row.multiplierOffset + 1] += dr.y
            product[row.multiplierOffset + 2] += dr.z
        }
        guard product.allSatisfy(\.isFinite) else { throw NXError.numericalBreakdown("contact Jv") }
    }

    public func admissibleStep(state: [Float], direction: [Float]) throws -> NXGeometricCertificate {
        guard state.count == stateDimension, direction.count == stateDimension,
              state.allSatisfy(\.isFinite), direction.allSatisfy(\.isFinite) else {
            throw NXError.invalidState("contact safe step")
        }
        // CCD owns geometric separation. Cone multipliers are globally projected by the residual and
        // therefore do not need a positivity clipping step here.
        return NXGeometricCertificate(minimumDistance: 1, minimumDeterminantF: 1,
            minimumVolume: 1, minimumRodLength: 1, finite: true, safeStep: 1)
    }

    public func maximumDiagnostics(state: [Float]) throws -> (coneDistance: Float, complementarity: Float) {
        guard state.count == stateDimension else { throw NXError.invalidState("contact diagnostic state") }
        var cone: Float = 0, complementarity: Float = 0
        for row in rows {
            let lambda = SIMD3<Float>(state[row.multiplierOffset], state[row.multiplierOffset + 1], state[row.multiplierOffset + 2])
            let velocity = SIMD3<Float>(evaluate(row.normal, state) + row.velocityBias.x,
                                        evaluate(row.tangent1, state) + row.velocityBias.y,
                                        evaluate(row.tangent2, state) + row.velocityBias.z)
            let d = try NXCoulombConeResidual.diagnostics(lambda: lambda, relativeVelocity: velocity, law: row.law)
            cone = max(cone, d.coneDistance); complementarity = max(complementarity, d.complementarity)
        }
        return (cone, complementarity)
    }

    private func evaluate(_ entries: [NXJacobianEntry], _ vector: [Float]) -> Float {
        var value: Float = 0
        for entry in entries { value += entry.coefficient * vector[entry.index] }
        return value
    }

    private func scatter(_ entries: [NXJacobianEntry], multiplier: Float, into vector: inout [Float]) {
        for entry in entries { vector[entry.index] += entry.coefficient * multiplier }
    }
}
