import Foundation

public struct NXEqualityConstraintRow: Codable, Equatable, Sendable {
    public var multiplierIndex: Int
    public var jacobian: [NXJacobianEntry]
    public var velocityBias: Float
    public var compliance: Float

    public init(multiplierIndex: Int, jacobian: [NXJacobianEntry],
                velocityBias: Float = 0, compliance: Float = 0) throws {
        let sorted = jacobian.sorted { $0.index < $1.index }
        guard multiplierIndex >= 0, !sorted.isEmpty,
              Set(sorted.map(\.index)).count == sorted.count,
              velocityBias.isFinite, compliance.isFinite, compliance >= 0 else {
            throw NXError.invalidConfiguration("equality constraint row")
        }
        self.multiplierIndex = multiplierIndex; self.jacobian = sorted
        self.velocityBias = velocityBias; self.compliance = compliance
    }
}

/// KKT equality block: momentum += Jᵀη; Rη = Jv + b + Cη. Attachments, bilateral joints,
/// pressure constraints and rod inextensibility can share this algebra with different row builders.
public struct NXEqualityConstraintContribution: NXPhysicsContribution {
    public let name: String
    public let stateDimension: Int
    public let rows: [NXEqualityConstraintRow]

    public init(name: String = "equality-kkt", stateDimension: Int,
                rows: [NXEqualityConstraintRow]) throws {
        guard stateDimension > 0, !name.isEmpty, !rows.isEmpty else {
            throw NXError.invalidConfiguration("equality contribution")
        }
        let multipliers = Set(rows.map(\.multiplierIndex))
        guard multipliers.count == rows.count else { throw NXError.invalidConfiguration("duplicate equality multiplier") }
        for row in rows {
            guard row.multiplierIndex < stateDimension,
                  row.jacobian.allSatisfy({ $0.index < stateDimension && !multipliers.contains($0.index) }) else {
                throw NXError.invalidConfiguration("equality row range")
            }
        }
        self.name = name; self.stateDimension = stateDimension; self.rows = rows
    }

    public func addResidual(state: [Float], to residual: inout [Float]) throws {
        guard state.count == stateDimension, residual.count == stateDimension, state.allSatisfy(\.isFinite) else {
            throw NXError.invalidState("equality residual")
        }
        for row in rows {
            let eta = state[row.multiplierIndex]
            var constraint = row.velocityBias + row.compliance * eta
            for entry in row.jacobian {
                residual[entry.index] += entry.coefficient * eta
                constraint += entry.coefficient * state[entry.index]
            }
            residual[row.multiplierIndex] += constraint
        }
        guard residual.allSatisfy(\.isFinite) else { throw NXError.numericalBreakdown("equality residual nonfinite") }
    }

    public func addJacobianVector(state: [Float], vector: [Float], to product: inout [Float]) throws {
        guard state.count == stateDimension, vector.count == stateDimension, product.count == stateDimension else {
            throw NXError.invalidState("equality Jv")
        }
        for row in rows {
            let dEta = vector[row.multiplierIndex]
            var constraint = row.compliance * dEta
            for entry in row.jacobian {
                product[entry.index] += entry.coefficient * dEta
                constraint += entry.coefficient * vector[entry.index]
            }
            product[row.multiplierIndex] += constraint
        }
        guard product.allSatisfy(\.isFinite) else { throw NXError.numericalBreakdown("equality Jv nonfinite") }
    }

    public func admissibleStep(state: [Float], direction: [Float]) throws -> NXGeometricCertificate {
        guard state.count == stateDimension, direction.count == stateDimension,
              state.allSatisfy(\.isFinite), direction.allSatisfy(\.isFinite) else {
            throw NXError.invalidState("equality safe step")
        }
        return NXGeometricCertificate(minimumDistance: 1, minimumDeterminantF: 1,
            minimumVolume: 1, minimumRodLength: 1, finite: true, safeStep: 1)
    }
}
