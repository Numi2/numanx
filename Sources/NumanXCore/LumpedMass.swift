import Foundation

public struct NXLumpedMassNode: Codable, Equatable, Sendable {
    public var velocityOffset: Int
    public var mass: Float
    public var previousVelocity: SIMD3<Float>
    public var externalImpulse: SIMD3<Float>

    public init(velocityOffset: Int, mass: Float, previousVelocity: SIMD3<Float>,
                externalImpulse: SIMD3<Float> = .zero) throws {
        guard velocityOffset >= 0, mass.isFinite, mass > 0,
              previousVelocity.x.isFinite, previousVelocity.y.isFinite, previousVelocity.z.isFinite,
              externalImpulse.x.isFinite, externalImpulse.y.isFinite, externalImpulse.z.isFinite else {
            throw NXError.invalidConfiguration("lumped mass node")
        }
        self.velocityOffset = velocityOffset; self.mass = mass
        self.previousVelocity = previousVelocity; self.externalImpulse = externalImpulse
    }
}

public struct NXLumpedMassContribution: NXPhysicsContribution {
    public let name = "lumped-mass-backward-euler"
    public let stateDimension: Int
    public let nodes: [NXLumpedMassNode]

    public init(stateDimension: Int, nodes: [NXLumpedMassNode]) throws {
        guard stateDimension > 0, !nodes.isEmpty else { throw NXError.invalidConfiguration("lumped mass") }
        var used = Set<Int>()
        for node in nodes {
            guard node.velocityOffset <= stateDimension - 3 else { throw NXError.invalidConfiguration("lumped mass range") }
            for index in node.velocityOffset..<(node.velocityOffset + 3) {
                guard used.insert(index).inserted else { throw NXError.invalidConfiguration("overlapping lumped mass node") }
            }
        }
        self.stateDimension = stateDimension; self.nodes = nodes
    }

    public func addResidual(state: [Float], to residual: inout [Float]) throws {
        guard state.count == stateDimension, residual.count == stateDimension else { throw NXError.invalidState("lumped mass residual") }
        for node in nodes {
            for axis in 0..<3 {
                let index = node.velocityOffset + axis
                residual[index] += node.mass * (state[index] - node.previousVelocity[axis]) - node.externalImpulse[axis]
            }
        }
        guard residual.allSatisfy(\.isFinite) else { throw NXError.numericalBreakdown("lumped mass residual") }
    }

    public func addJacobianVector(state: [Float], vector: [Float], to product: inout [Float]) throws {
        guard state.count == stateDimension, vector.count == stateDimension, product.count == stateDimension else {
            throw NXError.invalidState("lumped mass Jv")
        }
        for node in nodes {
            for axis in 0..<3 { product[node.velocityOffset + axis] += node.mass * vector[node.velocityOffset + axis] }
        }
    }

    public func admissibleStep(state: [Float], direction: [Float]) throws -> NXGeometricCertificate {
        guard state.count == stateDimension, direction.count == stateDimension,
              state.allSatisfy(\.isFinite), direction.allSatisfy(\.isFinite) else { throw NXError.invalidState("lumped mass safe step") }
        return NXGeometricCertificate(minimumDistance: 1, minimumDeterminantF: 1,
            minimumVolume: 1, minimumRodLength: 1, finite: true, safeStep: 1)
    }
}
