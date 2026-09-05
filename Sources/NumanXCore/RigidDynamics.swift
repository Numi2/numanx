import Foundation

public struct NXSpatialInertia: Codable, Equatable, Sendable {
    public var mass: Float
    public var centerOfMass: SIMD3<Float>
    /// Symmetric inertia tensor about the center of mass: xx, yy, zz, xy, xz, yz.
    public var inertia: SIMD6<Float>

    public init(mass: Float, centerOfMass: SIMD3<Float> = .zero, inertia: SIMD6<Float>) throws {
        guard mass.isFinite, mass > 0, centerOfMass.x.isFinite, centerOfMass.y.isFinite, centerOfMass.z.isFinite,
              inertia.x.isFinite, inertia.y.isFinite, inertia.z.isFinite,
              inertia.w.isFinite, inertia[4].isFinite, inertia[5].isFinite,
              inertia.x > 0, inertia.y > 0, inertia.z > 0 else {
            throw NXError.invalidConfiguration("spatial inertia")
        }
        self.mass = mass; self.centerOfMass = centerOfMass; self.inertia = inertia
    }

    public func generalizedMassMatrix() -> [Float] {
        let m = mass
        let c = centerOfMass
        // Spatial inertia in [linear, angular] ordering. Off-diagonal m*[c]x coupling.
        var a = Array(repeating: Float.zero, count: 36)
        for i in 0..<3 { a[i * 6 + i] = m }
        let skew: [[Float]] = [[0, -c.z, c.y], [c.z, 0, -c.x], [-c.y, c.x, 0]]
        for r in 0..<3 {
            for col in 0..<3 {
                a[r * 6 + (col + 3)] = -m * skew[r][col]
                a[(r + 3) * 6 + col] = m * skew[r][col]
            }
        }
        let ixx = inertia.x, iyy = inertia.y, izz = inertia.z
        let ixy = inertia.w, ixz = inertia[4], iyz = inertia[5]
        let Ic: [[Float]] = [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]
        // Parallel-axis: Ic + m*(||c||² I - c cᵀ).
        let c2 = c.x * c.x + c.y * c.y + c.z * c.z
        let cv = [c.x, c.y, c.z]
        for r in 0..<3 {
            for col in 0..<3 {
                let pa = m * ((r == col ? c2 : 0) - cv[r] * cv[col])
                a[(r + 3) * 6 + (col + 3)] = Ic[r][col] + pa
            }
        }
        return a
    }
}

public struct NXRigidBodyStep: Codable, Equatable, Sendable {
    public var velocityOffset: Int
    public var inertia: NXSpatialInertia
    public var previousVelocity: SIMD6<Float>
    public var externalImpulse: SIMD6<Float>

    public init(velocityOffset: Int, inertia: NXSpatialInertia,
                previousVelocity: SIMD6<Float>, externalImpulse: SIMD6<Float> = .zero) throws {
        guard velocityOffset >= 0,
              (0..<6).allSatisfy({ previousVelocity[$0].isFinite && externalImpulse[$0].isFinite }) else {
            throw NXError.invalidConfiguration("rigid body step")
        }
        self.velocityOffset = velocityOffset; self.inertia = inertia
        self.previousVelocity = previousVelocity; self.externalImpulse = externalImpulse
    }
}

/// Backward-Euler generalized inertia term M(v-v_n)-J_ext. Contacts/constraints contribute their
/// impulses through separate terms to the same residual; this block never performs split impulses.
public struct NXRigidDynamicsContribution: NXPhysicsContribution {
    public let name = "rigid-backward-euler"
    public let stateDimension: Int
    public let bodies: [NXRigidBodyStep]

    public init(stateDimension: Int, bodies: [NXRigidBodyStep]) throws {
        guard stateDimension > 0, !bodies.isEmpty,
              bodies.allSatisfy({ $0.velocityOffset + 6 <= stateDimension }) else {
            throw NXError.invalidConfiguration("rigid contribution")
        }
        var used = Set<Int>()
        for body in bodies {
            for i in body.velocityOffset..<(body.velocityOffset + 6) {
                guard used.insert(i).inserted else { throw NXError.invalidConfiguration("overlapping rigid velocity blocks") }
            }
        }
        self.stateDimension = stateDimension; self.bodies = bodies
    }

    public func addResidual(state: [Float], to residual: inout [Float]) throws {
        guard state.count == stateDimension, residual.count == stateDimension else { throw NXError.invalidState("rigid residual dimension") }
        for body in bodies {
            let matrix = body.inertia.generalizedMassMatrix()
            var delta = SIMD6<Float>.zero
            for i in 0..<6 { delta[i] = state[body.velocityOffset + i] - body.previousVelocity[i] }
            for row in 0..<6 {
                var value = -body.externalImpulse[row]
                for col in 0..<6 { value += matrix[row * 6 + col] * delta[col] }
                let index = body.velocityOffset + row
                residual[index] += value
                guard residual[index].isFinite else { throw NXError.numericalBreakdown("rigid inertia residual") }
            }
        }
    }

    public func addJacobianVector(state: [Float], vector: [Float], to product: inout [Float]) throws {
        guard state.count == stateDimension, vector.count == stateDimension, product.count == stateDimension else {
            throw NXError.invalidState("rigid Jv dimension")
        }
        for body in bodies {
            let matrix = body.inertia.generalizedMassMatrix()
            for row in 0..<6 {
                var value: Float = 0
                for col in 0..<6 { value += matrix[row * 6 + col] * vector[body.velocityOffset + col] }
                product[body.velocityOffset + row] += value
            }
        }
    }

    public func admissibleStep(state: [Float], direction: [Float]) throws -> NXGeometricCertificate {
        guard state.count == stateDimension, direction.count == stateDimension,
              state.allSatisfy(\.isFinite), direction.allSatisfy(\.isFinite) else {
            throw NXError.invalidState("rigid safe step")
        }
        return NXGeometricCertificate(minimumDistance: 1, minimumDeterminantF: 1,
            minimumVolume: 1, minimumRodLength: 1, finite: true, safeStep: 1)
    }

    public func densePatchMatrices() -> [UInt32: [Float]] {
        Dictionary(uniqueKeysWithValues: bodies.enumerated().map { (index, body) in
            (UInt32(index), body.inertia.generalizedMassMatrix())
        })
    }
}
