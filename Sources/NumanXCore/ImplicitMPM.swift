import Foundation

public struct NXMPMGridStencil: Codable, Equatable, Sendable {
    public var velocityOffset: Int
    public var gradient: SIMD3<Float>

    public init(velocityOffset: Int, gradient: SIMD3<Float>) throws {
        guard velocityOffset >= 0, gradient.x.isFinite, gradient.y.isFinite, gradient.z.isFinite else {
            throw NXError.invalidConfiguration("MPM stencil")
        }
        self.velocityOffset = velocityOffset; self.gradient = gradient
    }
}

public struct NXMPMParticle: Codable, Equatable, Sendable {
    public var previousDeformationGradient: NXMatrix3
    public var referenceVolume: Float
    public var material: NXNeoHookeanMaterial
    public var stencil: [NXMPMGridStencil]

    public init(previousDeformationGradient: NXMatrix3, referenceVolume: Float,
                material: NXNeoHookeanMaterial, stencil: [NXMPMGridStencil]) throws {
        guard previousDeformationGradient.allFinite,
              previousDeformationGradient.determinant > material.minimumDeterminant,
              referenceVolume.isFinite, referenceVolume > 0, !stencil.isEmpty else {
            throw NXError.invalidConfiguration("MPM particle")
        }
        self.previousDeformationGradient = previousDeformationGradient
        self.referenceVolume = referenceVolume; self.material = material; self.stencil = stencil
    }
}

/// Updated-Lagrangian implicit MPM constitutive block. Grid velocities are Newton unknowns; particle
/// deformation is F_trial=(I+dt*grad(v))F_n. Mass/P2G/G2P ownership remains separate, while this
/// contribution supplies exact particle stress and tangent action to the monolithic grid solve.
public struct NXImplicitMPMContribution: NXPhysicsContribution {
    public let name = "implicit-mpm-neo-hookean"
    public let stateDimension: Int
    public let timeStepSeconds: Float
    public let particles: [NXMPMParticle]

    public init(stateDimension: Int, timeStepSeconds: Float, particles: [NXMPMParticle]) throws {
        guard stateDimension > 0, timeStepSeconds.isFinite, timeStepSeconds > 0, !particles.isEmpty else {
            throw NXError.invalidConfiguration("implicit MPM")
        }
        for particle in particles {
            for node in particle.stencil {
                guard node.velocityOffset <= stateDimension - 3 else { throw NXError.invalidConfiguration("MPM grid range") }
            }
        }
        self.stateDimension = stateDimension; self.timeStepSeconds = timeStepSeconds; self.particles = particles
    }

    public func addResidual(state: [Float], to residual: inout [Float]) throws {
        guard state.count == stateDimension, residual.count == stateDimension, state.allSatisfy(\.isFinite) else {
            throw NXError.invalidState("MPM residual")
        }
        for particle in particles {
            let F = trialF(particle, state: state)
            let P = try firstPiola(F, material: particle.material)
            let PFt = P.multiplied(by: particle.previousDeformationGradient.transposed)
            let scale = timeStepSeconds * particle.referenceVolume
            for node in particle.stencil {
                let force = scale * PFt.apply(node.gradient)
                residual[node.velocityOffset] += force.x
                residual[node.velocityOffset + 1] += force.y
                residual[node.velocityOffset + 2] += force.z
            }
        }
        guard residual.allSatisfy(\.isFinite) else { throw NXError.numericalBreakdown("MPM residual nonfinite") }
    }

    public func addJacobianVector(state: [Float], vector: [Float], to product: inout [Float]) throws {
        guard state.count == stateDimension, vector.count == stateDimension, product.count == stateDimension,
              state.allSatisfy(\.isFinite), vector.allSatisfy(\.isFinite) else { throw NXError.invalidState("MPM Jv") }
        for particle in particles {
            let F = trialF(particle, state: state)
            let dGrad = velocityGradient(particle, vector: vector)
            let dFPerVelocity = dGrad.multiplied(by: particle.previousDeformationGradient)
            let dP = try firstPiolaDerivative(F, dF: dFPerVelocity, material: particle.material)
            let dPFt = dP.multiplied(by: particle.previousDeformationGradient.transposed)
            let scale = timeStepSeconds * timeStepSeconds * particle.referenceVolume
            for node in particle.stencil {
                let value = scale * dPFt.apply(node.gradient)
                product[node.velocityOffset] += value.x
                product[node.velocityOffset + 1] += value.y
                product[node.velocityOffset + 2] += value.z
            }
        }
        guard product.allSatisfy(\.isFinite) else { throw NXError.numericalBreakdown("MPM Jv nonfinite") }
    }

    public func admissibleStep(state: [Float], direction: [Float]) throws -> NXGeometricCertificate {
        guard state.count == stateDimension, direction.count == stateDimension,
              state.allSatisfy(\.isFinite), direction.allSatisfy(\.isFinite) else { throw NXError.invalidState("MPM safe step") }
        var safe: Float = 1
        var minJ = Float.greatestFiniteMagnitude
        var minVolume = Float.greatestFiniteMagnitude
        for particle in particles {
            let currentJ = trialF(particle, state: state).determinant
            guard currentJ.isFinite, currentJ > particle.material.minimumDeterminant else {
                return NXGeometricCertificate(minimumDistance: 1, minimumDeterminantF: currentJ,
                    minimumVolume: particle.referenceVolume * currentJ, minimumRodLength: 1,
                    finite: currentJ.isFinite, safeStep: 0)
            }
            let fullJ = trialF(particle, state: state, direction: direction, alpha: 1).determinant
            if !fullJ.isFinite || fullJ <= particle.material.minimumDeterminant {
                var low: Float = 0, high: Float = safe
                for _ in 0..<28 {
                    let middle = 0.5 * (low + high)
                    let j = trialF(particle, state: state, direction: direction, alpha: middle).determinant
                    if j.isFinite && j > particle.material.minimumDeterminant { low = middle } else { high = middle }
                }
                safe = min(safe, low * 0.95)
            }
        }
        for particle in particles {
            let j = trialF(particle, state: state, direction: direction, alpha: safe).determinant
            minJ = min(minJ, j); minVolume = min(minVolume, particle.referenceVolume * j)
        }
        return NXGeometricCertificate(minimumDistance: 1, minimumDeterminantF: minJ,
            minimumVolume: minVolume, minimumRodLength: 1,
            finite: minJ.isFinite && minVolume.isFinite, safeStep: safe)
    }

    private func trialF(_ particle: NXMPMParticle, state: [Float], direction: [Float]? = nil,
                        alpha: Float = 0) -> NXMatrix3 {
        var gradient = velocityGradient(particle, vector: state)
        if let direction { gradient = gradient + alpha * velocityGradient(particle, vector: direction) }
        return (NXMatrix3.identity + timeStepSeconds * gradient).multiplied(by: particle.previousDeformationGradient)
    }

    private func velocityGradient(_ particle: NXMPMParticle, vector: [Float]) -> NXMatrix3 {
        var result = NXMatrix3.zero
        for node in particle.stencil {
            let o = node.velocityOffset
            let velocity = SIMD3<Float>(vector[o], vector[o + 1], vector[o + 2])
            result = result + NXMatrix3.outer(velocity, node.gradient)
        }
        return result
    }

    private func firstPiola(_ F: NXMatrix3, material: NXNeoHookeanMaterial) throws -> NXMatrix3 {
        let J = F.determinant
        guard J.isFinite, J > material.minimumDeterminant else { throw NXError.geometryRejected }
        let invT = try F.inverse().transposed
        return material.shearModulus * F
            + (-material.shearModulus + material.lameLambda * log(J)) * invT
    }

    private func firstPiolaDerivative(_ F: NXMatrix3, dF: NXMatrix3,
                                      material: NXNeoHookeanMaterial) throws -> NXMatrix3 {
        let J = F.determinant
        guard J.isFinite, J > material.minimumDeterminant else { throw NXError.geometryRejected }
        let inv = try F.inverse(), invT = inv.transposed
        let trace = NXMatrix3.dot(inv.transposed, dF)
        let sandwich = invT.multiplied(by: dF.transposed).multiplied(by: invT)
        return material.shearModulus * dF
            + material.lameLambda * trace * invT
            + (material.shearModulus - material.lameLambda * log(J)) * sandwich
    }
}
