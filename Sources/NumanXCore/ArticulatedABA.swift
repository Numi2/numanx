import Foundation

public struct NXArticulatedJoint: Codable, Equatable, Sendable {
    /// -1 denotes the world; otherwise parent must precede this joint in topological order.
    public var parent: Int
    public var stateIndex: Int
    /// Motion transform maps parent spatial acceleration into this body's coordinates.
    public var parentMotionTransform: NXMatrix6
    public var motionSubspace: NXVector6
    public var bodySpatialInertia: NXMatrix6
    public var previousJointVelocity: Float
    public var externalGeneralizedImpulse: Float

    public init(parent: Int, stateIndex: Int, parentMotionTransform: NXMatrix6,
                motionSubspace: NXVector6, bodySpatialInertia: NXMatrix6,
                previousJointVelocity: Float = 0,
                externalGeneralizedImpulse: Float = 0) throws {
        guard parent >= -1, stateIndex >= 0, motionSubspace.allFinite,
              previousJointVelocity.isFinite, externalGeneralizedImpulse.isFinite else {
            throw NXError.invalidConfiguration("articulated joint")
        }
        self.parent = parent; self.stateIndex = stateIndex
        self.parentMotionTransform = parentMotionTransform; self.motionSubspace = motionSubspace
        self.bodySpatialInertia = bodySpatialInertia
        self.previousJointVelocity = previousJointVelocity
        self.externalGeneralizedImpulse = externalGeneralizedImpulse
    }
}

/// One-DoF tree articulation. Multi-DoF joints are represented by consecutive virtual joints with
/// zero inter-joint translation, which keeps the ABA kernels uniform and cohortable on GPU.
public struct NXArticulation: Codable, Equatable, Sendable {
    public var joints: [NXArticulatedJoint]

    public init(joints: [NXArticulatedJoint]) throws {
        guard !joints.isEmpty, Set(joints.map(\.stateIndex)).count == joints.count else {
            throw NXError.invalidConfiguration("articulation")
        }
        for (index, joint) in joints.enumerated() {
            guard joint.parent < index,
                  joint.bodySpatialInertia.values.allSatisfy(\.isFinite),
                  joint.parentMotionTransform.values.allSatisfy(\.isFinite),
                  joint.motionSubspace.dot(joint.bodySpatialInertia.apply(joint.motionSubspace)) > 0 else {
                throw NXError.invalidConfiguration("articulation topology/inertia")
            }
        }
        self.joints = joints
    }

    /// Recursive Newton-Euler mass action M*qdd with zero velocity bias/gravity.
    public func massMultiply(_ qdd: [Float]) throws -> [Float] {
        guard qdd.count == joints.count, qdd.allSatisfy(\.isFinite) else {
            throw NXError.invalidState("articulated mass input")
        }
        var acceleration = Array(repeating: NXVector6.zero, count: joints.count)
        var force = Array(repeating: NXVector6.zero, count: joints.count)
        for i in joints.indices {
            let joint = joints[i]
            let parentAcceleration = joint.parent >= 0 ? acceleration[joint.parent] : .zero
            acceleration[i] = joint.parentMotionTransform.apply(parentAcceleration)
                + qdd[i] * joint.motionSubspace
            force[i] = joint.bodySpatialInertia.apply(acceleration[i])
        }
        var tau = Array(repeating: Float.zero, count: joints.count)
        for i in joints.indices.reversed() {
            let joint = joints[i]
            tau[i] = joint.motionSubspace.dot(force[i])
            if joint.parent >= 0 {
                force[joint.parent] = force[joint.parent]
                    + joint.parentMotionTransform.transposed.apply(force[i])
            }
        }
        guard tau.allSatisfy(\.isFinite) else { throw NXError.numericalBreakdown("articulated mass action") }
        return tau
    }

    /// Articulated-body algorithm for M^-1*tau with no bias terms. This is suitable for the SPD
    /// surrogate/preconditioner and supports repeated right-hand sides without materializing M.
    public func inverseMassMultiply(_ tau: [Float]) throws -> [Float] {
        guard tau.count == joints.count, tau.allSatisfy(\.isFinite) else {
            throw NXError.invalidState("ABA right-hand side")
        }
        var articulatedInertia = joints.map(\.bodySpatialInertia)
        var articulatedBias = Array(repeating: NXVector6.zero, count: joints.count)
        var U = Array(repeating: NXVector6.zero, count: joints.count)
        var d = Array(repeating: Float.zero, count: joints.count)
        var u = Array(repeating: Float.zero, count: joints.count)

        for i in joints.indices.reversed() {
            let joint = joints[i]
            U[i] = articulatedInertia[i].apply(joint.motionSubspace)
            d[i] = joint.motionSubspace.dot(U[i])
            guard d[i].isFinite, d[i] > 1e-10 else {
                throw NXError.numericalBreakdown("ABA singular joint inertia")
            }
            u[i] = tau[i] - joint.motionSubspace.dot(articulatedBias[i])
            if joint.parent >= 0 {
                let reduced = articulatedInertia[i] - (1 / d[i]) * NXMatrix6.outer(U[i], U[i])
                let propagatedBias = articulatedBias[i] + (u[i] / d[i]) * U[i]
                let X = joint.parentMotionTransform
                articulatedInertia[joint.parent] = articulatedInertia[joint.parent]
                    + X.transposed.multiplied(by: reduced).multiplied(by: X)
                articulatedBias[joint.parent] = articulatedBias[joint.parent]
                    + X.transposed.apply(propagatedBias)
            }
        }

        var acceleration = Array(repeating: NXVector6.zero, count: joints.count)
        var qdd = Array(repeating: Float.zero, count: joints.count)
        for i in joints.indices {
            let joint = joints[i]
            let parentAcceleration = joint.parent >= 0 ? acceleration[joint.parent] : .zero
            let transformed = joint.parentMotionTransform.apply(parentAcceleration)
            qdd[i] = (u[i] - U[i].dot(transformed)) / d[i]
            acceleration[i] = transformed + qdd[i] * joint.motionSubspace
        }
        guard qdd.allSatisfy(\.isFinite) else { throw NXError.numericalBreakdown("ABA result") }
        return qdd
    }

    public func inverseMassMultiply(rightHandSides: [[Float]]) throws -> [[Float]] {
        try rightHandSides.map { try inverseMassMultiply($0) }
    }
}

public struct NXArticulatedDynamicsContribution: NXPhysicsContribution {
    public let name = "articulated-rnea-mass"
    public let stateDimension: Int
    public let articulation: NXArticulation

    public init(stateDimension: Int, articulation: NXArticulation) throws {
        guard stateDimension > 0,
              articulation.joints.allSatisfy({ $0.stateIndex < stateDimension }) else {
            throw NXError.invalidConfiguration("articulated contribution")
        }
        self.stateDimension = stateDimension; self.articulation = articulation
    }

    public func addResidual(state: [Float], to residual: inout [Float]) throws {
        guard state.count == stateDimension, residual.count == stateDimension else {
            throw NXError.invalidState("articulated residual")
        }
        let delta = articulation.joints.map { state[$0.stateIndex] - $0.previousJointVelocity }
        let inertia = try articulation.massMultiply(delta)
        for i in articulation.joints.indices {
            let joint = articulation.joints[i]
            residual[joint.stateIndex] += inertia[i] - joint.externalGeneralizedImpulse
        }
    }

    public func addJacobianVector(state: [Float], vector: [Float], to product: inout [Float]) throws {
        guard state.count == stateDimension, vector.count == stateDimension, product.count == stateDimension else {
            throw NXError.invalidState("articulated Jv")
        }
        let local = articulation.joints.map { vector[$0.stateIndex] }
        let inertia = try articulation.massMultiply(local)
        for i in articulation.joints.indices { product[articulation.joints[i].stateIndex] += inertia[i] }
    }

    public func admissibleStep(state: [Float], direction: [Float]) throws -> NXGeometricCertificate {
        guard state.count == stateDimension, direction.count == stateDimension,
              state.allSatisfy(\.isFinite), direction.allSatisfy(\.isFinite) else {
            throw NXError.invalidState("articulated safe step")
        }
        return NXGeometricCertificate(minimumDistance: 1, minimumDeterminantF: 1,
            minimumVolume: 1, minimumRodLength: 1, finite: true, safeStep: 1)
    }
}

/// Exact articulated inverse on its joint subspace; untouched state entries pass through. This is
/// the multi-RHS ABA building block used by Tensor-Schur patches around articulated subtrees.
public struct NXArticulatedABAPreconditioner: NXPreconditioner {
    public let dimension: Int
    public let articulation: NXArticulation

    public init(dimension: Int, articulation: NXArticulation) throws {
        guard dimension > 0, articulation.joints.allSatisfy({ $0.stateIndex < dimension }) else {
            throw NXError.invalidConfiguration("ABA preconditioner")
        }
        self.dimension = dimension; self.articulation = articulation
    }

    public func apply(_ residual: [Float], iteration: Int, into correction: inout [Float]) throws {
        guard residual.count == dimension, residual.allSatisfy(\.isFinite), iteration >= 0 else {
            throw NXError.invalidState("ABA preconditioner residual")
        }
        correction = residual
        let rhs = articulation.joints.map { residual[$0.stateIndex] }
        let local = try articulation.inverseMassMultiply(rhs)
        for i in articulation.joints.indices { correction[articulation.joints[i].stateIndex] = local[i] }
    }
}
