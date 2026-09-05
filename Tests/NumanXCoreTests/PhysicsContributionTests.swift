import XCTest
@testable import NumanXCore

final class PhysicsContributionTests: XCTestCase {
    func testRigidMassMatrixIsSymmetricAndPositiveForPhysicalInertia() throws {
        let inertia = try NXSpatialInertia(mass: 2, centerOfMass: SIMD3<Float>(0.2, -0.1, 0.3),
            inertia: NXVector6(0.5, 0.7, 0.8, 0.02, -0.01, 0.03))
        let m = inertia.generalizedMassMatrix()
        for r in 0..<6 { for c in 0..<6 { XCTAssertEqual(m[r * 6 + c], m[c * 6 + r], accuracy: 1e-6) } }
        for basis in 0..<6 {
            XCTAssertGreaterThan(m[basis * 6 + basis], 0)
        }
    }

    func testRigidResidualUsesGeneralizedInertia() throws {
        let inertia = try NXSpatialInertia(mass: 2, inertia: NXVector6(1, 1, 1))
        let body = try NXRigidBodyStep(velocityOffset: 0, inertia: inertia,
            previousVelocity: .zero, externalImpulse: NXVector6(2, 0, 0, 0, 0, 0))
        let contribution = try NXRigidDynamicsContribution(stateDimension: 6, bodies: [body])
        var residual = Array(repeating: Float.zero, count: 6)
        try contribution.addResidual(state: [1, 0, 0, 0, 0, 0], to: &residual)
        XCTAssertEqual(residual[0], 0, accuracy: 1e-6)
    }

    func testContactAnalyticalJvMatchesDirectionalFiniteDifferenceAwayFromConeBoundary() throws {
        let n = try NXJacobianEntry(index: 0, coefficient: 1)
        let t1 = try NXJacobianEntry(index: 1, coefficient: 1)
        let t2 = try NXJacobianEntry(index: 2, coefficient: 1)
        let law = try NXCoulombContact(frictionCoefficient: 0.6, normalCompliance: 0.01,
            tangentialCompliance: 0.02, stabilizationVelocity: -0.1)
        let row = try NXContactRow(multiplierOffset: 3, normal: [n], tangent1: [t1], tangent2: [t2], law: law)
        let contribution = try NXContactContribution(stateDimension: 6, rows: [row])
        let state: [Float] = [-0.4, 0.2, -0.1, 1.2, 0.25, -0.15]
        let direction: [Float] = [0.3, -0.1, 0.2, 0.05, -0.04, 0.03]
        var analytical = Array(repeating: Float.zero, count: 6)
        try contribution.addJacobianVector(state: state, vector: direction, to: &analytical)
        let epsilon: Float = 1e-4
        var plus = state, minus = state
        for i in state.indices { plus[i] += epsilon * direction[i]; minus[i] -= epsilon * direction[i] }
        var rp = Array(repeating: Float.zero, count: 6), rm = rp
        try contribution.addResidual(state: plus, to: &rp)
        try contribution.addResidual(state: minus, to: &rm)
        for i in state.indices {
            let finite = (rp[i] - rm[i]) / (2 * epsilon)
            XCTAssertEqual(analytical[i], finite, accuracy: 2e-3)
        }
    }

    func testContactAddsEqualJacobianTransposeImpulseToGeneralizedResidual() throws {
        let law = try NXCoulombContact(frictionCoefficient: 0.5)
        let row = try NXContactRow(multiplierOffset: 2,
            normal: [NXJacobianEntry(index: 0, coefficient: 2)],
            tangent1: [NXJacobianEntry(index: 1, coefficient: 3)], tangent2: [], law: law)
        let contribution = try NXContactContribution(stateDimension: 5, rows: [row])
        var residual = Array(repeating: Float.zero, count: 5)
        try contribution.addResidual(state: [0, 0, 4, 5, 0], to: &residual)
        XCTAssertEqual(residual[0], 8, accuracy: 1e-6)
        XCTAssertEqual(residual[1], 15, accuracy: 1e-6)
    }
}
