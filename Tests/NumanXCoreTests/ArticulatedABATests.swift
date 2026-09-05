import XCTest
@testable import NumanXCore

final class ArticulatedABATests: XCTestCase {
    private func diagonalInertia(_ diagonal: [Float]) throws -> NXMatrix6 {
        XCTAssertEqual(diagonal.count, 6)
        var values = Array(repeating: Float.zero, count: 36)
        for i in 0..<6 { values[i * 6 + i] = diagonal[i] }
        return try NXMatrix6(values)
    }

    func testSingleJointABAInvertsMassAction() throws {
        let joint = try NXArticulatedJoint(parent: -1, stateIndex: 0,
            parentMotionTransform: .identity,
            motionSubspace: NXVector6(1, 0, 0, 0, 0, 0),
            bodySpatialInertia: diagonalInertia([2, 3, 4, 5, 6, 7]))
        let articulation = try NXArticulation(joints: [joint])
        XCTAssertEqual(try articulation.massMultiply([3])[0], 6, accuracy: 1e-6)
        XCTAssertEqual(try articulation.inverseMassMultiply([6])[0], 3, accuracy: 1e-6)
    }

    func testTwoJointTreeABAIsInverseOfRNEAMassOperator() throws {
        let inertia = try diagonalInertia([1, 1, 1, 1, 1, 1])
        let first = try NXArticulatedJoint(parent: -1, stateIndex: 0,
            parentMotionTransform: .identity,
            motionSubspace: NXVector6(1, 0, 0, 0, 0, 0), bodySpatialInertia: inertia)
        let second = try NXArticulatedJoint(parent: 0, stateIndex: 1,
            parentMotionTransform: .identity,
            motionSubspace: NXVector6(1, 0, 0, 0, 0, 0), bodySpatialInertia: inertia)
        let articulation = try NXArticulation(joints: [first, second])
        for source: [Float] in [[1, 0], [0, 1], [0.3, -0.8], [2, 3]] {
            let tau = try articulation.massMultiply(source)
            let recovered = try articulation.inverseMassMultiply(tau)
            XCTAssertEqual(recovered[0], source[0], accuracy: 1e-5)
            XCTAssertEqual(recovered[1], source[1], accuracy: 1e-5)
        }
    }

    func testMultiRHSABAProducesIndependentSolutions() throws {
        let inertia = try diagonalInertia([3, 2, 2, 1, 1, 1])
        let joint = try NXArticulatedJoint(parent: -1, stateIndex: 4,
            parentMotionTransform: .identity,
            motionSubspace: NXVector6(1, 0, 0, 0, 0, 0), bodySpatialInertia: inertia)
        let articulation = try NXArticulation(joints: [joint])
        let result = try articulation.inverseMassMultiply(rightHandSides: [[3], [6], [-9]])
        XCTAssertEqual(result.map { $0[0] }, [1, 2, -3])
    }
}
