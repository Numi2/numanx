import Foundation

public struct NXNeoHookeanMaterial: Codable, Equatable, Sendable {
    public var shearModulus: Float
    public var lameLambda: Float
    public var minimumDeterminant: Float

    public init(shearModulus: Float, lameLambda: Float, minimumDeterminant: Float = 0.05) throws {
        guard shearModulus.isFinite, shearModulus > 0, lameLambda.isFinite, lameLambda >= 0,
              minimumDeterminant.isFinite, minimumDeterminant > 0, minimumDeterminant < 1 else {
            throw NXError.invalidConfiguration("neo-hookean material")
        }
        self.shearModulus = shearModulus; self.lameLambda = lameLambda
        self.minimumDeterminant = minimumDeterminant
    }
}

public struct NXTetrahedron: Codable, Equatable, Sendable {
    public var velocityOffsets: SIMD4<Int32>
    public var previousPositions: [SIMD3<Float>]
    public var inverseRestMatrix: NXMatrix3
    public var restVolume: Float
    public var material: NXNeoHookeanMaterial

    public init(velocityOffsets: SIMD4<Int32>, previousPositions: [SIMD3<Float>],
                inverseRestMatrix: NXMatrix3, restVolume: Float,
                material: NXNeoHookeanMaterial) throws {
        guard previousPositions.count == 4, velocityOffsets.min() >= 0,
              restVolume.isFinite, restVolume > 0, inverseRestMatrix.allFinite,
              abs(inverseRestMatrix.determinant) > 1e-12,
              previousPositions.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }) else {
            throw NXError.invalidConfiguration("tetrahedral element")
        }
        self.velocityOffsets = velocityOffsets; self.previousPositions = previousPositions
        self.inverseRestMatrix = inverseRestMatrix; self.restVolume = restVolume; self.material = material
    }
}

/// Implicit backward-Euler internal-force contribution for compressible Neo-Hookean tetrahedra.
/// State slots are nodal velocities. Current positions are x_n + dt*v, so the Jv contribution
/// carries dt²*dP. Mass and external impulses remain separate monolithic contributions.
public struct NXTetrahedralFEMContribution: NXPhysicsContribution {
    public let name = "implicit-neo-hookean-tetrahedra"
    public let stateDimension: Int
    public let timeStepSeconds: Float
    public let elements: [NXTetrahedron]

    public init(stateDimension: Int, timeStepSeconds: Float, elements: [NXTetrahedron]) throws {
        guard stateDimension > 0, timeStepSeconds.isFinite, timeStepSeconds > 0, !elements.isEmpty else {
            throw NXError.invalidConfiguration("tetrahedral FEM")
        }
        for element in elements {
            for lane in 0..<4 {
                let offset = Int(element.velocityOffsets[lane])
                guard offset + 3 <= stateDimension else { throw NXError.invalidConfiguration("tetra velocity range") }
            }
        }
        self.stateDimension = stateDimension; self.timeStepSeconds = timeStepSeconds; self.elements = elements
    }

    public func addResidual(state: [Float], to residual: inout [Float]) throws {
        guard state.count == stateDimension, residual.count == stateDimension, state.allSatisfy(\.isFinite) else {
            throw NXError.invalidState("FEM residual state")
        }
        for element in elements {
            let positions = currentPositions(element, state: state, alpha: 1, direction: nil)
            let F = deformationGradient(element, positions: positions)
            let P = try firstPiola(F: F, material: element.material)
            let gradients = shapeGradients(element)
            for node in 0..<4 {
                let forceResidual = timeStepSeconds * element.restVolume * P.apply(gradients[node])
                let offset = Int(element.velocityOffsets[node])
                residual[offset] += forceResidual.x
                residual[offset + 1] += forceResidual.y
                residual[offset + 2] += forceResidual.z
            }
        }
        guard residual.allSatisfy(\.isFinite) else { throw NXError.numericalBreakdown("FEM residual nonfinite") }
    }

    public func addJacobianVector(state: [Float], vector: [Float], to product: inout [Float]) throws {
        guard state.count == stateDimension, vector.count == stateDimension, product.count == stateDimension,
              state.allSatisfy(\.isFinite), vector.allSatisfy(\.isFinite) else {
            throw NXError.invalidState("FEM Jv state")
        }
        for element in elements {
            let positions = currentPositions(element, state: state, alpha: 1, direction: nil)
            let F = deformationGradient(element, positions: positions)
            let dF = directionalDeformationGradient(element, direction: vector)
            let dP = try firstPiolaDirectionalDerivative(F: F, dF: dF, material: element.material)
            let gradients = shapeGradients(element)
            let scale = timeStepSeconds * timeStepSeconds * element.restVolume
            for node in 0..<4 {
                let value = scale * dP.apply(gradients[node])
                let offset = Int(element.velocityOffsets[node])
                product[offset] += value.x
                product[offset + 1] += value.y
                product[offset + 2] += value.z
            }
        }
        guard product.allSatisfy(\.isFinite) else { throw NXError.numericalBreakdown("FEM Jv nonfinite") }
    }

    public func admissibleStep(state: [Float], direction: [Float]) throws -> NXGeometricCertificate {
        guard state.count == stateDimension, direction.count == stateDimension,
              state.allSatisfy(\.isFinite), direction.allSatisfy(\.isFinite) else {
            throw NXError.invalidState("FEM safe-step state")
        }
        var safe: Float = 1
        var minimumJ = Float.greatestFiniteMagnitude
        for element in elements {
            let current = deformationGradient(element,
                positions: currentPositions(element, state: state, alpha: 1, direction: nil))
            let currentJ = current.determinant
            guard currentJ.isFinite, currentJ > element.material.minimumDeterminant else {
                return NXGeometricCertificate(minimumDistance: 1, minimumDeterminantF: currentJ,
                    minimumVolume: element.restVolume * currentJ, minimumRodLength: 1,
                    finite: currentJ.isFinite, safeStep: 0)
            }
            let candidatePositions = currentPositions(element, state: state, alpha: 1, direction: direction)
            let candidateJ = deformationGradient(element, positions: candidatePositions).determinant
            minimumJ = min(minimumJ, candidateJ)
            if !candidateJ.isFinite || candidateJ <= element.material.minimumDeterminant {
                var low: Float = 0, high: Float = safe
                for _ in 0..<24 {
                    let middle = 0.5 * (low + high)
                    let positions = currentPositions(element, state: state, alpha: middle, direction: direction)
                    let j = deformationGradient(element, positions: positions).determinant
                    if j.isFinite && j > element.material.minimumDeterminant { low = middle } else { high = middle }
                }
                safe = min(safe, max(0, low * 0.95))
            }
        }
        return NXGeometricCertificate(minimumDistance: 1, minimumDeterminantF: minimumJ,
            minimumVolume: elements.map({ $0.restVolume }).min() ?? 1,
            minimumRodLength: 1, finite: minimumJ.isFinite, safeStep: safe)
    }

    private func currentPositions(_ element: NXTetrahedron, state: [Float], alpha: Float,
                                  direction: [Float]?) -> [SIMD3<Float>] {
        var result = element.previousPositions
        for node in 0..<4 {
            let offset = Int(element.velocityOffsets[node])
            var v = SIMD3<Float>(state[offset], state[offset + 1], state[offset + 2])
            if let direction {
                v += alpha * SIMD3<Float>(direction[offset], direction[offset + 1], direction[offset + 2])
            }
            result[node] += timeStepSeconds * v
        }
        return result
    }

    private func deformationGradient(_ element: NXTetrahedron, positions: [SIMD3<Float>]) -> NXMatrix3 {
        let ds = NXMatrix3(positions[1] - positions[0], positions[2] - positions[0], positions[3] - positions[0])
        return ds.multiplied(by: element.inverseRestMatrix)
    }

    private func directionalDeformationGradient(_ element: NXTetrahedron, direction: [Float]) -> NXMatrix3 {
        func dv(_ node: Int) -> SIMD3<Float> {
            let o = Int(element.velocityOffsets[node])
            return SIMD3<Float>(direction[o], direction[o + 1], direction[o + 2])
        }
        let dds = NXMatrix3(dv(1) - dv(0), dv(2) - dv(0), dv(3) - dv(0))
        return dds.multiplied(by: element.inverseRestMatrix)
    }

    private func shapeGradients(_ element: NXTetrahedron) -> [SIMD3<Float>] {
        // Reference gradients: columns of Dm^{-T} for N1,N2,N3; N0 = -sum.
        let invT = element.inverseRestMatrix.transposed
        let g1 = invT.c0, g2 = invT.c1, g3 = invT.c2
        return [-(g1 + g2 + g3), g1, g2, g3]
    }

    private func firstPiola(F: NXMatrix3, material: NXNeoHookeanMaterial) throws -> NXMatrix3 {
        let J = F.determinant
        guard J.isFinite, J > material.minimumDeterminant else { throw NXError.geometryRejected }
        let invT = try F.inverse().transposed
        return material.shearModulus * F
             + (-material.shearModulus + material.lameLambda * log(J)) * invT
    }

    private func firstPiolaDirectionalDerivative(F: NXMatrix3, dF: NXMatrix3,
                                                 material: NXNeoHookeanMaterial) throws -> NXMatrix3 {
        let J = F.determinant
        guard J.isFinite, J > material.minimumDeterminant else { throw NXError.geometryRejected }
        let inv = try F.inverse()
        let invT = inv.transposed
        let trace = NXMatrix3.dot(inv.transposed, dF)
        let sandwich = invT.multiplied(by: dF.transposed).multiplied(by: invT)
        let coefficient = material.shearModulus - material.lameLambda * log(J)
        return material.shearModulus * dF
             + material.lameLambda * trace * invT
             + coefficient * sandwich
    }
}
