import Foundation

public enum NXPatchClass: String, Codable, Sendable { case micro, tensor, superPatch }
public enum NXPhysicsBlock: String, Codable, Sendable {
    case rigid, articulated, fem, mpm, rod, transport, pressure, equality, contact
}

public struct NXPatchDescriptor: Codable, Equatable, Sendable {
    public var identifier: UInt32
    public var patchClass: NXPatchClass
    public var physicsBlock: NXPhysicsBlock
    public var unknownIndices: [Int]
    public var constraintIndices: [Int]
    public var preferredPrecision: NXPrecision
    public var conditionIndicator: Float

    public init(identifier: UInt32, patchClass: NXPatchClass, physicsBlock: NXPhysicsBlock,
                unknownIndices: [Int], constraintIndices: [Int] = [],
                preferredPrecision: NXPrecision = .fp32, conditionIndicator: Float = 1) throws {
        guard !unknownIndices.isEmpty, Set(unknownIndices).count == unknownIndices.count,
              Set(constraintIndices).count == constraintIndices.count,
              unknownIndices.allSatisfy({ $0 >= 0 }), constraintIndices.allSatisfy({ $0 >= 0 }),
              conditionIndicator.isFinite, conditionIndicator > 0 else {
            throw NXError.invalidConfiguration("patch descriptor")
        }
        switch patchClass {
        case .micro: guard unknownIndices.count <= 24 else { throw NXError.capacity("micropatch >24") }
        case .tensor: guard (25...160).contains(unknownIndices.count) else { throw NXError.capacity("tensor patch size") }
        case .superPatch: break
        }
        self.identifier = identifier; self.patchClass = patchClass; self.physicsBlock = physicsBlock
        self.unknownIndices = unknownIndices; self.constraintIndices = constraintIndices
        self.preferredPrecision = preferredPrecision; self.conditionIndicator = conditionIndicator
    }
}

public struct NXCohortKey: Hashable, Codable, Sendable {
    public var topologyFingerprint: UInt64
    public var materialFingerprint: UInt64
    public var shapeFingerprint: UInt64
    public var precision: NXPrecision
    public var environmentClass: UInt32
}

public struct NXPatchCohort: Codable, Equatable, Sendable {
    public var key: NXCohortKey
    public var patches: [NXPatchDescriptor]
    public init(key: NXCohortKey, patches: [NXPatchDescriptor]) throws {
        guard !patches.isEmpty, Set(patches.map(\.identifier)).count == patches.count else {
            throw NXError.invalidConfiguration("patch cohort")
        }
        self.key = key; self.patches = patches.sorted { $0.identifier < $1.identifier }
    }
}

public struct NXMixedPrecisionPolicy: Codable, Equatable, Sendable {
    public var maximumRefinementIterations: Int
    public var promotionThreshold: Float
    public var stagnationRatio: Float

    public init(maximumRefinementIterations: Int = 3, promotionThreshold: Float = 64,
                stagnationRatio: Float = 0.85) throws {
        guard maximumRefinementIterations >= 0, promotionThreshold > 1,
              stagnationRatio > 0, stagnationRatio < 1 else {
            throw NXError.invalidConfiguration("mixed precision policy")
        }
        self.maximumRefinementIterations = maximumRefinementIterations
        self.promotionThreshold = promotionThreshold
        self.stagnationRatio = stagnationRatio
    }

    public func authoritativePrecision(for patch: NXPatchDescriptor) -> NXPrecision {
        if patch.conditionIndicator >= promotionThreshold { return .fp32 }
        return patch.preferredPrecision
    }
}

/// CPU reference contract for additive/overlapping patch corrections. Production Metal execution
/// batches same-cohort patches and may use tensor hardware for local approximate inverses, but the
/// returned correction is always promoted to FP32 before it reaches the authoritative Krylov vector.
public protocol NXLocalPatchSolver: Sendable {
    func solve(patch: NXPatchDescriptor, residual: [Float], precision: NXPrecision) throws -> [Float]
}

public struct NXTensorSchurPreconditioner: NXPreconditioner {
    public let dimension: Int
    public let patches: [NXPatchDescriptor]
    public let policy: NXMixedPrecisionPolicy
    public let localSolver: any NXLocalPatchSolver
    public let diagonalFloor: Float

    public init(dimension: Int, patches: [NXPatchDescriptor], policy: NXMixedPrecisionPolicy,
                localSolver: any NXLocalPatchSolver, diagonalFloor: Float = 1e-6) throws {
        guard dimension > 0, diagonalFloor.isFinite, diagonalFloor > 0,
              !patches.isEmpty, patches.allSatisfy({ $0.unknownIndices.allSatisfy { $0 < dimension } }) else {
            throw NXError.invalidConfiguration("Tensor-Schur preconditioner")
        }
        self.dimension = dimension; self.patches = patches; self.policy = policy
        self.localSolver = localSolver; self.diagonalFloor = diagonalFloor
    }

    public func apply(_ residual: [Float], iteration: Int, into correction: inout [Float]) throws {
        guard residual.count == dimension, residual.allSatisfy(\.isFinite), iteration >= 0 else {
            throw NXError.invalidState("Tensor-Schur residual")
        }
        correction = Array(repeating: 0, count: dimension)
        var weights = Array(repeating: Float.zero, count: dimension)
        for patch in patches {
            let localResidual = patch.unknownIndices.map { residual[$0] }
            var precision = policy.authoritativePrecision(for: patch)
            var local = try localSolver.solve(patch: patch, residual: localResidual, precision: precision)
            guard local.count == patch.unknownIndices.count, local.allSatisfy(\.isFinite) else {
                throw NXError.numericalBreakdown("local patch correction")
            }
            if precision != .fp32 && policy.maximumRefinementIterations > 0 {
                // Local solvers may encode their own operator; condition-triggered promotion is a hard
                // fallback because a low-precision correction is advisory, never authoritative state.
                let indicator = patch.conditionIndicator
                if indicator >= policy.promotionThreshold * policy.stagnationRatio {
                    precision = .fp32
                    local = try localSolver.solve(patch: patch, residual: localResidual, precision: .fp32)
                }
            }
            for (localIndex, globalIndex) in patch.unknownIndices.enumerated() {
                correction[globalIndex] += local[localIndex]
                weights[globalIndex] += 1
            }
        }
        for index in 0..<dimension {
            if weights[index] > 0 { correction[index] /= weights[index] }
            else { correction[index] = residual[index] }
            guard correction[index].isFinite else {
                throw NXError.numericalBreakdown("assembled patch correction")
            }
        }
    }
}

/// Deterministic dense local reference solve used for micropatches and validation. Matrix is supplied
/// row-major and solved in FP32 with partial pivoting; tensor/superpatch production backends can
/// implement the same protocol using batched GPU kernels, ABA, banded CR, multigrid or local Krylov.
public struct NXDensePatchSolver: NXLocalPatchSolver {
    public var matrices: [UInt32: [Float]]
    public init(matrices: [UInt32: [Float]]) { self.matrices = matrices }

    public func solve(patch: NXPatchDescriptor, residual: [Float], precision: NXPrecision) throws -> [Float] {
        let n = residual.count
        guard let stored = matrices[patch.identifier], stored.count == n * n else {
            throw NXError.invalidConfiguration("missing dense patch matrix")
        }
        var a = stored
        var b = residual
        for k in 0..<n {
            var pivot = k
            var largest = abs(a[k * n + k])
            if k + 1 < n {
                for row in (k + 1)..<n where abs(a[row * n + k]) > largest {
                    largest = abs(a[row * n + k]); pivot = row
                }
            }
            guard largest.isFinite, largest > 1e-12 else { throw NXError.numericalBreakdown("singular patch") }
            if pivot != k {
                for column in k..<n { a.swapAt(k * n + column, pivot * n + column) }
                b.swapAt(k, pivot)
            }
            let diagonal = a[k * n + k]
            for row in (k + 1)..<n {
                let factor = a[row * n + k] / diagonal
                a[row * n + k] = 0
                if k + 1 < n {
                    for column in (k + 1)..<n { a[row * n + column] -= factor * a[k * n + column] }
                }
                b[row] -= factor * b[k]
            }
        }
        var x = Array(repeating: Float.zero, count: n)
        for row in stride(from: n - 1, through: 0, by: -1) {
            var value = b[row]
            if row + 1 < n {
                for column in (row + 1)..<n { value -= a[row * n + column] * x[column] }
            }
            x[row] = value / a[row * n + row]
            guard x[row].isFinite else { throw NXError.numericalBreakdown("dense patch backsolve") }
        }
        return x
    }
}
