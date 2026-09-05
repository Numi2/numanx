import Foundation

public struct NXContactHistoryEntry: Codable, Equatable, Sendable {
    public var pair: NXCollisionPair
    public var normal: SIMD3<Float>
    public var tangent1: SIMD3<Float>
    public var tangent2: SIMD3<Float>
    public var multiplier: SIMD3<Float>
    public var lastGeneration: UInt64
    public var lastTimeNanoseconds: UInt64

    public init(pair: NXCollisionPair, normal: SIMD3<Float>, tangent1: SIMD3<Float>, tangent2: SIMD3<Float>,
                multiplier: SIMD3<Float>, lastGeneration: UInt64, lastTimeNanoseconds: UInt64) throws {
        guard [normal.x,normal.y,normal.z,tangent1.x,tangent1.y,tangent1.z,tangent2.x,tangent2.y,tangent2.z,
               multiplier.x,multiplier.y,multiplier.z].allSatisfy(\.isFinite), multiplier.x >= 0 else {
            throw NXError.invalidState("contact history")
        }
        self.pair = pair; self.normal = normal; self.tangent1 = tangent1; self.tangent2 = tangent2
        self.multiplier = multiplier; self.lastGeneration = lastGeneration; self.lastTimeNanoseconds = lastTimeNanoseconds
    }
}

/// Serializable history is part of authoritative physical state. Updating it before root publication
/// would leak rejected contact impulses, so callers update a shadow copy and publish it with the body.
public struct NXContactHistory: Codable, Equatable, Sendable {
    public var entries: [NXContactHistoryEntry]
    public var maximumEntries: Int
    public var maximumGenerationAge: UInt64

    public init(entries: [NXContactHistoryEntry] = [], maximumEntries: Int = 65_536,
                maximumGenerationAge: UInt64 = 120) throws {
        guard maximumEntries > 0, entries.count <= maximumEntries,
              Set(entries.map(\.pair)).count == entries.count else { throw NXError.invalidConfiguration("contact history") }
        self.entries = entries.sorted { $0.pair < $1.pair }
        self.maximumEntries = maximumEntries; self.maximumGenerationAge = maximumGenerationAge
    }

    public func warmStart(pair: NXCollisionPair, newNormal: SIMD3<Float>,
                          newTangent1: SIMD3<Float>, newTangent2: SIMD3<Float>,
                          generation: UInt64) -> SIMD3<Float> {
        guard let entry = entries.first(where: { $0.pair == pair }), generation >= entry.lastGeneration,
              generation - entry.lastGeneration <= maximumGenerationAge else { return .zero }
        let worldTangential = entry.multiplier.y * entry.tangent1 + entry.multiplier.z * entry.tangent2
        var result = SIMD3<Float>(max(0, entry.multiplier.x), dot(worldTangential, newTangent1), dot(worldTangential, newTangent2))
        // Large normal changes are likely a different manifold feature; keep only normal pressure.
        if dot(entry.normal, newNormal) < 0.5 { result.y = 0; result.z = 0 }
        return result
    }

    public mutating func publish(pair: NXCollisionPair, geometry: NXSphereContactGeometry,
                                 multiplier: SIMD3<Float>, generation: UInt64,
                                 timeNanoseconds: UInt64) throws {
        let t1 = contactTangent(geometry.normal), t2 = cross(geometry.normal, t1)
        let item = try NXContactHistoryEntry(pair: pair, normal: geometry.normal, tangent1: t1, tangent2: t2,
            multiplier: multiplier, lastGeneration: generation, lastTimeNanoseconds: timeNanoseconds)
        if let index = entries.firstIndex(where: { $0.pair == pair }) { entries[index] = item }
        else { entries.append(item) }
        entries.removeAll { entry in
            generation >= entry.lastGeneration && generation - entry.lastGeneration > maximumGenerationAge
        }
        if entries.count > maximumEntries {
            entries.sort {
                if $0.lastGeneration != $1.lastGeneration { return $0.lastGeneration > $1.lastGeneration }
                return $0.pair < $1.pair
            }
            entries.removeLast(entries.count - maximumEntries)
        }
        entries.sort { $0.pair < $1.pair }
    }
}

public struct NXActiveContactSet: Sendable {
    public var pairs: [NXCollisionPair]
    public var geometries: [NXSphereContactGeometry]
    public var rows: [NXContactRow]
    public var warmStartMultipliers: [SIMD3<Float>]

    public func seed(state: inout [Float]) throws {
        guard rows.count == warmStartMultipliers.count else { throw NXError.invalidState("contact seed dimensions") }
        for (row, multiplier) in zip(rows, warmStartMultipliers) {
            guard row.multiplierOffset <= state.count - 3 else { throw NXError.invalidState("contact seed state") }
            state[row.multiplierOffset] = multiplier.x
            state[row.multiplierOffset + 1] = multiplier.y
            state[row.multiplierOffset + 2] = multiplier.z
        }
    }
}

public enum NXContactManifoldBuilder {
    public static func build(shapes: [NXRigidSphere], state: [Float],
                             timeStepSeconds: Float, contactMultiplierRange: Range<Int>,
                             history: NXContactHistory, generation: UInt64,
                             speculativeMargin: Float = 0.002,
                             stabilizationFraction: Float = 0.2,
                             compliance: Float = 0) throws -> NXActiveContactSet {
        guard contactMultiplierRange.lowerBound >= 0,
              contactMultiplierRange.upperBound <= state.count,
              contactMultiplierRange.count.isMultiple(of: 3),
              speculativeMargin.isFinite, speculativeMargin >= 0 else {
            throw NXError.invalidConfiguration("contact manifold allocation")
        }
        let candidates = try NXSweepAndPruneBroadphase.candidates(shapes: shapes, state: state,
            timeStepSeconds: timeStepSeconds, padding: speculativeMargin)
        let lookup = Dictionary(uniqueKeysWithValues: shapes.map { ($0.identifier, $0) })
        var active: [(NXCollisionPair, NXSphereContactGeometry, NXRigidSphere, NXRigidSphere)] = []
        for pair in candidates {
            guard let a = lookup[pair.first], let b = lookup[pair.second] else {
                throw NXError.invalidState("broadphase pair references missing shape")
            }
            let geometry = try NXSphereSphereCCD.geometry(a: a, b: b, state: state, timeStepSeconds: timeStepSeconds)
            if geometry.gap <= speculativeMargin || geometry.timeOfImpactFraction != nil {
                active.append((pair, geometry, a, b))
            }
        }
        active.sort { $0.0 < $1.0 }
        guard active.count <= contactMultiplierRange.count / 3 else { throw NXError.capacity("contact multiplier capacity") }
        var rows: [NXContactRow] = [], geometries: [NXSphereContactGeometry] = [], warm: [SIMD3<Float>] = []
        rows.reserveCapacity(active.count); geometries.reserveCapacity(active.count); warm.reserveCapacity(active.count)
        for (index, item) in active.enumerated() {
            let offset = contactMultiplierRange.lowerBound + 3 * index
            let row = try NXRigidSphereContactBuilder.contactRow(a: item.2, b: item.3, state: state,
                multiplierOffset: offset, timeStepSeconds: timeStepSeconds,
                stabilizationFraction: stabilizationFraction, compliance: compliance)
            let t1 = contactTangent(item.1.normal), t2 = cross(item.1.normal, t1)
            rows.append(row); geometries.append(item.1)
            warm.append(history.warmStart(pair: item.0, newNormal: item.1.normal,
                newTangent1: t1, newTangent2: t2, generation: generation))
        }
        return NXActiveContactSet(pairs: active.map(\.0), geometries: geometries,
            rows: rows, warmStartMultipliers: warm)
    }
}

@inline(__always) private func dot(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
    a.x*b.x + a.y*b.y + a.z*b.z
}
@inline(__always) private func cross(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> SIMD3<Float> {
    SIMD3<Float>(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x)
}
private func contactTangent(_ n: SIMD3<Float>) -> SIMD3<Float> {
    let axis = abs(n.x) < 0.57735 ? SIMD3<Float>(1,0,0)
        : (abs(n.y) < 0.57735 ? SIMD3<Float>(0,1,0) : SIMD3<Float>(0,0,1))
    let t = cross(axis, n)
    let length = sqrt(dot(t,t))
    return t / max(length, 1e-12)
}
