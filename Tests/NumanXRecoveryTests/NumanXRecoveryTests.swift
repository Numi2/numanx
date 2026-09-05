import Foundation
import XCTest
@testable import NumanXRecovery

@MainActor
final class NumanXRecoveryTests: XCTestCase {
    private func fixture() -> (NumanXPreparedStateManifest, Data, Data) {
        let metadata = Data("synthetic-native-metadata-fixture-not-a-solver".utf8)
        let gpu = Data((0..<128).map(UInt8.init))
        let manifest = NumanXPreparedStateManifest(transactionFingerprint: 42, baseGeneration: 9,
            shadowGeneration: 10, startTimeNanoseconds: 80_000_000, endTimeNanoseconds: 100_000_000,
            solverConfigurationFingerprint: 3, topologyFingerprint: 4,
            authoritativeStateSHA256: NumanXRecoveryHash.sha256(metadata),
            gpuBufferImageSHA256: NumanXRecoveryHash.sha256(gpu), gpuBufferImageBytes: UInt64(gpu.count))
        return (manifest, metadata, gpu)
    }
    private func directory() throws -> URL {
        let path = FileManager.default.temporaryDirectory.resolvingSymlinksInPath()
            .appendingPathComponent("numanx-recovery-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: path, withIntermediateDirectories: false)
        return path
    }
    func testGenerationOverflowAndInvalidDigestAreRejected() throws {
        var (m, _, _) = fixture()
        m.baseGeneration = UInt64.max
        XCTAssertThrowsError(try m.validated())
        m.baseGeneration = 9; m.authoritativeStateSHA256 = String(repeating: "g", count: 64)
        XCTAssertThrowsError(try m.validated())
    }
    func testImageVerificationRequiresExactNativeBytes() throws {
        let (m, metadata, gpu) = fixture()
        XCTAssertNoThrow(try m.verify(authoritativeState: metadata, gpuBufferImage: gpu))
        XCTAssertThrowsError(try m.verify(authoritativeState: Data([0]), gpuBufferImage: gpu))
        XCTAssertThrowsError(try m.verify(authoritativeState: metadata, gpuBufferImage: gpu.dropLast()))
    }
    func testPrepareSurvivesCloseReopenWithoutInventingDecision() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let (m, metadata, gpu) = fixture()
        let first = try FileNumanXPreparedStateStore(directoryURL: root)
        try await first.prepareDurably(manifest: m, authoritativeState: metadata, gpuBufferImage: gpu)
        try await first.close()
        let restored = try FileNumanXPreparedStateStore(directoryURL: root)
        let data = try await restored.loadPrepared(transactionFingerprint: 42)
        let decision = try await restored.decision(transactionFingerprint: 42)
        XCTAssertEqual(data.manifest, m); XCTAssertEqual(data.authoritativeState, metadata)
        XCTAssertEqual(data.gpuBufferImage, gpu); XCTAssertEqual(decision, .prepared)
        try await restored.close()
    }
    func testDecidedCommitSurvivesReopenAndCannotAbort() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let (m, metadata, gpu) = fixture()
        let first = try FileNumanXPreparedStateStore(directoryURL: root)
        try await first.prepareDurably(manifest: m, authoritativeState: metadata, gpuBufferImage: gpu)
        try await first.decideCommit(transactionFingerprint: 42); try await first.close()
        let second = try FileNumanXPreparedStateStore(directoryURL: root)
        let decision = try await second.decision(transactionFingerprint: 42)
        XCTAssertEqual(decision, .commitDecided)
        do { try await second.abortPrepared(transactionFingerprint: 42); XCTFail("must reject abort after decision") } catch {}
        try await second.markCommitted(transactionFingerprint: 42)
        try await second.markCommitted(transactionFingerprint: 42)
        let finished = try await second.decision(transactionFingerprint: 42)
        XCTAssertEqual(finished, .committed); try await second.close()
    }
    func testAbortCannotBecomeCommit() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let (m, metadata, gpu) = fixture(), store = try FileNumanXPreparedStateStore(directoryURL: root)
        try await store.prepareDurably(manifest: m, authoritativeState: metadata, gpuBufferImage: gpu)
        try await store.abortPrepared(transactionFingerprint: 42)
        do { try await store.decideCommit(transactionFingerprint: 42); XCTFail("abort is terminal") } catch {}
        try await store.close()
    }
    func testSecondWriterAndSymlinkDirectoryAreRejected() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let first = try FileNumanXPreparedStateStore(directoryURL: root)
        XCTAssertThrowsError(try FileNumanXPreparedStateStore(directoryURL: root))
        let alias = root.appendingPathComponent("alias")
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: root)
        XCTAssertThrowsError(try FileNumanXPreparedStateStore(directoryURL: alias))
        try await first.close()
    }
    func testCorruptedGPUBytesAreRejectedOnLoad() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let (m, metadata, gpu) = fixture(), store = try FileNumanXPreparedStateStore(directoryURL: root)
        try await store.prepareDurably(manifest: m, authoritativeState: metadata, gpuBufferImage: gpu)
        let path = root.appendingPathComponent("2a.prepared")
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: path.path)
        var bytes = try Data(contentsOf: path)
        bytes[bytes.count - 1] ^= 1
        try bytes.write(to: path)
        do { _ = try await store.loadPrepared(transactionFingerprint: 42); XCTFail("digest must fail") } catch {}
        try await store.close()
    }
    func testCoordinatorNeverInfersCommitFromPreparedFile() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let (m, metadata, gpu) = fixture(), store = try FileNumanXPreparedStateStore(directoryURL: root)
        try await store.prepareDurably(manifest: m, authoritativeState: metadata, gpuBufferImage: gpu)
        let solver = FixtureOwner(manifest: m, metadata: metadata, gpu: gpu)
        let coordinator = NumanXRecoveryCoordinator(store: store, solver: solver)
        do { _ = try await coordinator.recover(transactionFingerprint: 42); XCTFail("undecided") }
        catch NumanXRecoveryError.undecided {}
        let calls = await solver.publications
        XCTAssertEqual(calls, 0); try await store.close()
    }
    func testCoordinatorRestoresDecidedBytesAndPublishesOnlyOnce() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let (m, metadata, gpu) = fixture(), store = try FileNumanXPreparedStateStore(directoryURL: root)
        try await store.prepareDurably(manifest: m, authoritativeState: metadata, gpuBufferImage: gpu)
        try await store.decideCommit(transactionFingerprint: 42)
        let solver = FixtureOwner(manifest: m, metadata: metadata, gpu: gpu)
        let coordinator = NumanXRecoveryCoordinator(store: store, solver: solver)
        let a = try await coordinator.recover(transactionFingerprint: 42)
        let b = try await coordinator.recover(transactionFingerprint: 42)
        XCTAssertEqual(a, b); XCTAssertEqual(a.generation, 10)
        let calls = await solver.publications, state = try await store.decision(transactionFingerprint: 42)
        XCTAssertEqual(calls, 1); XCTAssertEqual(state, .committed)
        try await store.close()
    }
}

private actor FixtureOwner: NumanXVerifiedRecoverableSolver {
    let manifest: NumanXPreparedStateManifest
    let metadata: Data, gpu: Data
    var staged = false
    var receipt: NumanXPublicationReceipt?
    private(set) var publications = 0
    init(manifest: NumanXPreparedStateManifest, metadata: Data, gpu: Data) {
        self.manifest = manifest; self.metadata = metadata; self.gpu = gpu
    }
    func makeDurablePreparedImage(transactionFingerprint: UInt64) throws
        -> (manifest: NumanXPreparedStateManifest, authoritativeState: Data, gpuBufferImage: Data) {
        guard transactionFingerprint == manifest.transactionFingerprint else { throw NumanXRecoveryError.invalidManifest }
        return (manifest, metadata, gpu)
    }
    func restorePreparedImage(manifest: NumanXPreparedStateManifest, authoritativeState: Data, gpuBufferImage: Data) throws {
        guard manifest == self.manifest, authoritativeState == metadata, gpuBufferImage == gpu else { throw NumanXRecoveryError.corruptImage }
        staged = true
    }
    func publishRestoredPrepared(transactionFingerprint: UInt64) {
        guard staged, transactionFingerprint == manifest.transactionFingerprint else { return }
        if receipt == nil { publications += 1 }
        receipt = NumanXPublicationReceipt(manifest: manifest)
    }
    func committedPublicationReceipt() -> NumanXPublicationReceipt? { receipt }
}
