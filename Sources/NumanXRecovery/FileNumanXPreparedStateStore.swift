import Foundation
import Darwin

/// Executable local persistence, not a numerical solver. All paths are relative to a locked
/// existing directory; artifacts are immutable. A prepare contains real bytes, not only hashes.
public actor FileNumanXPreparedStateStore: NumanXDurablePreparedStateStore {
    private struct Header: Codable {
        var version: UInt32
        var manifest: NumanXPreparedStateManifest
        var authoritativeBytes: UInt64
    }
    private struct Marker: Codable, Equatable {
        var version: UInt32
        var transaction: UInt64
        var manifestSHA256: String
        var kind: String
    }
    private let directory: FileHandle
    private let lockFile: FileHandle
    private let maximumBytes: UInt64
    private var poisoned = false, closed = false

    public init(directoryURL: URL, maximumBytes: UInt64 = 536_870_912) throws {
        guard directoryURL.isFileURL, maximumBytes > 0, maximumBytes <= 1_073_741_824,
              !directoryURL.pathComponents.contains("..") else { throw NumanXRecoveryError.invalidStore("path or budget") }
        var path = directoryURL
        while path.path != "/" {
            let values = try path.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            guard values.isDirectory == true, values.isSymbolicLink != true else {
                throw NumanXRecoveryError.invalidStore("store parents must be real directories")
            }
            path.deleteLastPathComponent()
        }
        let fd = directoryURL.path.withCString { Darwin.open($0, O_RDONLY | O_DIRECTORY | O_NOFOLLOW) }
        guard fd >= 0 else { throw NumanXRecoveryError.invalidStore("directory open") }
        let directory = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
        let lockFD = openat(fd, ".writer.lock", O_RDWR | O_CREAT | O_NOFOLLOW, mode_t(0o600))
        guard lockFD >= 0 else { try? directory.close(); throw NumanXRecoveryError.invalidStore("lock open") }
        let lock = FileHandle(fileDescriptor: lockFD, closeOnDealloc: true)
        guard flock(lockFD, LOCK_EX | LOCK_NB) == 0 else {
            try? lock.close(); try? directory.close(); throw NumanXRecoveryError.writerBusy
        }
        self.directory = directory; lockFile = lock; self.maximumBytes = maximumBytes
    }

    public func close() throws {
        guard !closed else { return }
        closed = true
        try lockFile.close(); try directory.close()
    }

    public func prepareDurably(manifest source: NumanXPreparedStateManifest,
                              authoritativeState: Data, gpuBufferImage: Data) throws {
        try usable()
        let manifest = try source.validated()
        try manifest.verify(authoritativeState: authoritativeState, gpuBufferImage: gpuBufferImage)
        guard UInt64(authoritativeState.count) <= maximumBytes,
              UInt64(gpuBufferImage.count) <= maximumBytes - UInt64(authoritativeState.count) else {
            throw NumanXRecoveryError.capacity
        }
        let name = filename(manifest.transactionFingerprint, "prepared")
        if try exists(name) {
            let existing = try loadPrepared(transactionFingerprint: manifest.transactionFingerprint)
            guard existing.manifest == manifest, existing.authoritativeState == authoritativeState,
                  existing.gpuBufferImage == gpuBufferImage else { throw NumanXRecoveryError.conflictingDecision }
            return
        }
        let header = try NumanXRecoveryHash.encode(Header(version: 1, manifest: manifest,
            authoritativeBytes: UInt64(authoritativeState.count)))
        guard header.count <= 16_384 else { throw NumanXRecoveryError.capacity }
        var prefix = Data("NXPREP01".utf8)
        var size = UInt64(header.count).littleEndian
        withUnsafeBytes(of: &size) { prefix.append(contentsOf: $0) }
        try publish(name, parts: [prefix, header, authoritativeState, gpuBufferImage])
    }

    public func loadPrepared(transactionFingerprint: UInt64) throws
        -> (manifest: NumanXPreparedStateManifest, authoritativeState: Data, gpuBufferImage: Data) {
        try usable()
        let file = try openRead(filename(transactionFingerprint, "prepared"))
        defer { try? file.close() }
        let size = try file.seekToEnd()
        guard size >= 16, size <= maximumBytes + 16_400 else { throw NumanXRecoveryError.capacity }
        try file.seek(toOffset: 0)
        let prefix = try read(file, count: 16)
        guard prefix.prefix(8) == Data("NXPREP01".utf8) else { throw NumanXRecoveryError.corruptImage }
        let headerSize = prefix.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(fromByteOffset: 8, as: UInt64.self)) }
        guard headerSize > 0, headerSize <= 16_384 else { throw NumanXRecoveryError.capacity }
        let header = try JSONDecoder().decode(Header.self, from: read(file, count: Int(headerSize)))
        let manifest = try header.manifest.validated()
        guard header.version == 1, manifest.transactionFingerprint == transactionFingerprint,
              header.authoritativeBytes > 0, header.authoritativeBytes <= maximumBytes,
              manifest.gpuBufferImageBytes <= maximumBytes - header.authoritativeBytes,
              size == 16 + headerSize + header.authoritativeBytes + manifest.gpuBufferImageBytes else {
            throw NumanXRecoveryError.corruptImage
        }
        let metadata = try read(file, count: Int(header.authoritativeBytes))
        let gpu = try read(file, count: Int(manifest.gpuBufferImageBytes))
        try manifest.verify(authoritativeState: metadata, gpuBufferImage: gpu)
        return (manifest, metadata, gpu)
    }

    public func decision(transactionFingerprint: UInt64) throws -> NumanXRecoveryDecision {
        let prepared = try loadPrepared(transactionFingerprint: transactionFingerprint)
        let expected = NumanXRecoveryHash.sha256(try NumanXRecoveryHash.encode(prepared.manifest))
        var present = Set<String>()
        for kind in ["commit", "committed", "abort"] {
            let name = filename(transactionFingerprint, kind)
            if try exists(name) {
                let marker: Marker = try json(name)
                guard marker.version == 1, marker.transaction == transactionFingerprint,
                      marker.manifestSHA256 == expected, marker.kind == kind else { throw NumanXRecoveryError.corruptImage }
                present.insert(kind)
            }
        }
        guard !(present.contains("abort") && (present.contains("commit") || present.contains("committed"))),
              !present.contains("committed") || present.contains("commit") else {
            throw NumanXRecoveryError.conflictingDecision
        }
        if present.contains("committed") { return .committed }
        if present.contains("commit") { return .commitDecided }
        if present.contains("abort") { return .aborted }
        return .prepared // Undecided, never a presumed abort.
    }
    public func decideCommit(transactionFingerprint: UInt64) throws {
        switch try decision(transactionFingerprint: transactionFingerprint) {
        case .aborted: throw NumanXRecoveryError.conflictingDecision
        case .commitDecided, .committed: return
        case .prepared: try marker(transactionFingerprint, kind: "commit")
        }
    }
    public func markCommitted(transactionFingerprint: UInt64) throws {
        switch try decision(transactionFingerprint: transactionFingerprint) {
        case .committed: return
        case .commitDecided: try marker(transactionFingerprint, kind: "committed")
        default: throw NumanXRecoveryError.conflictingDecision
        }
    }
    public func abortPrepared(transactionFingerprint: UInt64) throws {
        switch try decision(transactionFingerprint: transactionFingerprint) {
        case .aborted: return
        case .prepared: try marker(transactionFingerprint, kind: "abort")
        default: throw NumanXRecoveryError.conflictingDecision
        }
    }
    private func marker(_ transaction: UInt64, kind: String) throws {
        let source = try loadPrepared(transactionFingerprint: transaction)
        let digest = NumanXRecoveryHash.sha256(try NumanXRecoveryHash.encode(source.manifest))
        try publish(filename(transaction, kind), parts: [NumanXRecoveryHash.encode(
            Marker(version: 1, transaction: transaction, manifestSHA256: digest, kind: kind))])
    }
    private func publish(_ name: String, parts: [Data]) throws {
        let temporary = ".\(UUID().uuidString).tmp"
        let fd = openat(directory.fileDescriptor, temporary, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, mode_t(0o400))
        guard fd >= 0 else { throw NumanXRecoveryError.invalidStore("temporary file") }
        let file = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
        defer { try? file.close(); _ = unlinkat(directory.fileDescriptor, temporary, 0) }
        do {
            for part in parts { try file.write(contentsOf: part) }
            try file.synchronize()
            guard fcntl(fd, F_FULLFSYNC) == 0,
                  linkat(directory.fileDescriptor, temporary, directory.fileDescriptor, name, 0) == 0,
                  fsync(directory.fileDescriptor) == 0 else {
                throw NumanXRecoveryError.invalidStore("immutable publication/full synchronization")
            }
        } catch { poisoned = true; throw error }
    }
    private func exists(_ name: String) throws -> Bool {
        var info = stat()
        if fstatat(directory.fileDescriptor, name, &info, AT_SYMLINK_NOFOLLOW) == 0 {
            guard info.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG) else { throw NumanXRecoveryError.invalidStore("nonregular artifact") }
            return true
        }
        guard errno == ENOENT else { throw NumanXRecoveryError.invalidStore("artifact stat") }
        return false
    }
    private func openRead(_ name: String) throws -> FileHandle {
        let fd = openat(directory.fileDescriptor, name, O_RDONLY | O_NOFOLLOW)
        guard fd >= 0 else { throw NumanXRecoveryError.invalidStore("missing or unsafe artifact") }
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG) else {
            _ = Darwin.close(fd); throw NumanXRecoveryError.invalidStore("artifact type")
        }
        return FileHandle(fileDescriptor: fd, closeOnDealloc: true)
    }
    private func json<T: Decodable>(_ name: String) throws -> T {
        let file = try openRead(name); defer { try? file.close() }
        let count = try file.seekToEnd()
        guard count > 0, count <= 16_384 else { throw NumanXRecoveryError.capacity }
        try file.seek(toOffset: 0)
        return try JSONDecoder().decode(T.self, from: read(file, count: Int(count)))
    }
    private func read(_ file: FileHandle, count: Int) throws -> Data {
        var result = Data(); result.reserveCapacity(count)
        while result.count < count {
            guard let chunk = try file.read(upToCount: min(count - result.count, 1_048_576)), !chunk.isEmpty else {
                throw NumanXRecoveryError.corruptImage
            }
            result.append(chunk)
        }
        return result
    }
    private func filename(_ transaction: UInt64, _ kind: String) -> String { "\(String(transaction, radix: 16)).\(kind)" }
    private func usable() throws {
        guard !closed, !poisoned else { throw NumanXRecoveryError.poisoned }
    }
}
