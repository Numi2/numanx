// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "NumanX",
    platforms: [.macOS("26.0")],
    products: [
        .library(name: "NumanXRecovery", targets: ["NumanXRecovery"]),
        .executable(name: "numanx-recovery", targets: ["NumanXRecoveryCLI"])
    ],
    targets: [
        .target(name: "NumanXRecovery"),
        .executableTarget(name: "NumanXRecoveryCLI", dependencies: ["NumanXRecovery"]),
        .testTarget(name: "NumanXRecoveryTests", dependencies: ["NumanXRecovery"])
    ],
    swiftLanguageModes: [.v6]
)
