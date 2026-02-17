{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
    utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
    crane = {
      url = "github:ipetkov/crane";
    };
    advisory-db = {
      url = "github:rustsec/advisory-db";
      flake = false;
    };
    treefmtSrc.url = "github:numtide/treefmt-nix";
  };

  outputs =
    {
      self,
      nixpkgs,
      utils,
      crane,
      rust-overlay,
      advisory-db,
      treefmtSrc,
    }:
    utils.lib.eachDefaultSystem (
      system:
      let
        pkgs =
          (import nixpkgs {
            inherit system;
            config.allowUnfree = true;
            overlays = [ rust-overlay.overlays.default ];
          }).extend
            (
              self: super: {
                rust_stable = self.rust-bin.stable.latest.default;
                rust_nightly = self.rust-bin.nightly.latest.default;
              }
            );

        treefmt = treefmtSrc.lib.evalModule pkgs ./nix/treefmt.nix;

        llvm = pkgs.llvmPackages_21;
        stdenv = llvm.stdenv;
        mkShell = pkgs.mkShell.override { inherit stdenv; };
        crane' = (crane.mkLib pkgs).overrideToolchain (pkgs.rust_stable);

        sourceFilter = path: type: (crane'.filterCargoSources path type);

        src = pkgs.lib.cleanSourceWith {
          src = ./.;
          filter = sourceFilter;
        };
        nativeBuildInputs = with pkgs; [
          llvm.llvm.dev
          llvm.mlir
          pkgconf
          libffi
          libxml2
          z3
          zlib
          clang
        ];

        mlirSysPrefix = pkgs.symlinkJoin {
          name = "mlir-sys-prefix";
          paths = [
            llvm.llvm.dev # llvm-config, LLVM headers
            llvm.llvm.lib # LLVM libraries
            llvm.mlir # MLIR libraries (libMLIR*)
            llvm.mlir.dev # MLIR headers (mlir-c/)
          ];
        };

        commonArgs = {
          inherit src nativeBuildInputs;
          MLIR_SYS_210_PREFIX = "${mlirSysPrefix}";
          TABLEGEN_210_PREFIX = "${mlirSysPrefix}";
          LIBCLANG_PATH = "${pkgs.libclang.lib}/lib/";
          # Disable fortify since debug builds use -O0 but _FORTIFY_SOURCE requires optimization
          hardeningDisable = [ "fortify" ];

        };

        cargoArtifacts = crane'.buildDepsOnly (commonArgs // { });
      in
      {
        checks = {
          clippy = crane'.cargoClippy (
            commonArgs
            // {
              inherit cargoArtifacts;
              cargoClippyExtraArgs = "--all-targets -- --deny warnings";
            }
          );

          test = crane'.cargoNextest (
            commonArgs
            // {
              inherit cargoArtifacts;
              cargoNextestExtraArgs = "--features z3,proptest";
            }
          );

          audit = crane'.cargoAudit {
            inherit src advisory-db;
          };

          rustfmt = crane'.cargoFmt { inherit src; };
          # treefmt = treefmt.config.build.check self;
        };

        devShells = rec {
          stable = mkShell (
            commonArgs
            // {
              packages =
                (with pkgs; [
                  rust_stable
                  cargo-outdated
                  git
                ])
                ++ nativeBuildInputs;
            }
          );

          nightly = mkShell (
            commonArgs
            // {
              packages =
                (with pkgs; [
                  rust_stable
                  cargo-outdated
                  git
                ])
                ++ nativeBuildInputs
                ++ [ pkgs.cargo-udeps ];
            }
          );

          default = stable;
        };

        formatter = treefmt.config.build.wrapper;
      }
    );
}
