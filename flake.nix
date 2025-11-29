{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
    utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
    crane = {
      url = "github:ipetkov/crane";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    advisory-db = {
      url = "github:rustsec/advisory-db";
      flake = false;
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      utils,
      crane,
      rust-overlay,
      advisory-db,
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
          pkgconf
          libffi
          libxml2
          z3
          zlib
        ];
        commonArgs = {
          inherit src nativeBuildInputs;
          LLVM_SYS_211_PREFIX = "${llvm.llvm.dev}";
          LIBCLANG_PATH = "${pkgs.libclang.lib}/lib/";
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

          format = crane'.cargoFmt { inherit src; };
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
      }
    );
}
