# Vendor-floor benchmark shims + dev shells for the svod-tk `vendor` bench.
# Built with ROCm from the pinned nixpkgs (clr brings hipcc + the HIP runtime and
# ships its own llvm/libxml2). The shims dlopen at bench runtime via libloading.
# `gpuTargets` is a fat-binary arch list: gfx1151 (RDNA3.5) + gfx942 (CDNA3/MI300X)
# by default; the shim's device code is built for each so one .so runs on any of
# them (the GEMM path is host hipBLASLt and already arch-blind).
{
  pkgs,
  stdenv,
  mkShell,
  commonArgs,
  nativeBuildInputs,
  gpuTargets ? [
    "gfx942"
    "gfx1151"
  ],
}:
let
  rocm = pkgs.rocmPackages;
  offloadFlags = pkgs.lib.concatMapStringsSep " " (t: "--offload-arch=${t}") gpuTargets;
  gpuTargetsSemi = pkgs.lib.concatStringsSep ";" gpuTargets;
  rocmLibPath = pkgs.lib.makeLibraryPath [
    rocm.hipblaslt
    rocm.clr
    rocm.rocm-runtime
    rocm.rocm-device-libs
  ];

  # hipBLASLt bf16->f32 GEMM + rocPRIM segmented-sort knn floor (gpuTargets).
  hipblasltShim = stdenv.mkDerivation {
    pname = "svod-hipblaslt-shim";
    version = "0.1.0";
    dontUnpack = true;
    nativeBuildInputs = [ rocm.clr ];
    buildInputs = [
      rocm.hipblaslt
      rocm.hipblas-common
      rocm.rocprim
      rocm.rocm-runtime
    ];
    buildPhase = ''
      runHook preBuild
      ${rocm.clr}/bin/hipcc -O3 -std=c++17 -fPIC -shared ${offloadFlags} \
        ${../tk/benches/shims/hipblaslt_shim.cpp} \
        -I${rocm.hipblaslt}/include -I${rocm.hipblas-common}/include -I${rocm.rocprim}/include \
        -L${rocm.hipblaslt}/lib -lhipblaslt -o libsvod_hipblaslt_shim.so
      runHook postBuild
    '';
    installPhase = ''
      runHook preInstall
      mkdir -p $out/lib && cp libsvod_hipblaslt_shim.so $out/lib/
      runHook postInstall
    '';
  };

  # CK ck_tile FMHA-forward floor for the `fa` arm — WIP scaffold, built from the
  # vendored CK source. Build with `nix build '.?submodules=1#ckFmhaShim'`.
  ckFmhaShim = stdenv.mkDerivation {
    pname = "svod-ck-fmha-shim";
    version = "0.1.0";
    src = pkgs.lib.cleanSource ../submodules/rocm-libraries/projects/composablekernel;
    nativeBuildInputs = [
      pkgs.cmake
      pkgs.ninja
      pkgs.python3
      rocm.clr
    ];
    buildInputs = [
      rocm.rocm-runtime
      rocm.rocm-device-libs
    ];
    configurePhase = ''
      runHook preConfigure
      cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=${rocm.clr}/bin/hipcc -DCMAKE_HIP_COMPILER=${rocm.clr}/bin/clang++ \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DGPU_TARGETS="${gpuTargetsSemi}" \
        -DPython3_EXECUTABLE=${pkgs.python3}/bin/python3
      runHook postConfigure
    '';
    buildPhase = ''
      runHook preBuild
      cmake --build build --target tile_fmha_fwd_instances
      ${rocm.clr}/bin/hipcc -O3 -std=c++17 -fPIC -fgpu-flush-denormals-to-zero \
        -DCK_TILE_FMHA_FWD_FAST_EXP2=1 ${offloadFlags} \
        -I./include -I./example/ck_tile/01_fmha -Ibuild -Ibuild/example/ck_tile/01_fmha \
        -c ${../tk/benches/shims/ck_fmha_shim.cpp} -o ck_fmha_shim.o
      objs=$(find build -path '*tile_fmha_fwd_instances.dir*' -name '*.o')
      ${rocm.clr}/bin/hipcc -shared -fPIC ${offloadFlags} ck_fmha_shim.o $objs -o libsvod_ck_fmha_shim.so
      runHook postBuild
    '';
    installPhase = ''
      runHook preInstall
      mkdir -p $out/lib && cp libsvod_ck_fmha_shim.so $out/lib/
      runHook postInstall
    '';
  };

  # A dev shell that runs the vendor bench: the shims + ROCm libs on LD_LIBRARY_PATH
  # (the bench dlopens them and self-skips when absent).
  mkBenchShell =
    libs:
    mkShell (
      commonArgs
      // {
        packages =
          (with pkgs; [
            rust_stable
            cargo-outdated
            git
          ])
          ++ nativeBuildInputs
          ++ [ rocm.rocminfo ];
        shellHook = ''
          export LD_LIBRARY_PATH=${libs}:${rocmLibPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
        '';
      }
    );
in
{
  packages = { inherit hipblasltShim ckFmhaShim; };
  shells = {
    bench = mkBenchShell "${hipblasltShim}/lib";
    "bench-fa" = mkBenchShell "${hipblasltShim}/lib:${ckFmhaShim}/lib";
  };
}
