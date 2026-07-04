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

  # CK ck_tile FMHA-forward floor for the `fa` arm. Source fetched from the standalone
  # ROCm/composable_kernel repo at the tag matching the pinned rocm (7.2.3) — same CK-root
  # layout as the monorepo's projects/composablekernel.
  #
  # Split in two so iterating on the shim doesn't recompile the 2047 gfx942 instances: the
  # expensive `ckFmhaInstances` (configure + `tile_fmha_fwd_instances`) is cached and installs
  # its whole built tree; the fast `ckFmhaShim` compiles the C-ABI wrapper and links those cached
  # objects into the dlopen'd .so. Both use the ROCm clang stdenv (22.x): ck_tile's hand-written
  # GCN asm (masked buffer_store_if<N>) needs ROCm clang's readfirstlane into the "s"(res) SGPR
  # operand; nixpkgs' default stdenv clang (21.x) errors on it. amdgcn also rejects several stdenv
  # hardening flags (e.g. zerocallusedregs → `-fzero-call-used-regs=used-gpr`), so disable them.
  ckFmhaInstances = rocm.llvm.rocmClangStdenv.mkDerivation {
    pname = "svod-ck-fmha-instances";
    version = "0.1.0";
    hardeningDisable = [ "all" ];
    src = pkgs.fetchFromGitHub {
      owner = "ROCm";
      repo = "composable_kernel";
      rev = "rocm-7.2.3";
      hash = "sha256-ABL0MSmWtqAeY5uyw8Ib64npB2v82baUnzLpmrEgDn4=";
      fetchSubmodules = true;
    };
    nativeBuildInputs = [
      pkgs.cmake
      pkgs.ninja
      pkgs.python3
      pkgs.gitMinimal
      rocm.clr
      rocm.rocminfo
    ];
    buildInputs = [
      rocm.rocm-cmake
      rocm.rocm-runtime
      rocm.rocm-device-libs
    ];
    # We only need ck_tile's `tile_fmha_fwd_instances` (in example/ck_tile/01_fmha) + the host
    # `utility`. Skip the classic `tensor_operation_instance/gpu` tree: it's unused here and its
    # `transpose` instance trips a CMake-4.1 strictness bug (empty COMPILE_FLAGS when the arch
    # filter empties the target list) at configure time.
    postPatch = ''
      substituteInPlace library/CMakeLists.txt \
        --replace-fail "add_subdirectory(src/tensor_operation_instance/gpu)" \
          "# skipped: classic instance tree (unneeded for ck_tile fmha)"
      # Build only the ck_tile FMHA example. The example/ glob pulls in every example, and
      # unrelated classic ones (e.g. 63_layernorm4d_fwd) fail the generate step with an empty
      # HIP_ARCHITECTURES under our gfx942+gfx1151 target set. Add just ck_tile/01_fmha and
      # empty the glob (its FOREACH then no-ops). tile_fmha_fwd_instances is a self-contained
      # OBJECT lib, so no sibling example/utility target is needed to compile it.
      substituteInPlace example/CMakeLists.txt \
        --replace-fail "file(GLOB dir_list LIST_DIRECTORIES true *)" \
          "add_subdirectory(ck_tile/01_fmha)
file(GLOB dir_list LIST_DIRECTORIES true __svod_none__/*)"
      # rocm_check_target_ids probes each arch via check_cxx_compiler_flag("-xhip ...") against
      # the CXX compiler (gcc), which rejects -xhip in the GPU-less sandbox → SUPPORTED_GPU_TARGETS
      # comes back empty and every gfx9 instance subdir early-returns. The targets are valid; force
      # the list right after the (failed) probe.
      substituteInPlace CMakeLists.txt \
        --replace-fail '        TARGETS ''${CK_GPU_TARGETS})' \
          '        TARGETS ''${CK_GPU_TARGETS})
set(SUPPORTED_GPU_TARGETS ''${CK_GPU_TARGETS})'
      # Trim the fwd codegen to what the bench needs. The full set is 2045 instances
      # (~30 min); bf16 + optdim 64,128 is 385 and still covers every bench shape
      # (d64/d128 x causal/non-causal x batch, 64 tile configs each — the runtime
      # fmha_fwd dispatcher picks among them). ~5x fewer instances → ~5 min build.
      substituteInPlace example/ck_tile/01_fmha/CMakeLists.txt \
        --replace-fail "--optdim 32,64,128,256" "--optdim 64,128
  --filter *_bf16_*"
    '';
    configurePhase = ''
      runHook preConfigure
      # Mirror nixpkgs' composable_kernel_base HIP-CMake setup: point CMake at clr's own hip
      # cmake modules (they resolve the clang runtime / clangrt builtins) + ROCM_PATH, rather
      # than hand-setting CMAKE_HIP_COMPILER (which mis-detects the arch/clangrt in the sandbox).
      # Use GPU_TARGETS (not GPU_ARCHS): the top-level CMake only adds the `example` subdir —
      # where tile_fmha_fwd_instances lives — when GPU_ARCHS is unset and GPU_TARGETS is user-set.
      export ROCM_PATH=${rocm.clr}
      cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_MODULE_PATH=${rocm.clr}/hip/cmake \
        -DROCM_PATH=${rocm.clr} -DCMAKE_HIP_COMPILER_ROCM_ROOT=${rocm.clr} \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DGPU_TARGETS="${gpuTargetsSemi}" \
        -DBUILD_TESTING=OFF \
        -DPython3_EXECUTABLE=${pkgs.python3}/bin/python3
      runHook postConfigure
    '';
    buildPhase = ''
      runHook preBuild
      cmake --build build --target tile_fmha_fwd_instances
      runHook postBuild
    '';
    # Ship the whole configured+built tree: the shim needs the source headers (include/,
    # example/ck_tile/01_fmha), the generated headers (build/…), and the instance objects.
    installPhase = ''
      runHook preInstall
      mkdir -p $out
      cp -r . $out/src
      runHook postInstall
    '';
  };

  # Default stdenv (NOT the ROCm clang stdenv): the wrapper is plain host code — the CK asm
  # lives in the cached ckFmhaInstances objects — and rocmClangStdenv bakes a gcc-prefix into
  # the .so's RUNPATH whose libdl needs a newer glibc than the host, breaking dlopen from the
  # native (system-glibc) cargo bench. This matches the working hipblasltShim.
  ckFmhaShim = stdenv.mkDerivation {
    pname = "svod-ck-fmha-shim";
    version = "0.1.0";
    dontUnpack = true;
    hardeningDisable = [ "all" ];
    nativeBuildInputs = [ rocm.clr ];
    buildInputs = [
      rocm.rocm-runtime
      rocm.rocm-device-libs
    ];
    # gfx942-only: the ck_tile FMHA instances are gfx9-only (INST_TARGETS filters to gfx9|gfx12)
    # and the bench runs on MI300X; a gfx1151 slice would carry no fmha device code anyway.
    buildPhase = ''
      runHook preBuild
      inst=${ckFmhaInstances}/src
      ${rocm.clr}/bin/hipcc -O3 -std=c++17 -fPIC -fgpu-flush-denormals-to-zero \
        -DCK_TILE_FMHA_FWD_FAST_EXP2=1 --offload-arch=gfx942 \
        -I$inst/include -I$inst/example/ck_tile/01_fmha -I$inst/build -I$inst/build/example/ck_tile/01_fmha \
        -c ${../tk/benches/shims/ck_fmha_shim.cpp} -o ck_fmha_shim.o
      objs=$(find $inst/build -path '*tile_fmha_fwd_instances.dir*' -name '*.o')
      ${rocm.clr}/bin/hipcc -shared -fPIC --offload-arch=gfx942 ck_fmha_shim.o $objs -o libsvod_ck_fmha_shim.so
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
          export LD_LIBRARY_PATH=${libs}:${rocmLibPath}:${
            pkgs.lib.makeLibraryPath [
              pkgs.sqlite
              pkgs.elfutils
              pkgs.zlib
              pkgs.zstd
              pkgs.libdrm
              pkgs.ncurses
              pkgs.stdenv.cc.cc.lib
            ]
          }''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
        '';
      }
    );
in
{
  packages = { inherit hipblasltShim ckFmhaInstances ckFmhaShim; };
  shells = {
    bench = mkBenchShell "${hipblasltShim}/lib";
    "bench-fa" = mkBenchShell "${hipblasltShim}/lib:${ckFmhaShim}/lib";
  };
}
