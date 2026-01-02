{
  pkgs,
  lib,
  config,
  ...
}:
let
  cuda = lib.getDev (
    pkgs.symlinkJoin {
      name = "cudatoolkit";
      paths = with pkgs.cudaPackages_12; [
        cudatoolkit
        "${cuda_nvcc}/nvvm"
        (lib.getStatic cuda_cudart)
      ];
      postBuild = ''
        ln -s $out/lib $out/lib64
      '';
    }
  );
  sphinx-host = "0.0.0.0";
  sphinx-port = "8000";
in
{
  env = {
    UV_PYTHON = toString config.languages.python.package.interpreter;
    CUDA_HOME = cuda;
    NUMBA_CUDA_DRIVER = "/run/opengl-driver/lib/libcuda.so";
    NUMBA_DISABLE_INTEL_SVML = true;
  };

  overlays = [
    (final: prev: {
      mstm = final.callPackage ./nix-pkgs/mstm { };
    })
  ];

  packages = with pkgs; [
    mstm
    linalg
    cudaPackages.cuda_nvcc
  ];

  scripts = {
    hello.exec = ''
      echo hello from $GREET
    '';
    yasf.exec = ''uv run yasf "$@"'';
    numba.exec = ''uv run numba "$@"'';

    docs.exec = ''rm -rf docs/_build docs/source/autoapi; uv run --group docs sphinx-build -b html docs/source docs/_build/html -W --keep-going'';
    docs-live.exec = ''rm -rf docs/_build docs/source/autoapi; uv run --group docs sphinx-autobuild docs/source docs/_build/html --host ${sphinx-host} --port ${sphinx-port} --watch yasfpy'';
  };

  processes = {
    docs-live.exec = ''rm -rf docs/_build docs/source/autoapi; uv run --group docs sphinx-autobuild docs/source docs/_build/html --host ${sphinx-host} --port ${sphinx-port} --watch yasfpy'';
  };

  enterShell = ''
    git --version
    if [ ! -L "$DEVENV_ROOT/.venv" ]; then
        ln -s "$DEVENV_STATE/venv/" "$DEVENV_ROOT/.venv"
    fi
  '';

  enterTest = ''
    echo "Running tests"
    git --version | grep --color=auto "${pkgs.git.version}"
  '';

  languages.python = {
    enable = true;

    uv = {
      enable = true;
      sync = {
        enable = true;
        groups = [
          "test"
          "docs"
          # "profiling"
        ];
      };
    };

    libraries = [
      pkgs.zlib
      cuda
      "/run/opengl-driver/lib/libcuda.so"
    ];
  };
}
