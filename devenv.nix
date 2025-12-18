{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  env = {
    GREET = "devenv";
    # CUDA_PATH = pkgs.cudaPackages.cudatoolkit;
    # CUDA_HOME = pkgs.cudaPackages.cudatoolkit;
    CUDA_HOME = lib.getDev (
      pkgs.symlinkJoin {
        name = "cudatoolkit";
        paths = with pkgs.cudaPackages; [
          cudatoolkit
          (lib.getStatic cuda_cudart)
        ];
        postBuild = ''
          ln -s $out/lib $out/lib64
        '';
      }
    );
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
  };

  # enterShell = ''
  #   hello         # Run scripts directly
  #   git --version # Use packages
  # '';

  # https://devenv.sh/tasks/
  # tasks = {
  #   "myproj:setup".exec = "mytool build";
  #   "devenv:enterShell".after = [ "myproj:setup" ];
  # };

  # https://devenv.sh/tests/
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

    libraries = [ pkgs.zlib ];
  };
}
