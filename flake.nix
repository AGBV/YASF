# Sources:
# https://pyproject-nix.github.io/pyproject.nix/use-cases/pyproject.html
# https://pyproject-nix.github.io/uv2nix/usage/hello-world.html

{
  description = "YASF flake devenv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs = {
        pyproject-nix.follows = "pyproject-nix";
        nixpkgs.follows = "nixpkgs";
      };
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs = {
        pyproject-nix.follows = "pyproject-nix";
        uv2nix.follows = "uv2nix";
        nixpkgs.follows = "nixpkgs";
      };
    };

    flake-parts = {
      url = "github:hercules-ci/flake-parts";
    };
  };

  outputs =
    inputs@{
      flake-parts,
      nixpkgs,
      uv2nix,
      pyproject-nix,
      pyproject-build-systems,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
      ];
      perSystem =
        {
          # system,
          self',
          lib,
          pkgs,
          ...
        }:
        let
          # pkgs = import inputs.nixpkgs {
          #   inherit system;
          #   config = {
          #     cudaSupport = true;
          #   };
          # };
          project = pyproject-nix.lib.project.loadPyproject { projectRoot = ./.; };
          workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

          overlay = workspace.mkPyprojectOverlay {
            sourcePreference = "wheel";
          };

          pyprojectOverrides = final: prev: {
            #
          };

          python = pkgs.python3.override {
            packageOverrides = final: prev: {
              inherit (self'.packages)
                refidxdb
                pywigxjpf
                mkdocs-coverage
                ;
              numba = pkgs.python3Packages.numbaWithCuda;
              # numba = final.numbaWithCuda;
              # numba = prev.numba.overridePythonAttrs (old: {
              #   cudaSupport = true;
              # });
            };
          };

          pythonSet =
            (pkgs.callPackage pyproject-nix.build.packages {
              inherit python;
            }).overrideScope
              (
                lib.composeManyExtensions [
                  pyproject-build-systems.overlays.default
                  overlay
                  pyprojectOverrides
                ]
              );

        in
        {
          # _module.args.pkgs = pkgs;

          packages = {
            default = self'.packages.yasfpy;
            yasfpy = python.pkgs.buildPythonPackage (
              (project.renderers.buildPythonPackage { inherit python; }) // { env.CUSTOM_ENVVAR = "hello"; }
            );
            yasfpy-env = pythonSet.mkVirtualEnv "yasfpy-env" workspace.deps.default;
          } // (import ./nix-pkgs/python.nix pkgs);

          devShells = {
            default =
              let
                arg = project.renderers.withPackages {
                  inherit python;
                  extras = [
                    "test"
                    # "docs"
                  ];
                  # extraPackages = ps: with ps; [ numbaWithCuda ];
                };

                pythonEnv = python.withPackages arg;
              in
              pkgs.mkShell {
                packages = [
                  pythonEnv
                  self'.packages.yasfpy
                ];
                shellHook = '''';
              };

            # Due to cudatoolkit, it needs to be run with
            # nix develop .#cuda --impure
            cuda = self'.devShells.default.overrideAttrs (old: {
              packages =
                old.nativeBuildInputs
                ++ (with pkgs.cudaPackages; [
                  cudatoolkit
                  cuda_nvcc
                  cuda_nvrtc
                ]);
              shellHook = ''
                # export CUDA_HOME=${pkgs.cudaPackages.cudatoolkit}
                # export NUMBA_CUDA_DRIVER=/run/opengl-driver/lib/libcuda.so
              '';
            });

            impure = pkgs.mkShell {
              packages = [
                python
                pkgs.uv
              ];
              shellHook = ''
                unset PYTHONPATH
              '';
            };

            uv2nix =
              let
                editableOverlay = workspace.mkEditablePyprojectOverlay {
                  root = "$REPO_ROOT";
                };

                editablePythonSet = pythonSet.overrideScope editableOverlay;

                virtualenv = editablePythonSet.mkVirtualEnv "yasfpy-dev-env" workspace.deps.default;

              in
              pkgs.mkShell {
                packages = [
                  virtualenv
                  pkgs.uv
                ];
                shellHook = ''
                  unset PYTHONPATH
                  export REPO_ROOT=$(git rev-parse --show-toplevel)
                '';
              };
          };
        };
    };
}
