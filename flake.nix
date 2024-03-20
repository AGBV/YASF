{
  description = "Flake for YASF Development using Python";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
  let
    system = "x86_64-linux";
    pythonVersion = "python311";
    pkgs = nixpkgs.legacyPackages.${system};
    packages = ps: with ps; [
      # Base
      numba
      numba-scipy
      numpy
      pandas
      requests
      scipy
      pyyaml
      pywigxjpf
      icc-rt
      # Testing
      astropy
      coverage
      pytest
      pytest-cov
      # Docs
      mkdocs
      mkdocs-autorefs
      mkdocs-coverage
      mkdocs-material
      mkdocstrings
      mkdocstrings-python
      pymdown-extensio
    ];
  in
  {
    devShells.${system}.default = pkgs.mkShell {
      buildInputs = [
        pkgs.${pythonVersion}
        (pkgs.${pythonVersion}.withPackages packages)
        #pkgs.cudaPackages.cudatoolkit
      ];

      shellHook = ''
        echo "YASF Env"
      '';
    };
  };
}
