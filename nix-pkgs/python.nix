pkgs: {
  refidxdb = pkgs.python3Packages.callPackage ./refidxdb { };
  pywigxjpf = pkgs.python3Packages.callPackage ./pywigxjpf { };
  mkdocs-coverage = pkgs.python3Packages.callPackage ./mkdocs-coverage { };
  stpyvista = pkgs.python3Packages.callPackage ./stpyvista { };
  # speedscope = pkgs.python3Packages.callPackage ./speedscope { };
}
