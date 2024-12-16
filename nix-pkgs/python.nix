pkgs: {
  refidxdb = pkgs.python3Packages.callPackage ./refidxdb { };
  pywigxjpf = pkgs.python3Packages.callPackage ./pywigxjpf { };
  mkdocs-coverage = pkgs.python3Packages.callPackage ./mkdocs-coverage { };
  # speedscope = pkgs.python3Packages.callPackage ./speedscope { };
}
