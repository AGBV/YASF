{
  lib,
  buildPythonPackage,
  fetchPypi,
}:
let
  pname = "mkdocs-coverage";
  version = "1.1.0";
in
buildPythonPackage {
  inherit pname version;
  src = fetchPypi {
    inherit pname version;
    extension = "tar.gz";
    hash = "sha256-a67cc6f6d548b8d6b4b21ecd777f2e3768b49e7a95e54c6df158e7c0f179134c";
  };

  meta = with lib; {
    homepage = "https://github.com/pawamoy/mkdocs-coverage";
    description = "MkDocs plugin to integrate your coverage HTML report into your site.";
    license = licenses.isc;
    platforms = platforms.linux;
    maintainers = with maintainers; [ arunoruto ];
  };
}
