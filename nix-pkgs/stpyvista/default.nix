{
  lib,
  buildPythonPackage,
  fetchFromGitHub,

  # build tool
  hatchling,

  # dependencies
  streamlit,
  pyvista,
  bokeh,
  panel,
}:
let
  pname = "stpyvista";
  version = "0.1.4";
in
buildPythonPackage {
  inherit pname version;
  pyproject = true;
  # format = "setuptools";
  src = fetchFromGitHub {
    owner = "edsaac";
    repo = pname;
    rev = "refs/tags/v${version}";
    hash = "sha256-jTpvzfQv5UxS74m5tJNAXDzT2F7IYJgXz60JfLwbTrs=";
  };

  nativeBuildInputs = [ hatchling ];

  dependencies = [
    streamlit
    pyvista
    bokeh
    panel
  ];

  doCheck = true;
  pythonImportsCheck = [ "stpyvista" ];

  meta = with lib; {
    homepage = "https://github.com/edsaac/stpyvista";
    description = "🧊 Show 3D visualizations from PyVista in Streamlit";
    license = licenses.gpl3;
    platforms = platforms.linux;
    maintainers = with maintainers; [ arunoruto ];
  };
}
