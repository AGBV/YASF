{
  lib,
  buildPythonPackage,
  fetchPypi,

  hatchling,

  pydantic,
  rich,
  numpy,
  polars,
  streamlit,
  plotly,
  pyyaml,
  click,
  plotext,
  tqdm,

  pytest,
  pytest-cov,
}:
let
  pname = "refidxdb";
  version = "0.0.8";
in
buildPythonPackage {
  inherit pname version;
  pyproject = true;

  src = fetchPypi {
    inherit pname version;
    extension = "tar.gz";
    hash = "sha256-hDuJeiqSoVZhMz/qLUMjqlCRSoINFvYwlz2IE78XTwg=";
  };

  build-system = [
    hatchling
  ];

  dependencies = [
    pydantic
    rich
    numpy
    polars
    streamlit
    plotly
    pyyaml
    click
    plotext
    tqdm
  ];

  nativeCheckInputs = [
    pytest
    pytest-cov
  ];

  # doChecks = true;

  meta = with lib; {
    homepage = "https://github.com/arunoruto/RefIdxDB/";
    description = "Python interface for various refractive index databases ";
    license = licenses.mit;
    platforms = platforms.linux;
    maintainers = with maintainers; [ arunoruto ];
  };
}
