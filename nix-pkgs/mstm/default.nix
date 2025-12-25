{
  stdenv,
  lib,
  makeWrapper,
  fetchFromGitHub,
  gfortran,
  mpi,

  enableMPI ? true,
  december2023 ? false,
}:
let
  compiler = if enableMPI then "mpif90" else "gfortran";
  type-str = if enableMPI then "parallel" else "serial";

  file-order =
    if december2023 then
      (
        [ "mstm-intrinsics" ]
        ++ (if enableMPI then [ "mpidefs-parallel-2" ] else [ "mpidefs-serial" ])
        ++ [
          "mstm-modules-33"
          "fft_translation-5"
          "mstm-scatprops-26"
          "mstm-solver-8"
          "random_configuration-10"
          "lmfit"
          "mstm-input-37"
          "mstm-main-3"
        ]
      )
    else
      (
        [ "mstm-intrinsics" ]
        ++ (if enableMPI then [ "mpidefs-parallel" ] else [ "mpidefs-serial" ])
        ++ [ "mstm-v4.0" ]
      );
in
stdenv.mkDerivation rec {
  pname = "mstm";
  version = "4.0.0-unstable-2025-01-14";
  # version = "4-2024-02-08";

  src = fetchFromGitHub {
    owner = "dmckwski";
    repo = pname;
    rev = "a0c982121cf9ac352531f4816639a07d814385bd";
    hash = "sha256-IjCYT6+fAS9Fysn/+v7CzpaUewPqT6k62HaevSBVp6g=";
    # hash = "sha256-gIlZMHE7vNDtRtzdYRtXeo+PBgJHi911AJaXzMn/wpI=";
  };

  nativeBuildInputs = [
    gfortran
  ];

  buildInputs = lib.optionals enableMPI [
    makeWrapper
    mpi
  ];

  buildPhase = ''
    runHook preBuild

    # cd ./code
    # ${compiler} -O2 -fallow-argument-mismatch -c -o mstm-intrinsics.obj mstm-intrinsics.f90
    # ${compiler} -O2 -fallow-argument-mismatch -c -o mpidefs-${type-str}.obj mpidefs-${type-str}.f90
    # ${compiler} -O2 -fallow-argument-mismatch -c -o mstm.obj mstm-v4.0.f90
    # ${compiler} -O2 -fallow-argument-mismatch    -o mstm-bin mstm-intrinsics.obj mpidefs-${type-str}.obj mstm.obj
    # cd ./december2023
    # ${compiler} -O2 -fallow-argument-mismatch -c -o mstm-intrinsics.obj mstm-intrinsics.f90
    # ${compiler} -O2 -fallow-argument-mismatch -c -o mpidefs-parallel-2.obj mpidefs-parallel-2.f90
    # ${compiler} -O2 -fallow-argument-mismatch -c -o mstm-modules-33.obj mstm-modules-33.f90
    # ${compiler} -O2 -fallow-argument-mismatch -c -o fft_translation-5.obj fft_translation-5.f90
    # ${compiler} -O2 -fallow-argument-mismatch -c -o mstm-scatprops-26.obj mstm-scatprops-26.f90
    # ${compiler} -O2 -fallow-argument-mismatch -c -o mstm-solver-8.obj mstm-solver-8.f90
    # ${compiler} -O2 -fallow-argument-mismatch -c -o random_configuration-10.obj random_configuration-10.f90
    # ${compiler} -O2 -fallow-argument-mismatch -c -o lmfit.obj lmfit.f90
    # ${compiler} -O2 -fallow-argument-mismatch -c -o mstm-input-37.obj mstm-input-37.f90
    # ${compiler} -O2 -fallow-argument-mismatch -c -o mstm-main-3.obj mstm-main-3.f90

    ${if december2023 then "cd ./december2023" else "cd ./code"}
    ${lib.strings.concatMapStringsSep "\n" (
      file: "${compiler} -O2 -fallow-argument-mismatch -c -o ${file}.obj ${file}.f90"
    ) file-order}
    ${compiler} -O2 -fallow-argument-mismatch -o mstm-bin ${
      lib.strings.concatMapStringsSep " " (x: "${x}.obj") file-order
    }

    runHook postBuild
  '';

  preInstall = lib.optionalString enableMPI ''
    mkdir -p $out/bin
    cat >> $out/bin/mstm << EOF
    #!/usr/bin/env bash

    if [[ \$# -eq 0 ]]; then
    echo "If one argument is provided its the input file"
    echo "and it defaults to four threads."
    echo "With two arguments:"
    echo "- The first argument is the number of processors (multiple of 4)."
    echo "- The second argument is the input file."
    elif [[ \$# -eq 1 ]]; then
    mpiexec -n 4 $out/bin/mstm-bin \$1
    elif [[ \$# -eq 2 ]]; then
    mpiexec -n \$1 $out/bin/mstm-bin \$2
    else
    mpiexec -n \$1 $out/bin/mstm-bin \$2
    fi
    EOF
    chmod +x $out/bin/mstm
  '';

  installPhase = ''
    runHook preInstall

    mkdir -p $out/bin
    cp mstm-bin $out/bin${lib.optionalString (!enableMPI) "/mstm"}

    runHook postInstall
  '';

  postInstall = lib.optionalString enableMPI ''
    wrapProgram $out/bin/mstm \
      --prefix PATH : ${lib.makeBinPath [ mpi ]}
  '';

  meta = with lib; {
    homepage = "https://github.com/dmckwski/MSTM";
    description = "Multiple Sphere T Matrix code in Fortran";
    license = licenses.mit;
    platforms = platforms.linux;
    maintainers = with maintainers; [ arunoruto ];
    mainProgram = "mstm";
  };
}
