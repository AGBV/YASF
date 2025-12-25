% Generate wavebundle (Gaussian beam) test data using CELES
%
% This script creates test data for validating YASFPY's wavebundle implementation
% against CELES. It computes initial field coefficients for a Gaussian beam
% at normal incidence.
%
% Requirements:
%   - CELES installed and in MATLAB path
%   - Run from YASF-new/tests/matlab directory
%
% Output:
%   - wavebundle_test_data.mat containing all simulation parameters and results

clear all;
close all;

%% Add CELES to path (adjust path as needed)
% addpath(genpath('/path/to/celes'));

%% Simulation parameters
fprintf('=======================================================\n');
fprintf('CELES WAVEBUNDLE TEST DATA GENERATION\n');
fprintf('=======================================================\n\n');

% Wavelength and medium
wavelength = 550;  % nm
mediumRefractiveIndex = 1.0;  % vacuum
k_medium = 2*pi / wavelength * mediumRefractiveIndex;

% Particles
% Simple configuration: 4 particles in different positions
particles.position = [
    0,   0,   0;     % Particle 1: at focal point
    100, 0,   0;     % Particle 2: off-axis (x)
    0,   100, 0;     % Particle 3: off-axis (y)
    0,   0,   200;   % Particle 4: along beam axis (z)
];
particles.radius = [50; 50; 50; 50];  % nm
particles.refractiveIndex = [1.5+0.01i; 1.5+0.01i; 1.5+0.01i; 1.5+0.01i];
particles.number = size(particles.position, 1);

fprintf('Particles:\n');
fprintf('  Number: %d\n', particles.number);
fprintf('  Radius: %.1f nm\n', particles.radius(1));
fprintf('  Refractive index: %.2f + %.2fi\n', real(particles.refractiveIndex(1)), imag(particles.refractiveIndex(1)));

% Gaussian beam parameters
initialField.polarization = 'TE';
initialField.beamWidth = 400;  % nm (finite beam)
initialField.focalPoint = [0, 0, 0];  % Focus at origin
initialField.polarAngle = 0;  % Normal incidence (along +z)
initialField.azimuthalAngle = 0;
initialField.amplitude = 1;

fprintf('\nGaussian beam:\n');
fprintf('  Beam width: %.1f nm\n', initialField.beamWidth);
fprintf('  Focal point: [%.1f, %.1f, %.1f]\n', initialField.focalPoint);
fprintf('  Polarization: %s\n', initialField.polarization);
fprintf('  Polar angle: %.1f deg\n', rad2deg(initialField.polarAngle));

% Numerics
lmax = 3;
nmax = 2 * lmax * (lmax + 2);

% For wavebundle integration, need angular grid
% Use similar grid to YASFPY test: [azimuthal, polar] = [15, 40]
numerics.lmax = lmax;
numerics.nmax = nmax;
numerics.polarAnglesArray = linspace(0, pi, 40)';  % 40 polar angles
numerics.azimuthalAnglesArray = linspace(0, 2*pi, 16);  % 15+1 azimuthal (periodic)
numerics.azimuthalAnglesArray = numerics.azimuthalAnglesArray(1:end-1)';  % Remove duplicate

fprintf('\nNumerics:\n');
fprintf('  lmax: %d\n', lmax);
fprintf('  nmax: %d\n', nmax);
fprintf('  Polar angles: %d points\n', length(numerics.polarAnglesArray));
fprintf('  Azimuthal angles: %d points\n', length(numerics.azimuthalAnglesArray));

%% Create simulation structure (CELES format)
simulation.input.wavelength = wavelength;
simulation.input.mediumRefractiveIndex = mediumRefractiveIndex;
simulation.input.k_medium = k_medium;

simulation.input.particles.positionArray = particles.position;
simulation.input.particles.radiusArray = particles.radius;
simulation.input.particles.refractiveIndexArray = particles.refractiveIndex;
simulation.input.particles.number = particles.number;

simulation.input.initialField = initialField;

simulation.numerics.lmax = numerics.lmax;
simulation.numerics.nmax = numerics.nmax;
simulation.numerics.polarAnglesArray = numerics.polarAnglesArray;
simulation.numerics.azimuthalAnglesArray = numerics.azimuthalAnglesArray;

% Device array function (CPU version)
simulation.numerics.deviceArray = @(x) x;

%% Compute initial field coefficients using CELES
fprintf('\nComputing initial field coefficients...\n');

try
    % Call CELES wavebundle function
    aI = initial_field_coefficients_wavebundle_normal_incidence(simulation);

    fprintf('  ✓ Computation successful\n');
    fprintf('  Initial field coefficients shape: [%d, %d]\n', size(aI));
    fprintf('  Max magnitude: %.3e\n', max(abs(aI(:))));

    % Convert to double precision if needed
    initial_field_coefficients = double(aI);

catch ME
    fprintf('  ✗ Error computing initial field coefficients:\n');
    fprintf('    %s\n', ME.message);
    rethrow(ME);
end

%% Also compute plane wave for comparison
fprintf('\nComputing plane wave for comparison...\n');

% Create plane wave configuration (same as above but beamWidth = 0)
initialField_pw = initialField;
initialField_pw.beamWidth = 0;  % Triggers plane wave

simulation_pw = simulation;
simulation_pw.input.initialField = initialField_pw;

try
    % Call CELES plane wave function
    aI_pw = initial_field_coefficients_planewave(simulation_pw);

    fprintf('  ✓ Plane wave computation successful\n');
    fprintf('  Max magnitude: %.3e\n', max(abs(aI_pw(:))));

    initial_field_coefficients_planewave = double(aI_pw);

catch ME
    fprintf('  ✗ Error computing plane wave:\n');
    fprintf('    %s\n', ME.message);
    % Continue anyway - plane wave is optional
    initial_field_coefficients_planewave = [];
end

%% Save test data
fprintf('\nSaving test data...\n');

% Ensure output directory exists
if ~exist('../data', 'dir')
    mkdir('../data');
end

output_file = '../data/wavebundle_test_data.mat';

% Create variables to save (MATLAB requires variables exist in workspace)
particles_position = particles.position;
particles_radius = particles.radius;
particles_refractiveIndex = particles.refractiveIndex;
particles_number = particles.number;

initial_field_beam_width = initialField.beamWidth;
initial_field_focal_point = initialField.focalPoint;
initial_field_polar_angle = initialField.polarAngle;
initial_field_azimuthal_angle = initialField.azimuthalAngle;
initial_field_polarization = initialField.polarization;
initial_field_amplitude = initialField.amplitude;

polar_angles = numerics.polarAnglesArray;
azimuthal_angles = numerics.azimuthalAnglesArray;

% Save all variables (use -v7 for scipy compatibility, not -v7.3)
save(output_file, ...
    'wavelength', ...
    'mediumRefractiveIndex', ...
    'k_medium', ...
    'lmax', ...
    'nmax', ...
    'particles_position', ...
    'particles_radius', ...
    'particles_refractiveIndex', ...
    'particles_number', ...
    'initial_field_beam_width', ...
    'initial_field_focal_point', ...
    'initial_field_polar_angle', ...
    'initial_field_azimuthal_angle', ...
    'initial_field_polarization', ...
    'initial_field_amplitude', ...
    'polar_angles', ...
    'azimuthal_angles', ...
    'initial_field_coefficients', ...
    'initial_field_coefficients_planewave', ...
    '-v7');

fprintf('  ✓ Saved to: %s\n', output_file);
fprintf('  File size: %.2f KB\n', dir(output_file).bytes / 1024);

%% Summary
fprintf('\n=======================================================\n');
fprintf('TEST DATA GENERATION COMPLETE\n');
fprintf('=======================================================\n');
fprintf('\nTo validate YASFPY implementation:\n');
fprintf('  1. Copy %s to tests/data/\n', output_file);
fprintf('  2. Run: pytest tests/test_wavebundle_validation.py -v\n');
fprintf('\n');
