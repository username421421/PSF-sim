%% --- INITIALIZATION AND PARAMETER DEFINITION ---
clear; clc; close all;

% Set the minimum integration angle for the simulation.
theta_min = 0.0; 


% Set the polarization angle of the incident light (0 for linear x-polarization).
phi_P = 0; 

%% --- IMMERSION OIL AND COVERSLIP ABERRATION PARAMETERS ---
% Design values (ideal conditions)
t_gd = 0.17e-3;  n_gd = 1.515; t_id = 0.09e-3; n_id = 1.515; n_sd = 1.515;

% Experimental values (actual conditions)
t_g = 0.17e-3;   n_g = 1.515;
t_s = 1e-6;      n_s = 1.474;
n_i = 1.515;

% Calculate required immersion oil thickness to maintain paraxial focus.
t_i = t_s + n_i * (-(t_s/n_s) - (t_g/n_g) + (t_gd/n_gd) + (t_id/n_id));

%% --- OPTICAL SIMULATION PARAMETERS ---
f = 0.26e-3;        % Focal length of the objective lens [m]
lambda = 780e-9;    % Wavelength of light [m]
NA = 1.4;           % Numerical Aperture of the objective lens
k = 2 * pi / lambda;% Wavenumber in vacuum
theta_max = asin(NA/n_i); % Maximum integration angle

%% --- ABERRATION AND TRANSMISSION MODEL (Nesse Paper) ---
h_1 = t_s + t_g; h_2 = t_s; h_1d = t_gd;

% Ray angles using Snell's Law
theta_3 = @(theta) asin(n_i * sin(theta)/n_s);
theta_2 = @(theta) asin(n_i * sin(theta)/n_g);

% Aberration function
w_abb = @(theta) h_2 * n_s * cos(theta_3(theta)) - h_1*n_i*cos(theta) + h_1d*n_id*cos(asin(n_i * sin(theta)/n_id));

% Fresnel coefficients (Interface 1: oil->glass, Interface 2: glass->sample)
t_1_s = @(theta) (2*n_i*cos(theta))./(n_i*cos(theta)+n_g*cos(theta_2(theta)));
t_1_p = @(theta) (2*n_i*cos(theta))./(n_g*cos(theta)+n_i*cos(theta_2(theta)));
r_1_s = @(theta) (n_i*cos(theta)-n_g*cos(theta_2(theta)))./(n_i*cos(theta)+n_g*cos(theta_2(theta)));
r_1_p = @(theta) (n_g*cos(theta)-n_i*cos(theta_2(theta)))./(n_g*cos(theta)+n_i*cos(theta_2(theta)));
t_2_s = @(theta) (2*n_g*cos(theta_2(theta)))./(n_g*cos(theta_2(theta))+n_s*cos(theta_3(theta)));
t_2_p = @(theta) (2*n_g*cos(theta_2(theta)))./(n_s*cos(theta_2(theta))+n_g*cos(theta_3(theta)));
r_2_s = @(theta) (n_g*cos(theta_2(theta))-n_s*cos(theta_3(theta)))./(n_g*cos(theta_2(theta))+n_s*cos(theta_3(theta)));
r_2_p = @(theta) (n_s*cos(theta_2(theta))-n_g*cos(theta_3(theta)))./(n_s*cos(theta_2(theta))+n_g*cos(theta_3(theta)));

% Phase and multiple reflection terms
beta_2 = @(theta) k * n_g * t_g * cos(theta_2(theta));
beta_2d = @(theta) k * n_gd * t_gd * cos(asin(n_gd * sin(theta)/n_id));
D_2_s = @(theta) 1 + r_1_s(theta) .* r_2_s(theta) .* exp(2i*(beta_2(theta)-beta_2d(theta)));
D_2_p = @(theta) 1 + r_1_p(theta) .* r_2_p(theta) .* exp(2i*(beta_2(theta)-beta_2d(theta)));

% Total transmission functions
T_s = @(theta) t_2_s(theta) .* t_1_s(theta) .* exp(1i*(beta_2(theta)-beta_2d(theta)))./D_2_s(theta);
T_p = @(theta) t_2_p(theta) .* t_1_p(theta) .* exp(1i*(beta_2(theta)-beta_2d(theta)))./D_2_p(theta);

%% Define the pupil field amplitude function e_inf(theta).
% Model: Annular illumination
e_inf = @(theta,phi) 1+0*theta+0*phi;

% --- Incident beam parameters ---
theta_inc = 0.3;    % Incident angle
phi_inc = pi/4;     % Incident direction
beam_width = 0.1;   % Angular width of incident beam

% --- Gaussian-weighted incident beam ---
e_inf = @(theta,phi) exp(1i * k * n_i * f * sin(theta) .* sin(theta_inc) .* cos(phi - phi_inc)) .* ...
                     exp(-((theta - theta_inc).^2 + (phi - phi_inc).^2) / (2 * beam_width^2));
%% --- SIMULATION GRID SETUP ---
range_microns = 2; % Plot range in micrometers (+/- from origin)
num_points = 21;  % Number of points (use odd for a center pixel at 0)

% Create Cartesian coordinate vectors in meters
x_coords = linspace(-range_microns*1e-6, range_microns*1e-6, num_points);
y_coords = linspace(-range_microns*1e-6, range_microns*1e-6, num_points);
z_coords = linspace(-range_microns*1e-6, range_microns*1e-6, num_points);

% Initialize arrays to store the complex electric field components
E_x = zeros(num_points, num_points);
E_y = zeros(num_points, num_points);
E_z = zeros(num_points, num_points);

%% --- CALCULATION FOR AXIAL (XZ) PLANE ---
fprintf('Calculating field distribution for XZ plane (Y=0)...\n');
y_val = 0; % We are in the XZ plane

for i = 1:num_points % Loop over z
    if mod(i, 10) == 0
        fprintf('Progress: %d/%d (%.1f%%)\n', i, num_points, 100*i/num_points);
    end
    for j = 1:num_points % Loop over x
        % Current point in Cartesian coordinates
        x_val = x_coords(j);
        z_val = z_coords(i);
        
        % Convert to cylindrical coordinates for the integrals
        r_val = sqrt(x_val^2 + y_val^2);
        theta_val = acos(z_val/r_val);
        if x_val == 0 && y_val == 0
            phi_val = 0; % Set arbitrary angle for on-axis point
        else
            phi_val = atan2(y_val, x_val);
        end
        % Ex_integrand = @(theta, phi) e_inf(theta,phi) .* cos(theta).^(1/2) .* (T_p(theta) .* cos(theta_3(theta)) .* cos(phi).^2 + T_s(theta).*sin(phi).^2).* ...
        %     exp(1i*r_val*n_i*k.*sin(theta).*sin(theta_val).*cos(phi-phi_val)).*exp(1i*k*(h_2*n_s*cos(theta_3(theta))-n_i*h_1*cos(theta))).*exp(1i*k*n_s*z_val*cos(theta_3(theta))).*sin(theta);
        % 
        % Ey_integrand = @(theta, phi) e_inf(theta,phi) .* cos(theta).^(1/2) .* (T_p(theta) .* cos(theta_3(theta)) .* sin(phi).*cos(phi) + T_s(theta).* sin(phi).*cos(phi)).* ...
        %     exp(1i*r_val*n_i*k.*sin(theta).*sin(theta_val).*cos(phi-phi_val)).*exp(1i*k*(h_2*n_s*cos(theta_3(theta))-n_i*h_1*cos(theta))).*exp(1i*k*n_s*z_val*cos(theta_3(theta))).*sin(theta);
        % 
        % Ez_integrand = @(theta, phi) e_inf(theta,phi) .* cos(theta).^(1/2) .* (-T_p(theta) .* sin(theta_3(theta)) .* cos(phi)).* ...
        %     exp(1i*r_val*n_i*k.*sin(theta).*sin(theta_val).*cos(phi-phi_val)).*exp(1i*k*(h_2*n_s*cos(theta_3(theta))-n_i*h_1*cos(theta))).*exp(1i*k*n_s*z_val*cos(theta_3(theta))).*sin(theta);
        Ex_integrand = @(theta, phi) e_inf(theta,phi) .* cos(theta).^(1/2) .* (T_p(theta) .* cos(theta_3(theta)) .* cos(phi).^2 + T_s(theta).*sin(phi).^2).* ...
            exp(1i*r_val*n_i*k.*sin(theta).*cos(phi-phi_val)).*exp(1i*k*w_abb(theta)).*exp(1i*k*n_s*z_val*cos(theta_3(theta))).*sin(theta);

        Ey_integrand = @(theta, phi) e_inf(theta,phi) .* cos(theta).^(1/2) .* (T_p(theta) .* cos(theta_3(theta)) .* sin(phi).*cos(phi) + T_s(theta).* sin(phi).*cos(phi)).* ...
            exp(1i*r_val*n_i*k.*sin(theta).*cos(phi-phi_val)).*exp(1i*k*w_abb(theta)).*exp(1i*k*n_s*z_val*cos(theta_3(theta))).*sin(theta);

        Ez_integrand = @(theta, phi) e_inf(theta,phi) .* cos(theta).^(1/2) .* (-T_p(theta) .* sin(theta_3(theta)) .* cos(phi)).* ...
            exp(1i*r_val*n_i*k.*sin(theta).*cos(phi-phi_val)).*exp(1i*k*w_abb(theta)).*exp(1i*k*n_s*z_val*cos(theta_3(theta))).*sin(theta);

        % Perform numerical integration
        C = -1i * f * n_i * k/(2*pi);
        E_x(i,j) = C * integral2(Ex_integrand, theta_min, theta_max, 0, 2*pi);
        E_y(i,j) = C * integral2(Ey_integrand, theta_min, theta_max, 0, 2*pi);
        E_z(i,j) = C * integral2(Ez_integrand, theta_min, theta_max, 0, 2*pi);

            

    end
end
fprintf('XZ Plane calculation complete!\n\n');

% Store results for the XZ plane
I_tot_xz = abs(E_x).^2 + abs(E_y).^2 + abs(E_z).^2;

%% --- CALCULATION FOR FOCAL (XY) PLANE ---
fprintf('Calculating field distribution for XY plane...\n');
% Find the z-position of maximum intensity from the XZ plane calculation
[~, max_idx] = max(I_tot_xz(:));
[max_i_z, ~] = ind2sub(size(I_tot_xz), max_idx);
z_focus_val = z_coords(max_i_z);
fprintf('Focal plane set at Z = %.3f um\n', z_focus_val * 1e6);

z_val = z_focus_val; % We are in the XZ plane

for i = 1:num_points % Loop over z
    if mod(i, 10) == 0
        fprintf('Progress: %d/%d (%.1f%%)\n', i, num_points, 100*i/num_points);
    end
    for j = 1:num_points % Loop over x
        % Current point in Cartesian coordinates
        x_val = x_coords(j);
        y_val = y_coords(i);
        
        % Convert to cylindrical coordinates for the integrals
        r_val = sqrt(x_val^2 + y_val^2);
        theta_val = acos(z_val/r_val);
        if x_val == 0 && y_val == 0
            phi_val = 0; % Set arbitrary angle for on-axis point
        else
            phi_val = atan2(y_val, x_val);
        end
        
        % Ex_integrand = @(theta, phi) e_inf(theta,phi) .* cos(theta).^(1/2) .* (T_p(theta) .* cos(theta_3(theta)) .* cos(phi).^2 + T_s(theta).*sin(phi).^2).* ...
        %     exp(1i*r_val*n_i*k.*sin(theta).*sin(theta_val).*cos(phi-phi_val)).*exp(1i*k*(h_2*n_s*cos(theta_3(theta))-n_i*h_1*cos(theta))).*exp(1i*k*n_s*z_val*cos(theta_3(theta))).*sin(theta);
        % 
        % Ey_integrand = @(theta, phi) e_inf(theta,phi) .* cos(theta).^(1/2) .* (T_p(theta) .* cos(theta_3(theta)) .* sin(phi).*cos(phi) + T_s(theta).* sin(phi).*cos(phi)).* ...
        %     exp(1i*r_val*n_i*k.*sin(theta).*sin(theta_val).*cos(phi-phi_val)).*exp(1i*k*(h_2*n_s*cos(theta_3(theta))-n_i*h_1*cos(theta))).*exp(1i*k*n_s*z_val*cos(theta_3(theta))).*sin(theta);
        % 
        % Ez_integrand = @(theta, phi) e_inf(theta,phi) .* cos(theta).^(1/2) .* (-T_p(theta) .* sin(theta_3(theta)) .* cos(phi)).* ...
        %     exp(1i*r_val*n_i*k.*sin(theta).*sin(theta_val).*cos(phi-phi_val)).*exp(1i*k*(h_2*n_s*cos(theta_3(theta))-n_i*h_1*cos(theta))).*exp(1i*k*n_s*z_val*cos(theta_3(theta))).*sin(theta);
        Ex_integrand = @(theta, phi) e_inf(theta,phi) .* cos(theta).^(1/2) .* (T_p(theta) .* cos(theta_3(theta)) .* cos(phi).^2 + T_s(theta).*sin(phi).^2).* ...
            exp(1i*r_val*n_i*k.*sin(theta).*cos(phi-phi_val)).*exp(1i*k*w_abb(theta)).*exp(1i*k*n_s*z_val*cos(theta_3(theta))).*sin(theta);

        Ey_integrand = @(theta, phi) e_inf(theta,phi) .* cos(theta).^(1/2) .* (T_p(theta) .* cos(theta_3(theta)) .* sin(phi).*cos(phi) + T_s(theta).* sin(phi).*cos(phi)).* ...
            exp(1i*r_val*n_i*k.*sin(theta).*cos(phi-phi_val)).*exp(1i*k*w_abb(theta)).*exp(1i*k*n_s*z_val*cos(theta_3(theta))).*sin(theta);

        Ez_integrand = @(theta, phi) e_inf(theta,phi) .* cos(theta).^(1/2) .* (-T_p(theta) .* sin(theta_3(theta)) .* cos(phi)).* ...
            exp(1i*r_val*n_i*k.*sin(theta).*cos(phi-phi_val)).*exp(1i*k*w_abb(theta)).*exp(1i*k*n_s*z_val*cos(theta_3(theta))).*sin(theta);

        % Perform numerical integration

        C = -1i * f * n_i * k/(2*pi);
        E_x(i,j) = C * integral2(Ex_integrand, theta_min, theta_max, 0, 2*pi);
        E_y(i,j) = C * integral2(Ey_integrand, theta_min, theta_max, 0, 2*pi);
        E_z(i,j) = C * integral2(Ez_integrand, theta_min, theta_max, 0, 2*pi);
            

    end
end
fprintf('XY Plane calculation complete!\n');

% Store results for the XY plane
I_tot_xy = abs(E_x).^2 + abs(E_y).^2 + abs(E_z).^2;

%% --- PLOTTING AND ANALYSIS ---
x_microns = x_coords * 1e6;
y_microns = y_coords * 1e6;
z_microns = z_coords * 1e6;

% Normalize intensities
I_tot_xz_norm = I_tot_xz / max(I_tot_xz(:));
I_tot_xy_norm = I_tot_xy / max(I_tot_xy(:));

% I_tot_xz_norm = I_tot_xz;
% I_tot_xy_norm = I_tot_xy;

figure('Position', [50, 50, 1200, 1000]);

% --- Plot XZ Plane Intensity ---
subplot(2, 2, 1);
imagesc(x_microns, z_microns, I_tot_xz_norm);
axis xy; colorbar; colormap(hot);
xlabel('X position (μm)');
ylabel('Z position (μm)');
title('Normalized Intensity |E|² in XZ Plane');
axis equal tight;

% --- Plot XY Plane Intensity ---
subplot(2, 2, 2);
imagesc(x_microns, y_microns, I_tot_xy_norm);
axis xy; colorbar; colormap(hot);
xlabel('X position (μm)');
ylabel('Y position (μm)');
title(sprintf('Normalized Intensity |E|² in XY Plane (at Z=%.2f μm)', z_focus_val*1e6));
axis equal tight;

% --- FWHM Analysis ---
% Find center indices
center_idx = ceil(num_points / 2);

% Axial (Z) profile from XZ plane (along z-axis, x=0)
z_profile = I_tot_xz_norm(:, center_idx);
fwhm_z_microns = fwhm(z_microns, z_profile);

% Lateral (X) profile from XY plane (along x-axis, y=0)
x_profile = I_tot_xy_norm(center_idx, :);
fwhm_x_microns = fwhm(x_microns, x_profile);

% Lateral (Y) profile from XY plane (along y-axis, x=0)
y_profile = I_tot_xy_norm(:, center_idx);
fwhm_y_microns = fwhm(y_microns, y_profile);

% --- Plot Axial (Z) Profile ---
subplot(2, 2, 3);
plot(z_microns, z_profile, 'b-', 'LineWidth', 2);
hold on;
plot(z_microns, 0.5 * ones(size(z_microns)), 'r--', 'LineWidth', 1);
title(sprintf('Axial (Z) Profile (FWHM = %.3f μm)', fwhm_z_microns));
xlabel('Z position (μm)'); ylabel('Normalized Intensity');
grid on; ylim([0, 1.1]);

% --- Plot Lateral (X & Y) Profiles ---
subplot(2, 2, 4);
plot(x_microns, x_profile, 'r-', 'LineWidth', 2, 'DisplayName', sprintf('X Profile (FWHM = %.3f μm)', fwhm_x_microns));
hold on;
plot(y_microns, y_profile, 'g--', 'LineWidth', 2, 'DisplayName', sprintf('Y Profile (FWHM = %.3f μm)', fwhm_y_microns));
plot(x_microns, 0.5 * ones(size(x_microns)), 'k:', 'LineWidth', 1);
title('Lateral Profiles in Focal Plane');
xlabel('Position (μm)'); ylabel('Normalized Intensity');
grid on; ylim([0, 1.1]); legend;

fprintf('\n=== FWHM Analysis Results ===\n');
fprintf('FWHM in Z direction (axial): %.3f μm\n', fwhm_z_microns);
fprintf('FWHM in X direction (lateral): %.3f μm\n', fwhm_x_microns);
fprintf('FWHM in Y direction (lateral): %.3f μm\n', fwhm_y_microns);
%% --- HIGH-RESOLUTION PUPIL PLANE VISUALIZATION ---
% Create high-resolution Cartesian grid
n_interp = 200;
x_interp = linspace(-sin(theta_max), sin(theta_max), n_interp);
y_interp = linspace(-sin(theta_max), sin(theta_max), n_interp);
[X_interp, Y_interp] = meshgrid(x_interp, y_interp);

% Convert back to angular coordinates
R_interp = sqrt(X_interp.^2 + Y_interp.^2);
THETA_interp = asin(min(R_interp, sin(theta_max)));  % Clamp to avoid domain errors
PHI_interp = atan2(Y_interp, X_interp);

% Only keep points within the numerical aperture
valid_mask = R_interp <= sin(theta_max);

% Calculate e_inf for interpolated grid
E_inf_interp = zeros(size(THETA_interp));
E_inf_interp(valid_mask) = e_inf(THETA_interp(valid_mask), PHI_interp(valid_mask));
E_inf_interp(~valid_mask) = NaN;

intensity_interp = abs(E_inf_interp).^2;
phase_interp = angle(E_inf_interp);

% Create smooth visualization
figure('Position', [100, 100, 1400, 600]);

% Intensity
subplot(1, 2, 1);
imagesc(x_interp, y_interp, intensity_interp);
axis xy; axis equal; axis([-sin(theta_max) sin(theta_max) -sin(theta_max) sin(theta_max)]);
colorbar; colormap(gca, 'hot');
xlabel('X (normalized pupil coordinate)');
ylabel('Y (normalized pupil coordinate)');
title('Intensity |e_{inf}|^2 in Pupil Plane');

% Add pupil boundary circle
hold on;
theta_circle = linspace(0, 2*pi, 100);
plot(sin(theta_max) * cos(theta_circle), sin(theta_max) * sin(theta_circle), 'w-', 'LineWidth', 2);

% Phase
subplot(1, 2, 2);
imagesc(x_interp, y_interp, phase_interp);
axis xy; axis equal; axis([-sin(theta_max) sin(theta_max) -sin(theta_max) sin(theta_max)]);
colorbar; colormap(gca, 'hsv');
xlabel('X (normalized pupil coordinate)');
ylabel('Y (normalized pupil coordinate)');
title('Phase arg(e_{inf}) in Pupil Plane');
caxis([-pi, pi]);

% Add pupil boundary circle
hold on;
plot(sin(theta_max) * cos(theta_circle), sin(theta_max) * sin(theta_circle), 'w-', 'LineWidth', 2);

sgtitle('Pupil Field e_{inf} - Smooth Interpolated Surface', 'FontSize', 14, 'FontWeight', 'bold');
%% --- HELPER FUNCTION: fwhm ---
function width = fwhm(x, y)
    x = x(:); y = y(:);
    [max_y, max_idx] = max(y);
    half_max = max_y / 2.0;
    
    left_idx = find(y(1:max_idx) <= half_max, 1, 'last');
    if isempty(left_idx), width = NaN; return; end
    
    right_idx = find(y(max_idx:end) <= half_max, 1, 'first') + max_idx - 1;
    if isempty(right_idx), width = NaN; return; end
    
    x_left = interp1(y(left_idx:left_idx+1), x(left_idx:left_idx+1), half_max);
    x_right = interp1(y(right_idx-1:right_idx), x(right_idx-1:right_idx), half_max);
    
    width = x_right - x_left;
end