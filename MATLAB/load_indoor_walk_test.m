% LOAD_INDOOR_WALK_TEST  Load and preview indoor_walk_test.mat
%
% The .mat bundles four structs, all in shared coordinate frames:
%   floor : material grid (uint8 ids), material names/losses, meters_per_px,
%           affine transforms px -> local meters / lon-lat / EPSG:3857
%   tx    : candidate transmitter positions (green pins), float px + meters
%   sim   : STEP_2 Motley-Keenan path loss maps (dB) and best-server dBm map
%   walk  : scanner walk-test rows normalized into the pixel/meter frames
%           (rsrp_dbm, rsrq_db, cinr_db, pci, freq_mhz, protocol 1=LTE 2=NR)
%
% Pixel coords are 0-based floats, y down. MATLAB is 1-based:
%   value = grid(round(y_px) + 1, round(x_px) + 1)

load(fullfile(fileparts(mfilename('fullpath')), 'indoor_walk_test.mat'), ...
     'floorplan', 'tx', 'sim', 'walk');
mpp = floorplan.meters_per_px;
[H, W] = size(floorplan.material_grid);

figure('Name', 'Indoor Walk Test 7-7', 'Position', [50 50 1250 750]);

% --- material grid -------------------------------------------------------
subplot(3, 1, 1);
imagesc([0 W * mpp], [0 H * mpp], double(floorplan.material_grid));
axis image; colormap(gca, lines(numel(floorplan.material_ids)));
title('STEP 1: material grid');
xlabel('m'); ylabel('m');
cb = colorbar; cb.Ticks = 0.5:1:7.5;
cb.TickLabels = cellstr(floorplan.material_names);

% --- best-server heatmap --------------------------------------------------
subplot(3, 1, 2);
prx = sim.prx_best_server_dbm;
prx(floorplan.outside_mask) = NaN;
imagesc([0 W * mpp], [0 H * mpp], prx, 'AlphaData', ~isnan(prx));
axis image; colormap(gca, turbo); clim([-120 -40]);
hold on;
plot(tx.x_px * mpp, tx.y_px * mpp, 'wv', 'MarkerFaceColor', 'g', ...
     'MarkerSize', 8);
title(sprintf('STEP 2: best server, %.0f MHz, EIRP %.0f dBm', ...
              sim.freq_mhz, sim.eirp_dbm));
xlabel('m'); ylabel('m'); colorbar;

% --- walk-test RSRP on the floor -----------------------------------------
subplot(3, 1, 3);
walls = floorplan.material_grid > 0 & floorplan.material_grid ~= 4;
imagesc([0 W * mpp], [0 H * mpp], ~walls); colormap(gca, gray); hold on;
ok = walk.on_floor & isfinite(walk.rsrp_dbm);
scatter(walk.x_px(ok) * mpp, walk.y_px(ok) * mpp, 8, walk.rsrp_dbm(ok), ...
        'filled');
axis image; clim([-125 -70]); colorbar;
title(sprintf(['walk-test RSRP from outdoor donors (%d of %d points ' ...
               'GPS-land on floor)'], nnz(ok), numel(walk.rsrp_dbm)));
xlabel('m'); ylabel('m');
