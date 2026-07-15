function pl = motley_keenan(grid, loss_db, loss_per_m, tx_px, mpp, opts)
% MOTLEY_KEENAN  Log-distance + multi-wall path loss over a material grid.
%
%   pl = motley_keenan(grid, loss_db, loss_per_m, [x_px y_px], mpp)
%   pl = motley_keenan(..., struct('freq_mhz',3500,'n',2.0,'wall_sat_db',60))
%
% MATLAB port of STEP_2/motley_keenan.py (same semantics, same defaults):
%   PL = FSPL(1 m, f) + 10 n log10(d) + saturate(sum wall/clutter losses)
%   - loss_db(id+1)     : dB charged once per contiguous crossing of id
%   - loss_per_m(id+1)  : dB per meter of path inside id (bulk clutter)
%   - n = 2.0 free space: walls are charged explicitly, n > 2 would
%     double-count the environment
%   - total obstruction loss saturates at wall_sat_db (straight rays
%     over-punish deep shadow; corridors reroute real energy)
%
% grid is uint8 HxW (ids 0..7), tx_px is 0-based float pixel coords (y down),
% mpp = meters per pixel. Returns pl (HxW, dB). ~30-60 s per transmitter.

if nargin < 6, opts = struct(); end
f    = getfield_or(opts, 'freq_mhz', 3500);
n    = getfield_or(opts, 'n', 2.0);
sat  = getfield_or(opts, 'wall_sat_db', 60);
step = getfield_or(opts, 'step_px', 0.6);

[H, W] = size(grid);
grid = double(grid);
[gx, gy] = meshgrid(0:W-1, 0:H-1);
dist_px = hypot(gx - tx_px(1), gy - tx_px(2));
K = ceil(max(dist_px(:)) / step) + 1;
t = linspace(0, 1, K);

fspl1m = 32.44 + 20 * log10(f) - 60;
pl = fspl1m + 10 * n * log10(max(dist_px * mpp, 1));

wall_ids = find(loss_db(:)' > 0) - 1;        % material ids charged per run
bulk_ids = find(loss_per_m(:)' > 0) - 1;     % ids charged per meter

for r = 1:H                                   % one image row of rays at a time
    ex = gx(r, :)'; ey = gy(r, :)';
    xi = min(max(round(tx_px(1) + t .* (ex - tx_px(1))), 0), W - 1);
    yi = min(max(round(tx_px(2) + t .* (ey - tx_px(2))), 0), H - 1);
    mats = grid(yi + 1 + H * xi);             % linear index, (W x K)
    extra = zeros(W, 1);
    for m = wall_ids
        hit = mats == m;
        runs = hit(:, 1) + sum(hit(:, 2:end) & ~hit(:, 1:end-1), 2);
        extra = extra + loss_db(m + 1) * runs;
    end
    spacing_m = dist_px(r, :)' / (K - 1) * mpp;
    for m = bulk_ids
        extra = extra + loss_per_m(m + 1) * sum(mats == m, 2) .* spacing_m;
    end
    pl(r, :) = pl(r, :) + (sat * (1 - exp(-extra / sat)))';
end
end

function v = getfield_or(s, f, d)
if isfield(s, f), v = s.(f); else, v = d; end
end
