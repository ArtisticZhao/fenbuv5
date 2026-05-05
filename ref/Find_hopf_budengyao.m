clc; clear; close all;

%% =========================
% 1. 参数区
% =========================
d1   = 1.0;
d2   = -5.8;
ubar = 1.0;
fu   = -0.5;
tau  = 2.0;
l    = 3.0;           % 改成你的 l

n_list = [1 3 5 7];   % 模态
k_list = [0 ];   % 想画的 k

eps_min = 0.02;
eps_max = tau - 0.02;
num_eps = 220;
eps_grid = linspace(eps_min, eps_max, num_eps);

omega_min = 1e-4;
omega_max = 80;
num_omega = 12000;
max_roots_per_eps = 12;   % 每个 epsilon 最多保留多少个正根

% --- 分支跟踪参数 ---
max_total_branches = 30;  % 全 ε-范围内可能出现过的分支总数上限（不会复用槽位，宽松设）
branch_gap_abs    = 1.5;  % 绝对 ω-跳转容差 |ω(新,ib) - ω(旧,ib)|
branch_gap_rel    = 0.15; % 相对 ω-跳转容差（为该分支 last_ω 的倍数，与 abs 取较大者）
branch_max_fill   = 3;    % 同一分支列内 ≤ 该步数的 NaN 缺口用线性插值填补

%% =========================
% 2. 颜色设置
% =========================
mode_colors = [
    0.0000 0.4470 0.7410;   % n=1 蓝
    0.8500 0.3250 0.0980;   % n=3 橙
    0.4660 0.6740 0.1880;   % n=5 绿
    0.6350 0.0780 0.1840    % n=7 红
];

k_markers = {'o-','s-','d-','^-','v-','>-','<-','p-'};  % 给不同 k 不同点型

%% =========================
% 3. 扫描所有根点
% 保存所有 (n, eps, omega)
% =========================
all_root_points = struct();

for in = 1:length(n_list)
    n = n_list(in);
    mu = n^2 / l^2;

    fprintf('\n=============================\n');
    fprintf('Scanning n=%d, mu=%.8f\n', n, mu);

    roots_cell = cell(length(eps_grid), 1);
    for ie = 1:length(eps_grid)
        e = eps_grid(ie);
        rr = find_roots_for_fixed_eps(e, mu, d1, d2, ubar, fu, tau, ...
            omega_min, omega_max, num_omega, max_roots_per_eps);
        roots_cell{ie} = rr(:)';
    end

    % 跨 ε 用最近邻把根固定到分支列；再补小 NaN 缺口
    omega_mat = assign_branches_track(roots_cell, max_total_branches, ...
                                      branch_gap_abs, branch_gap_rel);
    omega_mat = fill_small_gaps(omega_mat, branch_max_fill);

    % 剔除全为 NaN 的空列
    keep = any(~isnan(omega_mat), 1);
    omega_mat = omega_mat(:, keep);

    all_root_points(in).n = n;
    all_root_points(in).mu = mu;
    all_root_points(in).eps_grid = eps_grid;
    all_root_points(in).omega_mat = omega_mat;

    fprintf('n=%d: total root points = %d, branches kept = %d\n', ...
        n, sum(~isnan(omega_mat(:))), size(omega_mat, 2));
end

%% =========================
% 4. 图1：所有 G=0 根点图
% 在 (epsilon, omega) 平面上画所有根点
% =========================
figure('Color','w'); hold on;
leg_h = []; leg_n = {};

for in = 1:length(all_root_points)
    eg = all_root_points(in).eps_grid;
    om = all_root_points(in).omega_mat;
    color = mode_colors(in,:);
    first = true;

    for ib = 1:size(om, 2)
        col = om(:, ib);
        if all(isnan(col)), continue; end
        h = plot(eg, col, '-', 'Color', color, 'LineWidth', 1.2);
        if first
            leg_h(end+1) = h; %#ok<AGROW>
            leg_n{end+1} = sprintf('n=%d', all_root_points(in).n); %#ok<AGROW>
            first = false;
        else
            set(h, 'HandleVisibility', 'off');
        end
    end
end

xlabel('\epsilon', 'FontSize', 12);
ylabel('\omega', 'FontSize', 12);
title('All root points of G(\omega,\epsilon)=0', 'FontSize', 13);
if ~isempty(leg_h)
    legend(leg_h, leg_n, 'Location', 'eastoutside');
end
grid on; box on;

%% =========================
% 5. 图2：所有 S_{n,k} 根点图
% 对每个根点 (eps, omega) 计算 S_{n,k}
% 画 (epsilon, S_{n,k}) 散点图
% =========================
figure('Color','w'); hold on;
leg_h = []; leg_n = {};

for in = 1:length(all_root_points)
    n  = all_root_points(in).n;
    mu = all_root_points(in).mu;
    eg = all_root_points(in).eps_grid;
    om = all_root_points(in).omega_mat;
    color = mode_colors(in,:);

    for ik = 1:length(k_list)
        k = k_list(ik);
        first = true;

        for ib = 1:size(om, 2)
            col_om = om(:, ib);
            if all(isnan(col_om)), continue; end

            S_curve = nan(size(col_om));
            for ie = 1:length(eg)
                w = col_om(ie);
                if isnan(w), continue; end
                e = eg(ie);
                s = sin_rhs(w, e, mu, d2, ubar, tau);
                c = cos_rhs(w, e, mu, d1, d2, ubar, fu, tau);
                theta = atan2(s, c);
                if theta < 0
                    theta = theta + 2*pi;
                end
                S_curve(ie) = e - (theta + 2*pi*k)/w;
            end

            h = plot(eg, S_curve, '-', 'Color', color, 'LineWidth', 1.2);
            if first
                leg_h(end+1) = h; %#ok<AGROW>
                leg_n{end+1} = sprintf('n=%d, k=%d', n, k); %#ok<AGROW>
                first = false;
            else
                set(h, 'HandleVisibility', 'off');
            end
        end
    end
end

yline(0, '--k', 'LineWidth', 1);
xlabel('\epsilon', 'FontSize', 12);
ylabel('S_{n,k}', 'FontSize', 12);
title('All point values of S_{n,k} computed from all roots', 'FontSize', 13);
if ~isempty(leg_h)
    legend(leg_h, leg_n, 'Location', 'eastoutside');
end
grid on; box on;

%% =========================
% 6. 图3：单独看接近 S=0 的点（更清楚）
% 只保留 |S| < tol 的点，方便看 Hopf 候选
% =========================
tol = 0.05;   % 你可以改小，比如 0.02 / 0.01

figure('Color','w'); hold on;
leg_h = []; leg_n = {};

for in = 1:length(all_root_points)
    n  = all_root_points(in).n;
    mu = all_root_points(in).mu;
    eg = all_root_points(in).eps_grid;
    om = all_root_points(in).omega_mat;
    color = mode_colors(in,:);

    for ik = 1:length(k_list)
        k = k_list(ik);
        first = true;

        for ib = 1:size(om, 2)
            col_om = om(:, ib);
            if all(isnan(col_om)), continue; end

            om_keep = nan(size(col_om));
            for ie = 1:length(eg)
                w = col_om(ie);
                if isnan(w), continue; end
                e = eg(ie);
                s = sin_rhs(w, e, mu, d2, ubar, tau);
                c = cos_rhs(w, e, mu, d1, d2, ubar, fu, tau);
                theta = atan2(s, c);
                if theta < 0
                    theta = theta + 2*pi;
                end
                Sval = e - (theta + 2*pi*k)/w;
                if abs(Sval) < tol
                    om_keep(ie) = w;
                end
            end

            if any(~isnan(om_keep))
                h = plot(eg, om_keep, '-', 'Color', color, 'LineWidth', 1.2);
                if first
                    leg_h(end+1) = h; %#ok<AGROW>
                    leg_n{end+1} = sprintf('n=%d, k=%d', n, k); %#ok<AGROW>
                    first = false;
                else
                    set(h, 'HandleVisibility', 'off');
                end
            end
        end
    end
end

xlabel('\epsilon', 'FontSize', 12);
ylabel('\omega', 'FontSize', 12);
title(sprintf('Root points with |S_{n,k}| < %.3f', tol), 'FontSize', 13);
if ~isempty(leg_h)
    legend(leg_h, leg_n, 'Location', 'eastoutside');
end
grid on; box on;

%% =========================
% 7. 局部函数
% =========================
function roots = find_roots_for_fixed_eps(e, mu, d1, d2, u, fu, tau, ...
    wmin, wmax, num_w, max_roots)

    ws = linspace(wmin, wmax, num_w);
    vals = zeros(size(ws));

    for i = 1:length(ws)
        vals(i) = G(ws(i), e, mu, d1, d2, u, fu, tau);
    end

    roots = [];
    for i = 1:length(ws)-1
        a = ws(i);
        b = ws(i+1);
        fa = vals(i);
        fb = vals(i+1);

        if isnan(fa) || isnan(fb) || isinf(fa) || isinf(fb)
            continue;
        end

        if abs(fa) < 1e-8
            r = a;
            if isempty(roots) || all(abs(roots-r) > 1e-4)
                roots(end+1) = r; %#ok<AGROW>
            end

        elseif fa * fb < 0
            try
                r = fzero(@(w) G(w, e, mu, d1, d2, u, fu, tau), [a,b]);
                if isempty(roots) || all(abs(roots-r) > 1e-4)
                    roots(end+1) = r; %#ok<AGROW>
                    if length(roots) >= max_roots
                        break;
                    end
                end
            catch
            end
        end
    end

    roots = sort(roots);
end

function val = G(w,e,mu,d1,d2,u,fu,t)
    A = e*(t-e)*t/(4*d2*u*mu);
    B = e*(t-e)*t*(d1*mu-fu)^2/(4*d2*u*mu);

    val = A*w^6 + B*w^4 + e*sin(w*t)*w^3 ...
        - (d1*mu-fu)*(t-e+e*cos(w*t))*w^2 ...
        + (2*d2*u*mu/t)*(cos(w*t)-1);
end

function s = sin_rhs(w,e,mu,d2,u,t)
    s = e*(t-e)/(2*d2*u*mu)*w^3 + (e/t)*sin(w*t);
end

function c = cos_rhs(w,e,mu,d1,d2,u,fu,t)
    c = (t-e)/t + (e/t)*cos(w*t) ...
      - e*(t-e)*(d1*mu-fu)/(2*d2*u*mu)*w^2;
end

% ----- 分支跟踪：把每个 ε 切片的根集合按"最近邻"分配到固定的分支列 -----
% 思路：从第 1 个 ε 开始，每个根各占 1 条分支；之后每个新 ε，为每条
% 已激活的分支匹配最近的一个新根（距离阈值 = max(branch_gap_abs,
% branch_gap_rel * 该分支 last_ω) 实现 ω 越大允许越大跳变）。剩下未被
% 匹配的根开成新的分支列。这样真分支始终被钉死在自己的列里，不再因每个
% ε 根数变化而错位。
function omega_mat = assign_branches_track(roots_cell, max_branches, gap_abs, gap_rel)
    n_eps = length(roots_cell);
    omega_mat = nan(n_eps, max_branches);
    last_omega = nan(1, max_branches);
    n_active = 0;

    for ie = 1:n_eps
        roots = roots_cell{ie};
        if isempty(roots)
            continue;
        end
        roots = roots(:)';
        n_new = numel(roots);
        used = false(1, n_new);

        if n_active > 0
            la = last_omega(1:n_active);
            valid_b = ~isnan(la);
            ib_list = find(valid_b);

            if ~isempty(ib_list)
                la_v = la(valid_b)';                       % [|valid|, 1]
                D = abs(la_v - roots);                     % [|valid|, n_new]
                thresh = max(gap_abs, gap_rel * la_v);     % per-branch
                D(D > thresh) = Inf;

                % 贪心全局最近邻匹配（每轮取最小距离配对，配上后划掉行列）
                while true
                    [m, idx] = min(D(:));
                    if isinf(m), break; end
                    [bi, j] = ind2sub(size(D), idx);
                    ib = ib_list(bi);
                    omega_mat(ie, ib) = roots(j);
                    last_omega(ib) = roots(j);
                    used(j) = true;
                    D(bi, :) = Inf;
                    D(:, j) = Inf;
                end
            end
        end

        % 未匹配上的根 → 新开一条分支列
        for j = find(~used)
            if n_active >= max_branches
                break;
            end
            n_active = n_active + 1;
            omega_mat(ie, n_active) = roots(j);
            last_omega(n_active) = roots(j);
        end
    end
end

% ----- 同一分支列内的小 NaN 缺口用线性插值填补 ------
function omega_mat = fill_small_gaps(omega_mat, max_gap)
    n_branch = size(omega_mat, 2);
    for ib = 1:n_branch
        col = omega_mat(:, ib);
        idx = find(~isnan(col));
        if numel(idx) < 2
            continue;
        end
        for k = 1:numel(idx)-1
            i1 = idx(k);
            i2 = idx(k+1);
            gap = i2 - i1 - 1;
            if gap > 0 && gap <= max_gap
                vals = linspace(col(i1), col(i2), gap+2);
                col(i1+1:i2-1) = vals(2:end-1);
            end
        end
        omega_mat(:, ib) = col;
    end
end