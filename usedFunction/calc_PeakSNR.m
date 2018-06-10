function res = calc_PeakSNR(f, g)
F = im2double(imread(f)); % 原始图像
G = im2double(imread(g)); % 需要对比的图像
E = F - G; % 错误信息
N = numel(E); % 假设原始信号处于峰值状态 (|F|=1)
res = 10*log10( N / sum(E(:).^2) );
