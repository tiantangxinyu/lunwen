function res = calc_PeakSNR(f, g)
F = im2double(imread(f)); % ԭʼͼ��
G = im2double(imread(g)); % ��Ҫ�Աȵ�ͼ��
E = F - G; % ������Ϣ
N = numel(E); % ����ԭʼ�źŴ��ڷ�ֵ״̬ (|F|=1)
res = 10*log10( N / sum(E(:).^2) );
