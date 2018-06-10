function dir = qmkdir(dir)
%打开文件的函数
[success, message] = mkdir(dir);  %#ok<NASGU>
