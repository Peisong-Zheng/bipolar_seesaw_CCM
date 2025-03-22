function [data_pre] = fil_pre(data)
% the data must have two columns
% the first column is the time
% thhe second column is the proxy

        t=(ceil(data(1,1)):100:floor(data(end,1)))';
        data=[t,interp1(data(:,1),data(:,2),t)];
        fs= 10;
        [b,a]   = butter(4,0.1/(fs/2),'low');
        data_low = [t,filtfilt(b,a,data(:,2))];    % first, remove the high-frequency components

%         t=(ceil(data_low(1,1)):1000:floor(data_low(end,1)))';
        t=(1000:1000:floor(data_low(end,1)))';
        data_low=[t,interp1(data_low(:,1),data_low(:,2),t)];
        freq    = [0.04,0.06];
        fs= 1;
        [b,a]   = butter(4,freq/(fs/2),'bandpass');
        data_pre = [t,filtfilt(b,a,data_low(:,2))];
end

