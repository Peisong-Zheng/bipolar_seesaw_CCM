clear,clc,close all

% load the precession data of past 500kyr with data gap of 100yr
load precession_0_500kyr_inter100
pre=precession_0_500kyr_inter100;

pre=flipud(pre);
pre(:,1)=abs(pre(:,1))*1000;
t=(1000:1000:floor(pre(end,1)))'; % resample the data to 1000yr resolution
pre=[t,interp1(pre(:,1),pre(:,2),t)];

% % load the obliquity data
% load obliquity_0_800kyr_inter100 
% obl=obliquity_0_800kyr_inter100;
% obl=flipud(obl);
% obl(:,1)=abs(obl(:,1))*1000;

%load the intergrated annual insolation curve from Huybers et al.(2006,
%2008)
% load j65s_t275
% j65s=j65s_t275;
% j65s(:,1)=j65s(:,1)*1000;

% load he EDC d18O on AICC2012 timescale Stenni et al, 2010 QSR; Verse et al., 2013
load AICC2012_edc_d18O_stenni_2010 
edc=AICC2012_edc_d18O_stenni_2010;
edc_or=edc;

% load the ngrip d18O 
load ngrip_modelex_b2k_inter50_Ras2014
ngrip=ngrip_modelex_b2k_inter50_Ras2014;
ngrip(:,1)=ngrip(:,1)-50;

% the timing for greenland warming transitions, which is the start of each
% interstadial and end of previous stadial, data from Rasmussen et al.(2014)
G_wmp=[11703
    14692
    17480
    23020
    23340
    27780
    28900
    30840
    32500
    33740
    35480
    38220
    40160
    41460
    43340
    46860
    49280
    54220
    55000
    55800
    58040
    58280
    59080
    59440
    64100
    69620
    72340
    76440
    84760
    85060
    90040
    104040
    104520
    106750
    108280
    115370];

% the timing for greenland cooling transitions, which is the start of each
% stadial and end of previous interstadial, data from Rasmussen et al.(2014)
G_cmp=[12896
    17480
    22900
    23220
    27540
    28600
    30600
    32040
    33360
    34740
    36580
    39900
    40800
    42240
    44280
    48340
    49600
    54900
    55400
    56500
    58160
    58560
    59300
    63840
    69400
    70380
    74100
    77760
    84960
    87600
    90140
    104380
    105440
    106900
    110640
    119140];

gsd=G_cmp-G_wmp; % Greenland stadial duration
gid=G_wmp(2:end)-G_cmp(1:end-1); %interstadial duration

GS_label={'1';'2.1a';'2.1b-c';'2.2';'3';'4';'5.1';'5.2';'6';'7';'8';'9';'10';'11';'12';'13';'14';'15.1';'15.2';'16.1';'16.2';'17.1';'17.2';'18';'19.1';'19.2';'20';'21.1';'21.2';'22';'23.1';'23.2';'24.1';'24.2';'25';'26'};

%% the following code create a fake Antarctic temperature curve from the sequence of DO events. 
% by assign a fix 'slope' of warming and cooling for Greenland stadial and
% interstadial respectively.
% note that for periods without DO events, a horizontal line is added.
k=2;
h=1;
temp=[];
temp(1,:)=[G_cmp(end),0];

temp_lin(1,:)=[G_cmp(end),0];
temp_exp_node=[G_cmp(end),0];
    
figure
hold on
%     axes('Position',[0.1 0.8 0.8 0.16]);
axis([0 120000 -30 10]);
hold on
for i=1:length(G_cmp)
    rectangle('position',[G_wmp(i),min(ylim),G_cmp(i)-G_wmp(i),max(ylim)-min(ylim)],'linestyle','none','facecolor',[0.6 0.6 0.6]);
    if i<length(G_cmp)
        rectangle('position',[G_cmp(i),min(ylim),G_wmp(i+1)-G_cmp(i),max(ylim)-min(ylim)],'linestyle','none','facecolor',[0.9 0.9 0.9]);
    end
end
for i=1:length(G_cmp)
    text(G_cmp(i),max(ylim),char(GS_label(i)));
end

%     slope_w=interp1(lr04_s(:,1),lr04_s(:,2),G_cmp);
%     slope_c=mean(slope_w)*ones(35,1);

slope_w=0.001*ones(36,1); % warming slope
slope_c=0.001*ones(35,1); % cooling slope

lin_or_exp=1; % this parameter decide whether to use linear temperature correlation or use the thermal seesaw to build up the curve

for i=length(G_cmp):-1:1
    if lin_or_exp==1
        if k==22||k==24||k==64||k==68 % Set the warming rate of MIS 4 (GS-19.1, GS-18) and MIS 2 (GS-2.1b, GS-2.1c, GS-3)to zero, so that to plot horizontal lines for these periods.
            slope_w(h,1)=0;
        else
            slope_w(h,1)=0.001;
        end
        
        
        temp_lin(k,:)=[G_wmp(i),temp_lin(k-1,2)+gsd(i)*slope_w(h,1)];
        plot([G_cmp(i),G_wmp(i)],[temp_lin(k-1,2),temp_lin(k-1,2)+gsd(i)*slope_w(h,1)],'r');
        text(G_cmp(i),temp_lin(k-1,2),num2str(k));
        
        k=k+1;
        if i>=2
            temp_lin(k,:)=[G_cmp(i-1),temp_lin(k-1,2)-gid(i-1)*slope_c(h,1)];
            if gid(i-1)==0
                slope_c(h)=0;
            end
            plot([G_wmp(i),G_cmp(i-1)],[temp_lin(k-1,2),temp_lin(k-1,2)-gid(i-1)*slope_c(h,1)],'b');
            text(G_cmp(i),temp_lin(k-1,2)+5,num2str(h),'color',[1 0 0]);
            
            k=k+1;
            h=h+1;
        end
    end
    
    if lin_or_exp==0
        
        %         ktn_w(h,1)=unifrnd(-5.11548098199696,1.27170522812520,1,1);
        %         ktn_c(h,1)=unifrnd(0.305863598426980,7.46614134125473,1,1);
        ktn_w(h,1)=-1.5; % parameter to set the shape of exp curve of warming
        ktn_c(h,1)=1; % parameter to set the shape of exp curve of cooling
        
        temp_exp_node(k,:)=[G_wmp(i),temp_exp_node(k-1,2)+ktn_w(h)*(exp(-(1/1120)*gsd(i))-1)];
        k=k+1;
        if i>=2
            temp_exp_node(k,:)=[G_cmp(i-1),temp_exp_node(k-1,2)+ktn_c(h)*(exp(-(1/1120)*gid(i-1))-1)];
            if gid(i-1)==0
                ktn_c(h)=0;
            end
            k=k+1;
            h=h+1;
        end
    end
        switch lin_or_exp
        case 1
            temp=temp_lin;
        case 0
            temp=temp_exp_node;
    end
end

xlabel('Time (yr ago)')
ylabel('value')
title('The fake Antarctic temperature curve')

temp=flipud(temp);
temp=[0,temp(1,2);temp]; % enlengthen the record to 0 ka

[~,ia,~]=unique(temp(:,1));
temp=temp(ia,:);

%% compare the fake AT record with the EDC d18O


in_of_in=[11703,119140]; % interval of interest
t=(in_of_in(1):1000:in_of_in(2))';
x_edc=[t,interp1(edc(:,1),edc(:,2),t)];

% x_edc(:,2)=detrend(x_edc(:,2),3);

% t=(ceil(temp(1,1)):100:floor(temp(end,1)))';
% x_temp=[t,interp1(temp(:,1),temp(:,2),t)];
% fs= 10;
% [b,a]   = butter(4,0.1/(fs/2),'low');
% x_temp = [t,filtfilt(b,a,x_temp(:,2))];
% t=(in_of_in(1):1000:in_of_in(2))';

x_temp=[t,interp1(temp(:,1),temp(:,2),t)];
% x_temp(:,2)=detrend(x_temp(:,2),3);

figure
hold on
plot(x_edc(:,1),zscore(x_edc(:,2)));
plot(x_temp(:,1),zscore(x_temp(:,2)));
xlabel('Time (yr ago)')
ylabel('Normalized unit')
title('The fake Antarctic temperature curve VS EDC d18O')

% x_edc=x_edc(x_edc(:,1)>=in_of_in(1)&x_edc(:,1)<=in_of_in(:,2),:);
% pre=pre(pre(:,1)>=in_of_in(1)&pre(:,1)<=in_of_in(:,2),:);
%% plot the cross spectrum coherence and phase between EDC and fake AT
win=[];
overlap=[];

Fs=1;
[Cxy,F1] = mscohere(x_edc(:,2),x_temp(:,2),win,overlap,[],Fs);
figure
subplot(2,1,1)
hold on
plot(F1,Cxy)
plot([1/23,1/23],ylim)
title('EDC VS fake AT');
xlabel('Frequency')
ylabel('Coherence')

[Pxy,F2] = cpsd(x_edc(:,2),x_temp(:,2),win,overlap,[],Fs);
subplot(2,1,2)
hold on
plot(F2,angle(Pxy))
plot([1/23,1/23],ylim)
xlabel('Frequency')
ylabel('phase')

cw_lag=interp1(F2,angle(Pxy),1/(23))/(2*pi)*23;
cw_cohe=interp1(F1,Cxy,1/(23));
%% figure for warming rate and cooling rate = 1 unit pre kyr

figure('Position',[267,0,1060,800])
axes('Position',[0.1 0.8 0.8 0.16]);
axis([0 120000 -47 -30]);
hold on
for i=1:length(G_cmp)
    rectangle('position',[G_wmp(i),-47,G_cmp(i)-G_wmp(i),17],'linestyle','none','facecolor',[0 0 0]);
    if i<length(G_cmp)
        rectangle('position',[G_cmp(i),-47,G_wmp(i+1)-G_cmp(i),17],'linestyle','none','facecolor',[0.6 0.6 0.6]);
    end
end
rectangle('position',[2000,-40,3000,1],'linestyle','none','facecolor',[0 0 0]);
text(2200,-41,'stadial');
rectangle('position',[2000,-43.5,3000,1],'linestyle','none','facecolor',[0.6 0.6 0.6]);
text(2200,-44.5,'interstadial','color',[0.4 0.4 0.4]);

plot(ngrip(:,1),ngrip(:,2),'color',[0 1 1]);
tick=get(gca,'xtick');
set(gca,'xticklabel',num2str(tick'/1000));
xlabel('Age (kyr BP)');
ylabel('NGRIP δ^1^8O');
h = annotation('textbox',[0.06 0.5 0.43 0.46],'String','a','FitBoxToText','on');
set(h,'LineStyle', 'none','fontweight','bold');


axes('Position',[0.1 0.55 0.8 0.16]);
hold on
axis([0 120000 -4 4]);
p2=plot(edc(:,1),zscore(edc(:,2)));
p1=plot(temp(:,1),zscore(temp(:,2)),'color',[0.7 0.7 0.7]);
% plot(obl(:,1),zscore(obl(:,2)),'k');
% plot(j65s(:,1),zscore(j65s(:,2)),'k--');
L=legend([p1,p2],'DO sequence generated','EDC δ^1^8O');
set(L,'NumColumns',2);
tick=get(gca,'xtick');
set(gca,'xticklabel',num2str(tick'/1000));
xlabel('Age (kyr BP)');
ylabel('Normalized unit');
h = annotation('textbox',[0.06 0.5 0 0.22],'String','b','FitBoxToText','on');
set(h,'LineStyle', 'none','fontweight','bold');

% plot Lomb-Scargle periodogram
axes('Position',[0.1 0.3 0.35 0.16]);
hold on
axis([0 0.3 0 32]);
t=(ceil(edc(1,1)):1000:floor(edc(end,1)))';
temp_spa=[t,interp1(edc(:,1),edc(:,2),t)];
temp_spa(:,2)=detrend(temp_spa(:,2),3);
Pfa = 5/100;
Pd = 1-Pfa;
[pxx,f,pth] = plomb(temp_spa(:,2),temp_spa(:,1)./1000, 'normalized','Pd',Pd);
plot([1/41,1/41],ylim,'r')
rectangle('position',[1/23,0,1/19-1/23,max(ylim)],'linestyle','none','facecolor',[0.7 0.7 0.7]);
line(f,pxx)
L=line(f,pth*ones(size(f')));
legend(L,'Threshold of significance');
xlabel('Frequency (1/kyr)')
ylabel('Power')
title('EDC δ^1^8O')
text(0.2*[1],pth+3,[repmat('P_{fa} = ',[1 1]) num2str(Pfa')])
text(1/58,max(ylim)+2,'41 kyr');
text(1/18,max(ylim)-2,'19 to 23 kyr');
xlabel('Frequency (1/kyr)')
ylabel('Power density')
h = annotation('textbox',[0.06 0.47 0  0],'String','c','FitBoxToText','on');
set(h,'LineStyle', 'none','fontweight','bold');

axes('Position',[0.55 0.3 0.35 0.16]);
axis([0 0.1 0 30]);
temp_spa=temp(:,1:2);
temp_spa(:,2)=detrend(temp_spa(:,2),3);
Pfa = 5/100;
Pd = 1-Pfa;
axis([0 0.3 0 20]);
[pxx,f,pth] = plomb(temp_spa(:,2),temp_spa(:,1)./1000, 'normalized','Pd',Pd);
plot([1/41,1/41],ylim,'r')
rectangle('position',[1/23,0,1/19-1/23,max(ylim)],'linestyle','none','facecolor',[0.7 0.7 0.7]);
line(f,pxx)
L=line(f,pth*ones(size(f')));
legend(L,'Threshold of significance');
xlabel('Frequency (1/kyr)')
ylabel('Power density')
title('DO sequence generated')
text(0.2*[1],pth+3,[repmat('P_{fa} = ',[1 1]) num2str(Pfa')])
text(1/58,max(ylim)+1.5,'41 kyr');
text(1/18,max(ylim)-2,'19 to 23 kyr');
h = annotation('textbox',[0.5 0.47 0  0],'String','d','FitBoxToText','on');
set(h,'LineStyle', 'none','fontweight','bold');


% compare the precession component, where the precession component is got
% by a band pass filter between f=[0.04,0.06], (corresponds to period 16.6
% to 25 kyr)
axes('Position',[0.1 0.05 0.8 0.16]);
axis([0 120000 -4 4]);
hold on
temp_pre=fil_pre(temp);
edc_pre=fil_pre(edc);
p2=plot(edc_pre(:,1),zscore(edc_pre(:,2)),'k');
p3=plot(pre(:,1),-zscore(pre(:,2)),'b--');
p1=plot(temp_pre(:,1),zscore(temp_pre(:,2)),'color',[0.7 0.7 0.7]);

L=legend([p1,p2,p3],'precession component, DO sequence generated','precession component, EDC δ^1^8O','-1 times precession');
set(L,'NumColumns',3);
tick=get(gca,'xtick');
set(gca,'xticklabel',num2str(tick'/1000));
xlabel('Age (kyr BP)');
ylabel('Normalized unit');
h = annotation('textbox',[0.06 0.22 0  0],'String','e','FitBoxToText','on');
set(h,'LineStyle', 'none','fontweight','bold');

