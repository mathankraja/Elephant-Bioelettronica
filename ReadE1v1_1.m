% Demo script for reading data from E1
%
% The scritp generates the command to send, open a COM port and send it.
% Then, for a predefined number of times, plots the signals received
% and finally stops the data transfer and close the COM port.
%
% v.1.0
% First version
%
% September 17th 2025
% OT Bioelettronica

close all;
clear all;
fclose all;

% ----------------- Change these parameters as you need ----------------- %
PlotTime = 10;       % Plot time in s
Refresh = 0.2;     % How ofter data is updated
NumCyc = 1;         % Plot cycles
Mode = 0;           % Working mode: 0=EEG, 1=Not Used, 2=Impedance, 3=Test mode

sampFreq = 250;
NumChEEG = 32;

Notch = 1;          % Notch filter: 0=no filter, 1=50Hz, 2=60Hz, 3=Not used

Offset = 1;         % Plot offset in mV
ConvFact = 0.000286; % Conversion factor to get EEG signals in mV
ConvFactEDA = 3.3/4096*1000; % Conversion factor to get EDA signals in mV
ConvFactTemp = 0.0078125; % Conversion factor to get EDA signals in mV
NumChAcc = 6;
SerialCOM = 'COM13'; % !!! ATTENTION: Set the proper COM !!!

% Start data transfer by setting the GO bit = 1
ControlBytes(1) = 170;
ControlBytes(2) = Mode*2 + 1;
ControlBytes(3) = 85;
% Estimates the total number of channels received depending on the mode
NumChTot = NumChEEG + NumChAcc;

% Number of samples for each plot cycle
NumSamples = sampFreq * Refresh;
RefreshReal = NumSamples/sampFreq;
NumPlot = floor(PlotTime*sampFreq/NumSamples);
data = zeros(NumChTot, NumSamples+1);

Intrf = serialport(SerialCOM, 460800, "timeout", 8);

% Send the command string
write(Intrf, ControlBytes, 'uint8');

% Time estimation
tstart = tic;

ChInd = (1:3:NumChTot*3);


figure('units','normalized','outerposition',[0 0 1 1]);

for i=1:NumCyc

    i
    clf

    for j=1:NumPlot

        % Read data from the interface
        Temp = read(Intrf, NumChTot*3*NumSamples,'uint8');
        Temp1 = reshape(Temp, NumChTot*3, NumSamples);

        % Combine 3 bytes to a 24 bit value
        data(:,2:end) = Temp1(ChInd,:)*65536 + Temp1(ChInd+1,:)*256 + Temp1(ChInd+2,:);

        InSineEDA = floor(data(35,:)/4096);
        OutSineEDA = rem(data(35,:),4096);

        % Search for the negative values and make the two's complement
        ind = find(data >= 8388608);
        data(ind) = data(ind) - (16777216);

        if(j==1)
            data(:,1) = data(:,2);
        end

        t = linspace(RefreshReal*(j-1), RefreshReal*(j), NumSamples+1);

        subplot(3,1,1)
        hold on
        xlim([0 PlotTime])
        % Plot the EEG
        for k = 1 : 32
            plot(t,data(k,:)*ConvFact + Offset*(k-1), 'k');
        end
        ylabel('EEG (uV)')
        xlabel('Time(s)')

        subplot(3,2,3)
        hold on;
        xlim([0 PlotTime])
        % PPG
        for k = 33 : 34
            plot(t,data(k,:), 'k')
        end
        ylabel('PPG (AU)')
        xlabel('Time(s)')

        % EDA ----------------------
        %Remove average
        InSineEDA = InSineEDA - mean(InSineEDA);
        OutSineEDA = mean(OutSineEDA) - OutSineEDA; % Need to be flipped

        % Estimate RMS
        RawRMS = rms(InSineEDA);
        Rkohm = 21.6*RawRMS/(525-RawRMS);

        subplot(3,2,4)
        hold on;
        xlim([0 PlotTime])
        %plot(t, InSineEDA*ConvFactEDA, 'b')
        plot(t, OutSineEDA*ConvFactEDA, 'k')
        ylabel('EDA Votage Out (mV)')
        xlabel('Time(s)')

        subplot(3,2,5)
        hold on;
        xlim([0 PlotTime])
        plot(t,data(36,:)*ConvFactTemp, 'k')
        ylabel('Temperature (°C)')
        xlabel('Time(s)')
        ylim([25 40]);

        subplot(3,2,6)
        hold on;
        xlim([0 PlotTime])
        plot(t,data(38,:), 'k')
        ylabel('Control Ch2 (AU)')
        xlabel('Time(s)')
        data(:,1) = data(:,end);

        drawnow

    end
end

% Send the stop command string MotemaSens
ControlBytes(1) = 170;
ControlBytes(2) = 0;
ControlBytes(3) = 85;
write(Intrf, ControlBytes, 'uint8');

pause(1);

clear Intrf;