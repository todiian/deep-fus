% EXECUTE THIS SCRIPT TO ADD THE DATASETS BELOW ON A datasets_list.txt
% FILE CREATED USING prep_train_set_real.m
% 
% ADD DATA FROM SYSKET4 ACQUISITIONS
% 
% Prepare Training set for functional ultrasound imaging using deep
% learning and convolutional neural networks
%
% Select a number of frames in each dataset. Applies a median
% temporal filter on the power Doppler data.
% 
% Save 5 crops (4 corner plus center; if images are big enough) for each 
% frame and saves the power Doppler frame and the relative beamformed RF 
% data.
% 
% Crops are randomly flipped LR with a probability of 0.5.
% 
% The RF data (X) is composed of the real part of each
% beamformed frame (e.g., R(1), R(2), ..., R(250)).
% 
% TO DO:
% 

clear;
close all;
clc;

% Kernel size of median temporal filter
n_filt = 3;       % MUST BE ODD NUMBER

% Size of saved images - take powers of 2 to speed up processing
n_pix = 96;

% Number of frames to save for each dataset
n_frm_dataset = 1;

% Standard deviation used for noise padding (add noise instead of zeros)
% This value was found by looking at dark regions in a few datasets
xystd = 5e04;

% Plots directory
if ~exist('plots'); mkdir('plots'); end
    
%%%%%
n = 1;
rf_datain{n} = '/home/todiian/Data/20200707T104600';
processed_datain{n} = '/home/todiian/Data/20200707T104600_processed';
save_str{n} = '20200707T104600';

n = 2;
rf_datain{n} = '/home/todiian/Data/20200707T104800';
processed_datain{n} = '/home/todiian/Data/20200707T104800_processed';
save_str{n} = '20200707T104800';

n = 3;
rf_datain{n} = '/home/todiian/Data/20200707T104930';
processed_datain{n} = '/home/todiian/Data/20200707T104930_processed';
save_str{n} = '20200707T104930';

n = 4;
rf_datain{n} = '/home/todiian/Data/20200707T105130';
processed_datain{n} = '/home/todiian/Data/20200707T105130_processed';
save_str{n} = '20200707T105130';

n = 5;
rf_datain{n} = '/home/todiian/Data/20200707T105300';
processed_datain{n} = '/home/todiian/Data/20200707T105300_processed';
save_str{n} = '20200707T105300';

n = 6;
rf_datain{n} = '/home/todiian/Data/20200707T105501';
processed_datain{n} = '/home/todiian/Data/20200707T105501_processed';
save_str{n} = '20200707T105501';

n = 7;
rf_datain{n} = '/home/todiian/Data/20200707T105645';
processed_datain{n} = '/home/todiian/Data/20200707T105645_processed';
save_str{n} = '20200707T105645';

n = 8;
rf_datain{n} = '/home/todiian/Data/20200707T110046';
processed_datain{n} = '/home/todiian/Data/20200707T110046_processed';
save_str{n} = '20200707T110046';

n = 9;
rf_datain{n} = '/home/todiian/Data/20200707T110231';
processed_datain{n} = '/home/todiian/Data/20200707T110231_processed';
save_str{n} = '20200707T110231';

n = 10;
rf_datain{n} = '/home/todiian/Data/20200707T110415';
processed_datain{n} = '/home/todiian/Data/20200707T110415_processed';
save_str{n} = '20200707T110415';

n = 11;
rf_datain{n} = '/home/todiian/Data/20200707T110630';
processed_datain{n} = '/home/todiian/Data/20200707T110630_processed';
save_str{n} = '20200707T110630';

n = 12;
rf_datain{n} = '/home/todiian/Data/20200707T110750';
processed_datain{n} = '/home/todiian/Data/20200707T110750_processed';
save_str{n} = '20200707T110750';

n = 13;
rf_datain{n} = '/home/todiian/Data/20200707T110908';
processed_datain{n} = '/home/todiian/Data/20200707T110908_processed';
save_str{n} = '20200707T110908';

n = 14;
rf_datain{n} = '/home/todiian/Data/20200707T111030';
processed_datain{n} = '/home/todiian/Data/20200707T111030_processed';
save_str{n} = '20200707T111030';

n = 15;
rf_datain{n} = '/home/todiian/Data/20200707T111154';
processed_datain{n} = '/home/todiian/Data/20200707T111154_processed';
save_str{n} = '20200707T111154';

n = 16;
rf_datain{n} = '/home/todiian/Data/20200707T111418';
processed_datain{n} = '/home/todiian/Data/20200707T111418_processed';
save_str{n} = '20200707T111418';

n = 17;
rf_datain{n} = '/home/todiian/Data/20200707T111542';
processed_datain{n} = '/home/todiian/Data/20200707T111542_processed';
save_str{n} = '20200707T111542';

n = 18;
rf_datain{n} = '/home/todiian/Data/20200707T111725';
processed_datain{n} = '/home/todiian/Data/20200707T111725_processed';
save_str{n} = '20200707T111725';

n = 19;
rf_datain{n} = '/home/todiian/Data/20200707T111837';
processed_datain{n} = '/home/todiian/Data/20200707T111837_processed';
save_str{n} = '20200707T111837';

n = 20;
rf_datain{n} = '/home/todiian/Data/20200707T111952';
processed_datain{n} = '/home/todiian/Data/20200707T111952_processed';
save_str{n} = '20200707T111952';

%%%%%%%
if ~exist('datasets_list.txt'); error('A file datasets_list.txt must be present in the folder before executing this script.'); end

% Open text file to store names of datasets
fileID = fopen('datasets_list.txt','a');

% For each dataset in the list
for dataset=1:length(rf_datain)
    fr_id = [];
    
    % Find indices of first and last frame
    proc_frame_dir = dir([processed_datain{dataset} '/fr*']);
    for k=1:length(proc_frame_dir)
        fr_id(k) = str2double(extractBetween(proc_frame_dir(k).name,'fr','.'));
    end
    start_fr = min(fr_id)+floor(n_filt/2);
    end_fr = max(fr_id)-floor(n_filt/2);
    
%     % Number of frames available in the dataset
%     n_avail_frms = end_fr-start_fr+1;
%     frm_dist = round(n_avail_frms/n_frm_dataset);
    
%     fr_idx = [(0:n_frm_dataset-2)*frm_dist+start_fr end_fr];
    
    fr_idx = randi([start_fr end_fr]);
    
    for fr=1:n_frm_dataset
        % Load PD data and compute median filter
        data_load = [];
        filt_idx = (-floor(n_filt/2):floor(n_filt/2))+fr_idx(fr);
        for k=1:n_filt
            tmp = load([processed_datain{dataset} '/fr' num2str(filt_idx(k)) '.mat']);
            data_load(:,:,k) = tmp.powDopp;
        end
        ytmp = median(data_load,3);
        
        % Find size of input image
        [n_z,n_x] = size(ytmp);

        % Load beamformed RF data
        xtmp = zeros(n_z,n_x,250);
        for k=1:250
            tmp = load([rf_datain{dataset} '/fr' num2str(fr_idx(fr)) '/em' num2str(k) '.mat']);
            xtmp(:,:,k) = real(tmp.bmfEm);
        end
        
        % Zero pad input data 
        % Input smaller than output along x
        if n_x<=n_pix && n_z>n_pix
            ytmp2 = zeros(n_z,n_pix);
%             noise = rand(n_z,n_pix);
%             ytmp2 = xystd*noise/std(noise(:));
            start_x = floor((n_pix-n_x)/2)+1;
            ytmp2(:,start_x+(0:n_x-1)) = ytmp;
            ytmp = ytmp2;
            
            xtmp2 = zeros(n_z,n_pix,size(xtmp,3));
%             noise = rand(n_z,n_pix,size(xtmp,3));
%             xtmp2 = xystd*noise/std(noise(:));
            xtmp2(:,start_x+(0:n_x-1),:) = xtmp;
            xtmp = xtmp2;
            
            % Define x,z coordinates for crops [up; down; center]
            xcrop_start = [1 1 1];
            zcrop_start = [1 n_z-n_pix round((n_z-n_pix)/2)];
        end
        
        % Input smaller than output along z
        if n_z<=n_pix && n_x>n_pix
            ytmp2 = zeros(n_pix,n_x);
%             noise = rand(n_pix,n_x);
%             ytmp2 = xystd*noise/std(noise(:));
            start_z = n_pix-n_z;
            ytmp2(start_z+(0:n_z-1),:) = ytmp;
            ytmp = ytmp2;
            
            xtmp2 = zeros(n_pix,n_x,size(xtmp,3));
%             noise = rand(n_pix,n_x,size(xtmp,3));
%             xtmp2 = xystd*noise/std(noise(:));
            xtmp2(start_z+(0:n_z-1),:,:) = xtmp;
            xtmp = xtmp2;
            
            % Define x,z coordinates for crops [left; right; center]
            xcrop_start = [1 n_x-n_pix round((n_x-n_pix)/2)];
            zcrop_start = [1 1 1];
        end
        
        % Input smaller than output along x AND z
        if n_x<=n_pix && n_z<=n_pix
            ytmp2 = zeros(n_pix,n_pix);
%             noise = rand(n_pix,n_pix);
%             ytmp2 = xystd*noise/std(noise(:));
            start_x = floor((n_pix-n_x)/2)+1;
%             start_z = floor((n_pix-n_z)/2)+1;
            start_z = n_pix-n_z;
            ytmp2(start_z+(0:n_z-1),start_x+(0:n_x-1)) = ytmp;
            ytmp = ytmp2;
            
            xtmp2 = zeros(n_pix,n_pix,size(xtmp,3));
%             noise = rand(n_pix,n_pix,size(xtmp,3));
%             xtmp2 = xystd*noise/std(noise(:));
            xtmp2(start_z+(0:n_z-1),start_x+(0:n_x-1),:) = xtmp;
            xtmp = xtmp2;
            
            % Define x,z coordinates for crops [center]
            xcrop_start = [1];
            zcrop_start = [1];
        end
        
        if n_x>n_pix && n_z>n_pix
            % Define x,z coordinates for crops [left-up; right-up;
            % right-down; left-down; center]
            xcrop_start = [1 n_x-n_pix n_x-n_pix 1 round((n_x-n_pix)/2)];
            zcrop_start = [1 1 n_z-n_pix n_z-n_pix round((n_z-n_pix)/2)];
        end
        
        
        for cr=1:length(xcrop_start)
            % Select crop ROI
            y = ytmp(zcrop_start(cr)+(0:n_pix-1), xcrop_start(cr)+(0:n_pix-1));
            x = xtmp(zcrop_start(cr)+(0:n_pix-1), xcrop_start(cr)+(0:n_pix-1), :);
            
            % Random LR flip with probability 0.5
            if rand>0.5
                y = fliplr(y);
                x = fliplr(x);
            end

            % Save parameters of saved data
            params.rf_folder = rf_datain{dataset};
            params.proc_folder = processed_datain{dataset};
            params.select_frame = fr_idx(fr);
            params.x_start = xcrop_start(cr);
            params.z_start = zcrop_start(cr);                

            % Save current crop
            datasetID = [save_str{dataset} '_' num2str(fr) '_' num2str(cr)];
            save([datasetID '.mat'], 'y', 'x', 'params');
            fprintf(fileID,'%s\n',datasetID);
            
            % Plot current crop
            figure(1);
            tmp = 10*log10(y);
            imagesc(tmp-max(tmp(:)), [-35 0]);
            colormap hot;
            colorbar;
            axis image;
            
            % Save current crop plot
            saveas(figure(1),['plots/' datasetID '.png']);
        end
    end
end

fclose(fileID);

return
