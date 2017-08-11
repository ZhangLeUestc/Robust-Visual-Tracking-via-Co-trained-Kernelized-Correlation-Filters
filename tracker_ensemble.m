% tracker_ensemble: Correlation filter tracking with convolutional features
%
% Input:
%   - video_path:          path to the image sequence
%   - img_files:           list of image names
%   - pos:                 intialized center position of the target in (row, col)
%   - target_sz:           intialized target size in (Height, Width)
% 	- padding:             padding parameter for the search area
%   - lambda:              regularization term for ridge regression
%   - output_sigma_factor: spatial bandwidth for the Gaussian label
%   - interp_factor:       learning rate for model update
%   - cell_size:           spatial quantization level
%   - show_visualization:  set to True for showing intermediate results
% Output:
%   - positions:           predicted target position at each frame
%   - time:                time spent for tracking
%
%   It is provided for educational/researrch purpose only.
%   If you find the software useful, please consider cite our paper.
%
%   Hierarchical Convolutional Features for Visual Tracking
%   Chao Ma, Jia-Bin Huang, Xiaokang Yang, and Ming-Hsuan Yang
%   IEEE International Conference on Computer Vision, ICCV 2015
%
% Contact:
%   Chao Ma (chaoma99@gmail.com), or
%   Jia-Bin Huang (jbhuang1@illinois.edu).


function [positions,res, time] = tracker_ensemble(video_path, img_files, pos, target_sz, ...
    padding, lambda, output_sigma_factor, interp_factor, cell_size, show_visualization)


% ================================================================================
% Environment setting
% ================================================================================

%indLayers = [37, 28, 19];   % The CNN layers Conv5-4, Conv4-4, and Conv3-4 in VGG Net
indLayers = [37,28]; 
%nweights  = [1, 0.5, 0.02]; % Weights for combining correlation filter responses
numLayers = length(indLayers);
show_visualization=0;
% Get image size and search window size
im_sz     = size(imread([video_path img_files{1}]));
window_sz = get_search_window(target_sz, im_sz, padding);
%beta=exp(-4); 
beta=0;

lambda1=10^(-5);
lambda2=10^(-3);
gamma=2*beta+1;

% Compute the sigma for the Gaussian function label
output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;

%create regression labels, gaussian shaped, with a bandwidth
%proportional to target size    d=bsxfun(@times,c,[1 2]);

l1_patch_num = floor(window_sz/ cell_size);

% Pre-compute the Fourier Transform of the Gaussian function label
yf = fft2(gaussian_shaped_labels(output_sigma, l1_patch_num));

% Pre-compute and cache the cosine window (for avoiding boundary discontinuity)
cos_window = hann(size(yf,1)) * hann(size(yf,2))';

% Create video interface for visualization
if(show_visualization)
    update_visualization = show_video(img_files, video_path);
end

% Initialize variables for calculating FPS and distance precision
time      = 0;
positions = zeros(numel(img_files), 2);
%nweights  = reshape(nweights,1,1,[]);

% Note: variables ending with 'f' are in the Fourier domain.


% ================================================================================
% Start tracking
% ================================================================================
for frame = 1:numel(img_files),
   
    im = imread([video_path img_files{frame}]); % Load the image at the current frame
    if ismatrix(im)
        im = cat(3, im, im, im);
    end
    
    tic();
    % ================================================================================
    % Predicting the object position from the learned object model
    % ================================================================================
    if frame > 1
        % Extracting hierarchical convolutional features
        zf= extractFeature(im, pos, window_sz, cos_window, indLayers);
        zf1=fft2(zf{1});
        zf2=fft2(zf{2});
% 	    kzf1 = gaussian_correlation(zf1, model_xf1, kernel.sigma);
%         kzf2 = gaussian_correlation(zf2, model_xf2, kernel.sigma);
        kzf1 = linear_correlation(zf1, model_xf1);
        kzf2 = linear_correlation(zf2, model_xf2);
		response1 = real(ifft2(model_alphaf1 .* kzf1));  %equation for fast detection
        response2 = real(ifft2(model_alphaf2 .* kzf2)); 
	    [response,p1,p2]=choose_response(response1,response2);
		[vert_delta, horiz_delta] = find(response == max(response(:)), 1);
		if vert_delta > size(zf1,1) / 2,  %wrap around to negative half-space of vertical axis
				vert_delta = vert_delta - size(zf1,1);
		end
		if horiz_delta > size(zf1,2) / 2,  %same for horizontal axis
				horiz_delta = horiz_delta - size(zf1,2);
		end
		pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1];
    end
    xf  = extractFeature(im, pos, window_sz, cos_window, indLayers);
   
    xf1=fft2(xf{1});
    xf2=fft2(xf{2});
%     kf1 = gaussian_correlation(xf1, xf1, kernel.sigma);
%     kf2 = gaussian_correlation(xf2, xf2, kernel.sigma);
      kf1 = linear_correlation(xf1, xf1);
      kf2 = linear_correlation(xf2, xf2);
%       alphaf1=((1+4*beta)*kf2+lambda)./((4*beta+1)*kf1.*kf2+lambda*(2*beta+1)*(kf1+kf2)+lambda^2).*yf;
%       alphaf2=((1+4*beta)*kf1+lambda)./((4*beta+1)*kf2.*kf1+lambda*(2*beta+1)*(kf2+kf1)+lambda^2).*yf;
%         alphaf1=yf./(kf1+lambda+2*beta*lambda*(kf1-kf2)./((1+4*beta)*kf2+lambda));
%         alphaf2=yf./(kf2+lambda+2*beta*lambda*(kf2-kf1)./((1+4*beta)*kf1+lambda));
        alphaf1=((1+4*beta)*kf2+lambda2)./((4*beta+1)*kf1.*kf2+(2*beta+1)*(lambda2*kf1+lambda1*kf2)+lambda1*lambda2).*yf;
        alphaf2=((1+4*beta)*kf1+lambda1)./((4*beta+1)*kf2.*kf1+(2*beta+1)*(lambda1*kf2+lambda2*kf1)+lambda1*lambda2).*yf;
        if frame == 1,  %first frame, train with a single image
			model_alphaf1 = alphaf1;
			model_xf1 = xf1;
            model_alphaf2 = alphaf2;
			model_xf2 = xf2;
		else
			%subsequent frames, interpolate model
            
                model_alphaf1 = (1 - interp_factor) * model_alphaf1 +interp_factor * alphaf1;
                model_xf1 = (1 - interp_factor) * model_xf1 + interp_factor * xf1;
                model_alphaf2 = (1 - interp_factor) * model_alphaf2 + interp_factor * alphaf2;
                model_xf2 = (1 -  interp_factor) * model_xf2 +  interp_factor* xf2;
%             if p1>6&&  p2>6               
% 			model_alphaf1 = (1 - p1*interp_factor/10) * model_alphaf1 +p1* interp_factor/10 * alphaf1;
%            
% 			model_xf1 = (1 - p1*interp_factor/10) * model_xf1 + p1*interp_factor/10 * xf1;
%            
%             model_alphaf2 = (1 - p2*interp_factor/10) * model_alphaf2 +  p2*interp_factor/10 * alphaf2;
%             model_xf2 = (1 -  p2*interp_factor/10) * model_xf2 + p2* interp_factor/10 * xf2;
%             end
		end
    % ================================================================================
    % Save predicted position and timing
    % ================================================================================
    positions(frame,:) = pos;
    time = time + toc();
    box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
    res(frame,:)=box;  
    % Visualization
    if show_visualization,
        box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        stop = update_visualization(frame, box);
        if stop, break, end  %user pressed Esc, stop early
        drawnow
        % 			pause(0.05)  % uncomment to run slower
    end
end

end
function feat  = extractFeature(im, pos, window_sz, cos_window, indLayers)

% Get the search window from previous detection
patch = get_subwindow(im, pos, window_sz);
% Extracting hierarchical convolutional features
feat  = get_features(patch, cos_window, indLayers);

end