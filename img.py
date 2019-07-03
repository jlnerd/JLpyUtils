from __init__ import *
import skimage, skimage.transform, skimage.restoration, skimage.measure
import matplotlib.pyplot as plt
import matplotlib as mpl
#    from transform import rescale, resize, downscale_local_mean
    

def resize_img(img, y_size, x_size):
    return skimage.transform.resize(img, (y_size,x_size),mode= 'reflect')

def denoise_img(img):
    return skimage.restoration.denoise_tv_chambolle(img)

def build_crop_array(img,yx_min,yx_max,padding):
    y_min_index = max(yx_min[0]-padding,0)
    x_min_index = max(yx_min[1]-padding,0)
    
    y_max_index = min(yx_max[0]+padding,img.shape[0])
    x_max_index = min(yx_max[1]+padding,img.shape[1])
    
    crop_array = [y_min_index, y_max_index, x_min_index, x_max_index]
    
    return crop_array

def find_img_contours_and_cropping_array(img, contour_level = 0.1, padding = 50):
    
    # Find contours
    contours = skimage.measure.find_contours(img, level = contour_level)
    
    if contours == []:
        yx_max = [img.shape[0]-1,img.shape[1]-1]
        yx_min = [0,0]
    else:
        #get corner indices
        yx_max = np.array([[contours[i][:, 0].max(), contours[i][:, 1].max()] for i in range(len(contours))])
        yx_max = [int(yx_max[:,0].max()),int(yx_max[:,1].max())]

        yx_min = np.array([[contours[i][:, 0].min(), contours[i][:, 1].min()] for i in range(len(contours))])
        yx_min = [int(yx_min[:,0].min()), int(yx_min[:,1].min())]
    
    #Build Cropping array  
    crop_array = build_crop_array(img,yx_min,yx_max,padding)
    
    return contours, crop_array

def preprocess_img(img,
                  y_size_resize1 = 512,
                  y_size_resize2 = 256,
                   plot_steps = False
                  ):
    
    if plot_steps == True:
        plt.subplot(421)
        plt.imshow(img)
        plt.title('(1) original img')
    
    #fetch gray scale img
    img_gray = skimage.color.rgb2gray(img)
    if plot_steps == True:
        plt.subplot(422)
        plt.imshow(img_gray, cmap = 'binary')
        plt.title('(2) gray img')

    #perform 1st resize
    y_size = y_size_resize1
    img_resized = resize_img(img,
                             y_size=y_size,
                             x_size = int(y_size*img.shape[1]/img.shape[0]))
    img_gray_resized = resize_img(img_gray,
                                  y_size=y_size,
                                  x_size = int(y_size*img.shape[1]/img.shape[0]))
    
    if plot_steps == True:
        plt.subplot(423)
        plt.imshow(img_gray_resized,cmap='binary')
        plt.title('(3) scaled for y_size = '+str(y_size))
    
    #denoise img
    img_gray_resized_denoised = denoise_img(img_gray_resized)
    if plot_steps == True:
        plt.subplot(424)
        plt.imshow(img_gray_resized_denoised,cmap = 'binary')
        plt.title('(4) denoised img')
    
    contours, crop_array = find_img_contours_and_cropping_array(img_gray_resized_denoised,
                                                             contour_level = 0.1,
                                                             padding = 50)

    if plot_steps == True:
        plt.subplot(425)
        plt.imshow(img_gray_resized_denoised, interpolation='nearest', cmap='binary')
        for n, contour in enumerate(contours):
             plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color = 'r')
        plt.plot(crop_array[2:4],crop_array[0:2],'bo')
        plt.title('(5) Cropping pts: '+str(crop_array))
    
    #crop images
    img_gray_resized_cropped = img_gray_resized[crop_array[0]:crop_array[1],crop_array[2]:crop_array[3]]
    img_resized_cropped = img_resized[crop_array[0]:crop_array[1],crop_array[2]:crop_array[3]]
    
    if plot_steps == True:
        plt.subplot(426)
        plt.imshow(img_resized_cropped)
        plt.title('(6) cropped img')
    
    #resize the cropped image
    y_size = y_size_resize2
    img_resized_cropped_resized = resize_img(img_resized_cropped,
                                             y_size=y_size,
                                             x_size = y_size)
    
    if plot_steps == True:
        plt.subplot(427)
        plt.imshow(img_resized_cropped_resized)
        plt.title('(7 (final)) resized for xy_size = '+str(y_size))
    
    return img_resized_cropped_resized


def auto_crop_img(img, 
                  padding = 50,
                  show_plots = {'processed':True, 
                                'processing steps':False},
                  use_square=False):
    
    img = img/img.max()
    img_gray = skimage.color.rgb2gray(img)

    contours, crop_array = find_img_contours_and_cropping_array(img_gray,
                                                         contour_level = img.max()/10,
                                                         padding = padding)

    if use_square:
        mean_width = np.mean((crop_array[1]-crop_array[0],crop_array[3]-crop_array[2]))
        x_offset = mean_width - (crop_array[1]-crop_array[0])
        y_offset = mean_width - (crop_array[3]-crop_array[2])
        
        crop_array[0] = crop_array[0]-int(x_offset/2)
        crop_array[1] = crop_array[1]+int(x_offset/2)
        crop_array[2] = crop_array[2]-int(y_offset/2)
        crop_array[3] = crop_array[3]+int(y_offset/2)

    img_cropped = img[crop_array[0]:crop_array[1],crop_array[2]:crop_array[3]]


    if show_plots['processing steps']:
        #original image
        plt.title(title+'\noriginal img')
        plt.imshow(img)
        plt.grid(which='both')
        plt.axis('off')
        plt.show()

        #gray with cropping points and contours
        plt.imshow(img_gray, interpolation='nearest', cmap='binary')
        for n, contour in enumerate(contours):
             plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color = 'r')
        plt.plot(crop_array[2:4],crop_array[0:2],'bo')
        plt.title('Cropping pts: '+str(crop_array))
        plt.grid(which='both')
        plt.axis('off')
        plt.show()

    if show_plots['processed']:
        plt.title(img.shape)
        plt.imshow(img_cropped)
        plt.grid(which='both')
        plt.axis('off')
        plt.show()
    
    return img_cropped

def eval_pixel_yield(img, 
                     n_LEDs_per_Row, 
                     failed_pixel_mean_offset_frcn = (0.3,np.inf),
                     show_plots = True):

    img_gray = skimage.color.rgb2gray(img)
    img_gray = img_gray / img_gray.max()
    
    # calculate downscale factor
    downscale_factor = int(img_gray.shape[0]/n_LEDs_per_Row)

    img_gray_downscale = skimage.transform.downscale_local_mean(img_gray, 
                                                                factors=(downscale_factor,downscale_factor))

    if np.min(img_gray_downscale.shape)>2:
        img_gray_downscale = img_gray_downscale[1:-1,1:-1]
    
    img_gray_downscale_flat = img_gray_downscale.flatten()
    
    #fit norm dist. model
    model = scipy.stats.norm
    mean, sigma = model.fit(img_gray_downscale_flat)
    
    # eval pixel count summaries
    yield_stats_dict = {}
    yield_stats_dict['n_pixels'] = len(img_gray_downscale_flat)
    yield_stats_dict['n_failed_pixels'] = len(img_gray_downscale_flat[~np.logical_and(img_gray_downscale_flat>(mean*(1-failed_pixel_mean_offset_frcn[0])), 
                                                                img_gray_downscale_flat<mean*(1+failed_pixel_mean_offset_frcn[1]))])
    yield_stats_dict['n_passed_pixels'] = yield_stats_dict['n_pixels'] - yield_stats_dict['n_failed_pixels']
    
    yield_stats_dict['yield'] = yield_stats_dict['n_passed_pixels'] /  yield_stats_dict['n_pixels']

    
    x = np.linspace(img_gray_downscale_flat.min(), 
                    img_gray_downscale_flat.max(), 100)
    
    print(yield_stats_dict)

    #img_gray_rescale = skimage.transform.resize(img_gray, output_shape=(30, 30),)
    img_failed_pixels = np.ones_like(img_gray_downscale)

    #map on failed pixels
    img_failed_pixels[~np.logical_and(img_gray_downscale>(mean*(1-failed_pixel_mean_offset_frcn[0])), 
                                      img_gray_downscale<(mean*(1+failed_pixel_mean_offset_frcn[1])))] = 0

    fig, ax_list = plt.subplots(1,4)
    if show_plots:
        
        counts, bins, _ = ax_list[0].hist(img_gray_downscale_flat, 
                                   bins = 100, 
                                   density=True) 
                                   #label='pixel int')

        ax_list[0].plot(x, model.pdf(x, mean, sigma))#, label = 'model')
        ax_list[0].vlines(mean*(1-failed_pixel_mean_offset_frcn[0]), 0, counts.max(), linestyles='--', color='grey')
        ax_list[0].vlines(mean*(1+failed_pixel_mean_offset_frcn[1]), 0, counts.max(), linestyles='--', color='grey', 
                   label = 'thresholds: '+str(failed_pixel_mean_offset_frcn))
        ax_list[0].grid(which='both')
        ax_list[0].legend()
        ax_list[0].set_xlabel('normalized\nimg_gray_downscale')

        ax_list[1].imshow(img_cropped)
        ax_list[1].grid(which='both', visible=False)
        
        ax_list[2].imshow(img_gray_downscale)
        ax_list[2].grid(which='both', visible=False)
        
        ax_list[3].imshow(img_failed_pixels, vmin=0, vmax = 1)
        ax_list[3].grid(which='both', visible=False)
        
        plt.tight_layout(rect= [0,0,3,1])
        plt.show()
        
    return yield_stats_dict

