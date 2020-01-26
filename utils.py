# Author: Michael Notter, January 2020
#
# Support script used for the Image Classification hands-on exercise presented
# during the EPFL Extension School Workshop - Machine Learning, at the Applied
# Machine Learning Days 2020, Lausanne, Switzerland. (See: https://appliedmldays.org/workshops/epfl-extension-school-workshop-machine-learning-and-data-visualization)
#
# For more information about this hands-on exercise, and the full code base,
# see: https://github.com/epfl-exts/amdl20-image-classification

import os
import time
import shutil
import hashlib
import warnings
import imageio
import requests
import numpy as np
import pandas as pd
from glob import glob
from tqdm.notebook import tqdm
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
sns.set_context('talk')

from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

from google_images_download import google_images_download

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_filenames(class_labels):
    
    # Create list of filepaths
    imgs = []
    for c in class_labels:
        folder_id = c.replace(' ', '_')
        imgs += glob('downloads/*%s*/amld_?????.jpg' % folder_id)

    # Return a shuffled list of image filepaths
    return shuffle(imgs)


def collect_images(class_labels, suffix='photo,close up,portrait'):

    # Create object to download images
    response = google_images_download.googleimagesdownload()

    # Go through class labels, download images and preprocess them
    for label in class_labels:

        # Specify class identifier
        folder_id = label.replace(' ', '_')

        print('Collecting data about: %s' % folder_id)

        # Specify list of arguments required for the download
        arguments = {
            'keywords': label,
            'limit': 100,
            'output_directory': 'downloads',
            'image_directory': 'images_{}'.format(folder_id),
            'silent_mode': True,
            'print_urls': False,
            'safe_search': True,
            'format': 'jpg',
            'size': 'medium',
            'suffix_keywords': suffix,
            #'color_type': 'full-color',
            #'type': 'photo',
            'usage_rights': 'labeled-for-reuse',
        }

        # Constantly restart download during 60 seconds if it hasn't downloaded anything new
        start = time.time()

        # Count how many images were already downloaded
        current_img_count = len(glob(os.path.join('downloads', 'images_*%s*' % folder_id, '*')))
        new_img_count = current_img_count

        # Only collect images if less than 100 samples are available
        if current_img_count <= 100:
            while time.time() - start < 60 and current_img_count==new_img_count:
                
                # Download the images
                response.download(arguments)

                # Count new image count
                new_img_count = len(glob(os.path.join('downloads', 'images_*%s*' % folder_id, '*')))

        # Collect a list of all files, starting with 'amld_' ones
        imgs = sorted(glob(os.path.join('downloads', 'images_*%s*' % folder_id, '*')))
        imgs_amld = [f for f in imgs if 'amld_' in f]
        imgs_new = [f for f in imgs if 'amld_' not in f]
        imgs = imgs_amld + imgs_new
        
        # Verify integrity of image and stored it as RGB images in JPEG format
        # Loop through all images and try to open them
        for i, img in enumerate(imgs):
            try:
                # Load image and convert to RGB color scheme
                im = Image.open(img).convert(mode='RGB', colors=256)

            except IOError:
                continue

            # Extract size of image
            width, height = im.size

            # Define 'amld_' file name
            filename = os.path.join(os.path.dirname(img), 'amld_%05d.jpg' % (i + 1))

            # If image is not almost squared, resize it, otherwise leave it as it is
            if np.abs(width-height)>5:

                # Extract min dimension for squaring of image
                min_dim = min(width, height)

                # Compute image offset for squaring of image
                offset_x = width - min_dim
                offset_y = height - min_dim

                # Square image
                im_squared = im.crop((offset_x/2., offset_y/2.,
                                        offset_x/2. + min_dim,
                                        offset_y/2. + min_dim))

                # Save preprocessed and squared image
                im_squared.save(filename)

            elif 'amld_' not in img:
                # Store image file with 'amld_' prefix if it doesn't have it yet
                im.save(filename)

                # Remove old file
                if os.path.exists(img):
                    os.remove(img)

    # Return list of images
    imgs = get_filenames(class_labels)

    # Collection Feedback
    print('---\nA Total of N=%d images were collected!' % len(imgs))
    return imgs


def plot_images(imgs, n_col=6, n_row=3, show_histogram=False):

    # Reshuffle image order for every call
    imgs = shuffle(imgs)

    # Plot images without histogram
    if not show_histogram:

        # Visualize the first x images
        plt.figure(figsize=(n_col * 3, n_row * 3))

        for i in range(n_row * n_col):
            ax = plt.subplot(n_row, n_col, i + 1)
            img = imageio.imread(imgs[i])
            plt.imshow(img)
            plt.axis("off")

        plt.subplots_adjust(hspace=0.1, wspace=0.1)
        plt.show()

    # Plot images with histogram
    else:

        for i in range(n_row):

            # Specify figure size
            fig = plt.figure(figsize=(n_col * 2.5, 4))
            gs = gridspec.GridSpec(2, n_col, height_ratios=[1, 4]) 

            for j in range(n_col):

                # Load data
                img = imageio.imread(imgs[n_col * i + j])
                img_array = img.reshape(-1, 3)
                ax = plt.subplot(gs[0, j])
                for idx, c in enumerate(['red', 'green', 'blue']):
                    ax.hist(img_array[..., idx], bins=128, range=(0, 255), color=c, alpha=0.5)
                plt.xticks([])
                plt.yticks([])
                ax = plt.subplot(gs[1, j])
                plt.imshow(img)
                plt.xticks([])
                plt.yticks([])
            plt.tight_layout()
            plt.show()


def file_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def remove_duplicates(imgs):
    
    # Get hash number of all files
    hashidx = [file_hash(i) for i in imgs]

    # Look for duplicates
    mask = pd.Series(hashidx).duplicated().values

    # List of unique images
    imgs_unique = np.array(imgs)[~mask].tolist()

    # Number of duplicates
    n_duplicates = len(imgs) - len(imgs_unique)

    # Report status report
    print('Total number of images in the dataset: {:4}'.format(len(imgs)))
    print('Number of duplicates in the dataset: {:6}'.format(n_duplicates))

    return imgs_unique


def detect_outliers(filename, thr_shift=20, thr_unicolor=0.80):

    # Load image from filename
    img = imageio.imread(filename).reshape(-1, 3)

    # Extract RGB histogram information
    hist = []
    for i in range(3):
        hist.append(np.histogram(img[:, i], bins=256, range=(0, 255))[0])
    hist = np.transpose(hist)

    # Detect different outlier types
    outlier = False

    # Detect extrem histogram shifts within any of the 3 color channels
    if np.sum(100*hist.T/np.prod(img.shape[0])>thr_shift):
        outlier = True

    # Detect images with high amount of unicolor regions
    elif np.std(100 * hist / np.sum(hist, axis=0)) >= thr_unicolor:
        outlier = True

    # Detect gray scaled images
    elif np.corrcoef(hist.T).mean() >= 0.99:
        outlier = True

    # Detect images with low RGB variance and range
    else:
        corr = np.corrcoef(hist.T)
        np.fill_diagonal(corr, 0)
        corr_mean = corr[corr!=0].mean()
        corr_std = corr[corr!=0].std()
        corr_range = corr[corr!=0].max() - corr[corr!=0].min()
        if corr_mean >= 0.95 and corr_std <= 0.01 and corr_range <= 0.025:
            outlier = True

    return outlier


def remove_outliers(imgs_unique):
    
    # Detect images with a spike in the RGB histogram plots
    outliers = []
    for img in tqdm(imgs_unique):
        outliers.append(detect_outliers(img))
    outliers = np.array(outliers)
    
    # Drop outlier images from the list of unique images
    imgs_clean = np.array(imgs_unique)[~outliers].tolist()
    imgs_outliers = np.array(imgs_unique)[outliers].tolist()
    
    # Report status report
    print('Total number of images in the dataset: {:4}'.format(len(imgs_unique)))
    print('Number of outliers in the dataset: {:8}'.format(len(imgs_outliers)))
    
    return imgs_clean, imgs_outliers


def load_dataset(target_size=(64, 64), n_iter=2):

    transform_args={'rescale': 1/255,
                    'horizontal_flip': True,
                    'rotation_range': 22.5,
                    'width_shift_range': 0.05,
                    'height_shift_range': 0.05,
                    'brightness_range': (0.9, 1.1),
                    'zoom_range': (0.95, 1.05),
                    'fill_mode': 'reflect',
                    }
    
    # Create generator object to collect images
    generator = ImageDataGenerator(**transform_args).flow_from_directory(
        'data', target_size=target_size, batch_size=32, shuffle=True, seed=0)
    
    # Collect data from folders
    X, y, indeces = [], [], []
    n_elements = len(generator)
    for batch_i in tqdm(range(len(generator) * n_iter)):
        imgs, labels = generator.next()
        X.extend(imgs)
        y.extend(labels.argmax(axis=1))
        if batch_i % n_elements == 0:
            indeces.extend(generator.index_array)
    if n_iter >1:
        print('\nDataset was augmented to a total of N=%d images through means of:' % len(y))
        print('Image rotation, flipping, shifting, zooming and brightness variation.\n')

    # Shuffle images
    X = np.array(X)
    y = np.array(y)
    indeces = np.array(indeces)
    generator.indeces = indeces
    filenames = np.array(generator.filepaths)[indeces]
    generator.file_paths = filenames

    return X, y, generator


def create_dataset(imgs_clean, class_labels, img_dim=32, n_iter=2):

    # Name of parent folder
    parent_folder = 'data'

    # Overwrite parten folder if it already exists
    if os.path.exists(parent_folder):
        shutil.rmtree(parent_folder)
    os.makedirs(parent_folder)

    # Create subfolder per class and move images to corresponding location
    for label in class_labels:
        label_name = label.replace(' ', '_')
        os.makedirs(os.path.join(parent_folder, label_name))

        # Extract image label from relative path
        labels = np.array([os.path.dirname(temp)[17:] for temp in imgs_clean])
        imgs_class = shuffle(np.array(imgs_clean)[labels==label_name])

        for i, img in enumerate(imgs_class):
            file_ext = os.path.splitext(img)[1]
            file_name = '%03d%s' % (i + 1, file_ext)
            shutil.copy(img, os.path.join('data', label_name, 'img_' + file_name))

    # Resample images to desired resolution and load them to memory
    img_res = [img_dim] * 2
    X, y, generator = load_dataset(target_size=img_res, n_iter=n_iter)
    print('Images resampled to a resolution of %d x %d!' % tuple(img_res))

    # Extract meta information
    metainfo = {}
    metainfo['n_classes'] = generator.num_classes
    metainfo['categories'] = generator.class_indices
    metainfo['class_names'] = list(metainfo['categories'])
    metainfo['img_dim'] = img_dim
    metainfo['filenames'] = np.array(generator.file_paths)
    metainfo['indeces'] = np.array(generator.indeces)

    return np.array(X), np.array(y), metainfo


def plot_class_distribution(y, metainfo):

    # Create dataframe
    df = pd.value_counts(y).sort_index()
    
    df = df/df.sum() * 100
    df.index = metainfo['class_names']

    # Plot distribution
    plt.figure(figsize=(8, 4))
    df.plot.bar(title='Number of images per class')
    plt.xlabel('Class labels')
    plt.ylabel('Images per class [%]')


def plot_class_average(X, y, metainfo):

    # Number of classes
    n_classes = metainfo['n_classes']
    categories = metainfo['categories']

    # Plot first few images
    plt.figure(figsize=(n_classes * 2.25, 4))

    for i, c in enumerate(categories):

        # Display images
        ax = plt.subplot(1, n_classes, i + 1)
        img = np.median(X[y==i], axis=0)
        img /= img.max()
        plt.imshow(img)
        plt.title(c)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def plot_class_RGB(X, y, metainfo):

    # Number of classes
    n_classes = metainfo['n_classes']
    categories = metainfo['categories']

    # Plot first few images
    plt.figure(figsize=(n_classes * 2.25, 1))

    for i, c in enumerate(categories):

        # Display images
        ax = plt.subplot(1, n_classes, i + 1)
        img_array = X[y==i].reshape(-1, 3)
        for idx, color in enumerate(['red', 'green', 'blue']):
            ax.hist(img_array[:, idx], bins=64, range=(0, 1), color=color, alpha=0.5)
        plt.title(c)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def extract_RGB_features(y, metainfo, nbins=256):

    # Get list of file names
    filenames = metainfo['filenames']

    # Create place holder variable to fill up
    X_rgb = []

    # Iterate through each image and extract RGB color profile
    for img in tqdm(filenames):
        pixels = imageio.imread(img) / 255
        rgb_profile = np.ravel([np.histogram(pixels[:, 0], bins=nbins, range=(0, 1), density=True)[0]/nbins,
                                np.histogram(pixels[:, 1], bins=nbins, range=(0, 1), density=True)[0]/nbins,
                                np.histogram(pixels[:, 2], bins=nbins, range=(0, 1), density=True)[0]/nbins])
        X_rgb.append(rgb_profile)

    return np.array(X_rgb), y


def extract_neural_network_features(n_iter=2):

    print('Building model.')
    # Extract features using Mobilenet
    module_url = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4'

    # Load module
    m = tf.keras.Sequential([hub.KerasLayer(module_url, trainable=False)])

    # Build module
    m.build([None, 224, 224, 3])

    # Create generator object to collect images

    transform_args={'rescale': 1/255,
                    'horizontal_flip': True,
                    'rotation_range': 22.5,
                    'width_shift_range': 0.05,
                    'height_shift_range': 0.05,
                    'brightness_range': (0.9, 1.1),
                    'zoom_range': (0.95, 1.05),
                    'fill_mode': 'reflect',
                    }

    # Create generator object to collect images
    generator = ImageDataGenerator(**transform_args).flow_from_directory(
        'data', target_size=(224, 224), batch_size=32, shuffle=True, seed=0)

    # Extract features in batches
    print('Extracting features.')

    # Collect data from folders
    X_temp, y_temp, indeces_temp = [], [], []
    n_elements = len(generator)
    for batch_i in tqdm(range(len(generator) * n_iter)):
        imgs, labels = generator.next()
        X_temp.extend(m.predict(imgs))
        y_temp.extend(labels.argmax(axis=1))
        if batch_i % n_elements == 0:
            indeces_temp.extend(generator.index_array)

    if n_iter >1:
        print('Dataset was augmented to a total of N=%d images through means of:' % len(y_temp))
        print('Image rotation, flipping, shifting, zooming and brightness variation.\n')

    # Extract features
    X_nn = np.array(X_temp)
    y_nn = np.array(y_temp)
    indeces_nn = np.array(indeces_temp)
    filenames = np.array(generator.filepaths)[indeces_nn]

    return X_nn, y_nn


def plot_recap(X, X_rgb, X_nn):
    
    # Choose a random picture
    idx = np.random.choice(range(len(X_nn)))
    
    # Specify figure size
    fig = plt.figure(figsize=(15, 4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[3, 4, 6])

    ax = plt.subplot(gs[0, 0])
    ax.imshow(X[idx])
    plt.title('Pixel Format')
    plt.xlabel('Shape: %s' % str(X[idx].shape))
    plt.xticks([])
    plt.yticks([])

    ax = plt.subplot(gs[0, 1])
    rgb_temp = np.array(np.split(X_rgb[idx], 3))
    for i, c in enumerate(['r', 'g', 'b']):
        ax.plot(rgb_temp[i], color=c)
    plt.title('RGB Format')
    plt.xlabel('Shape: %s' % str(rgb_temp.shape[::-1]))
    plt.xticks([])
    plt.yticks([])

    ax = plt.subplot(gs[0, 2])
    ax.step(range(1, 1281), X_nn[idx],linewidth=0.5)
    plt.title('MobileNet Format')
    plt.xlabel('Shape: %s' % str(X_nn[idx].shape))
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()


def model_fit(X, y, test_size=0.5, alpha_low=-6, alpha_high=6,
              n_steps=25, cv=4, plot_figures=False):

    # Prepare datasets
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_temp = X.reshape((len(X), -1))
    X_temp = scaler.fit_transform(X_temp)
    indexes = list(range(len(X_temp)))

    # Split Dataset into training and test set
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_temp, y, indexes, test_size=test_size, random_state=0, stratify=y)

    # Model creation
    ridge = RidgeClassifier(class_weight='balanced')
    alphas = np.logspace(alpha_low, alpha_high, num=n_steps)
    clf = GridSearchCV(estimator=ridge,
                       param_grid={'alpha': alphas},
                       cv=cv, return_train_score=True,
                       n_jobs=-1, verbose=1)

    # Fit the model to the data
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        start = time.time()
        results = clf.fit(x_train, y_train)
        comp_time_total = time.time() - start

    # Plot the model fit curves
    if plot_figures:

        # Extract relevant modelling metrics
        train_scores = 100 * clf.cv_results_['mean_train_score']
        valid_scores = 100 * clf.cv_results_['mean_test_score']
        std_tr = 100 * clf.cv_results_['std_train_score']
        std_va = 100 * clf.cv_results_['std_test_score']

        plt.figure(figsize=(10, 5))
        plt.semilogx(alphas, train_scores, label='Training Set')
        plt.semilogx(alphas, valid_scores, label='Validation Set')

        # Add marker and text for best score
        x_pos = clf.best_params_['alpha']
        y_pos = 100 * clf.best_score_
        txt = '{:0.2f}%'.format(y_pos)
        plt.scatter(x_pos, y_pos, marker='x', c='red', zorder=10)
        plt.text(x_pos, y_pos - 7.5, txt, fontdict={'size': 18})

        # Quantify variance with ±std curves
        plt.fill_between(alphas, train_scores-std_tr, train_scores+std_tr, alpha=0.3)
        plt.fill_between(alphas, valid_scores-std_va, valid_scores+std_va, alpha=0.3)
        plt.title('Model Performance')
        plt.ylabel('Classification Accuracy [%]')
        plt.xlabel('Model Parameter [alpha]')
        
        # Adjust x-lim, y-lim, add legend and adjust layout
        plt.xlim(10**alpha_low, 10**alpha_high)
        plt.ylim(15, 105)
        plt.legend()
        plt.tight_layout()
        plt.show()

    else:
        # Provide written performance feedback
        best_score_test = clf.best_score_ * 100
        feedback_txt = 'Model trained for {:.2f}s total '.format(comp_time_total)
        feedback_txt += 'and reached an accuracy of: {:.2f}%'.format(best_score_test)
        time.sleep(0.25)
        print(feedback_txt)

    # Store everything in model
    model = {'model': results.best_estimator_,
             'best_score': results.best_score_,
             'x_train': x_train,
             'x_test': x_test,
             'y_train': y_train,
             'y_test': y_test,
             'idx_train': idx_train,
             'idx_test': idx_test}

    return model


def check_model_performance(model, metainfo):

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot confusion matrix
    ax.set_title('Confusion Matrix')
    plt.rc('font', size=14)
    plot_confusion_matrix(model['model'],
                          model['x_test'],
                          model['y_test'],
                          display_labels=metainfo['class_names'],
                          cmap='Blues', normalize=None,
                          values_format='d',
                          xticks_rotation='vertical', ax=ax)

    # Remove colorbar
    im = ax.images
    cb = im[-1].colorbar
    cb.remove()

    # Compute chance level
    chance_level = pd.Series(model['y_test']).value_counts(normalize=True).max()
    score_valid = model['best_score']
    score_test = model['model'].score(model['x_test'], model['y_test'])

    # Create report text
    report = 'Chance level on test set: {:8.02f}%\n'.format(chance_level * 100)
    report += 'Accuracy on validation set: {:6.02f}%\n'.format(score_valid * 100)
    report += 'Accuracy on test set: {:12.02f}%'.format(score_test * 100)

    # Plot title for report
    #ax.text(-18, -1.1, 'Classification Report', fontsize=19, verticalalignment='top')

    # Plot report in text box
    props = dict(boxstyle='round', facecolor='blue', alpha=0.1)
    ax.text(-2.16, 0.95, report, transform=ax.transAxes, fontsize=18,
            verticalalignment='top', bbox=props, family='monospace')


def get_predictions(model):

    # Establish y_pred and y_true
    y_pred = model['model'].predict(model['x_test'])
    y_true = model['y_test']

    # Collect idx of correct and wrong classifications
    id_correct = np.argwhere(y_pred==y_true).ravel()
    id_wrong = np.argwhere(y_pred!=y_true).ravel()

    # Compute probability
    sigm = model['model'].decision_function(model['x_test'])
    if np.sign(sigm).sum()>0:
        prob = 1 / np.log(sigm)
        prob = prob / np.sum(prob, axis=1)[..., None]
    else:
        prob = np.exp(sigm) / np.sum(np.exp(sigm), axis=1)[..., None]
        
    return prob, id_correct, id_wrong


def investigate_predictions(model, metainfo, show_correct=True, nimg=8):

    # Compute class probabilities
    prob, id_correct, id_wrong = get_predictions(model)

    if show_correct:
        img_ids = id_correct
    else:
        img_ids = id_wrong
    img_ids = shuffle(img_ids)

    # Establish filename list
    filenames = metainfo['filenames']
    filenames_test = filenames[np.array(model['idx_test'])]
    
    # Plot first N image prediction information
    fig = plt.figure(figsize=(nimg*3, 4.5))
    gs = gridspec.GridSpec(2, nimg, height_ratios=[1, 4]) 

    for i_pos, idx in enumerate(img_ids[:nimg]):

        # Get image
        img = imageio.imread(filenames_test[idx])

        # Get probability
        probability = prob[idx]

        # Get predicted and true label of image
        predicted_label = np.argmax(probability)
        true_label = model['y_test'][idx]

        # Get class names
        class_names = metainfo['class_names']

        # Identify the text color
        if predicted_label == true_label:
            color = '#004CFF'
            info_txt = "{} {:2.0f}%\nCorrect!".format(
            class_names[predicted_label], 100*np.max(probability))
        else:
            color = '#F50000'
            info_txt = "{} {:2.0f}%\nTrue: {}".format(
            class_names[predicted_label], 100*np.max(probability),
            class_names[true_label])

        # Plot prediction probabilities
        ax = plt.subplot(gs[0, i_pos])
        pred_plot = plt.bar(range(len(probability)), probability, color='#BFBFBF')
        pred_plot[predicted_label].set_color('#FF4747')
        pred_plot[true_label].set_color('#477EFF')
        xlim = list(plt.xlim())
        plt.hlines(1/len(probability), *xlim, linestyles=':', linewidth=2)
        plt.xlim(xlim)
        plt.xticks([])
        plt.yticks([])
        plt.title('Class Probability')

        # Plot image
        ax = plt.subplot(gs[1, i_pos])
        plt.imshow(img)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        # Add information text to image
        plt.xlabel(info_txt, color=color)

    plt.tight_layout()
    plt.show()


def predict_new_image(img_url, model_nn, metainfo):

    # Download the image
    print('Downloading image.')
    response = requests.get(img_url, stream=True)
    with open('temp_img.jpg', 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)

    # Load image and convert to RGB color scheme
    try:
        im = Image.open('temp_img.jpg').convert(mode='RGB', colors=256)
    except IOError:
        print('Image cannot be loaded.')

    # Extract size of image
    width, height = im.size

    # Extract min dimension for squaring of image
    min_dim = min(width, height)

    # Compute image offset for squaring of image
    offset_x = width - min_dim
    offset_y = height - min_dim

    # Square image
    im_squared = im.crop((offset_x/2., offset_y/2.,
                        offset_x/2. + min_dim, offset_y/2. + min_dim))

    # Resize image 
    im_resized = im_squared.resize((224, 224), resample=1)

    # Put image into numpy array
    img = np.array(im_resized) / 255

    # Extract features using Mobilenet
    print('Feature Extraction.')
    module_url = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4'
    m = tf.keras.Sequential([hub.KerasLayer(module_url, trainable=False)])
    m.build([None, 224, 224, 3])
    nn_features = m.predict(img[None, ...])

    # Compute class predictino probabilities
    print('Plotting report.')
    sigm = model_nn['model'].decision_function(nn_features).squeeze()
    probability = np.exp(sigm) / np.sum(np.exp(sigm)) * 100

    # Get class names
    class_names = metainfo['class_names']

    # Get predicted and true label of image
    predicted_idx = np.argmax(probability)
    predicted_prob = probability[predicted_idx]
    predicted_label = class_names[predicted_idx]

    # Plot overview figure
    fig = plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    # Plot prediction probabilities
    ax = plt.subplot(gs[0])
    plt.title('Prediction Probability')
    y_pos = np.arange(len(probability))
    plt.barh(y_pos, probability, color='#BFBFBF')

    # Set y-label text
    y_label_text = ['{}: {:5.1f}%'.format(e, probability[i]) for i, e in enumerate(class_names)]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_label_text)
    ylim = list(plt.ylim())
    plt.vlines(1/len(probability) * 100, *ylim, linestyles=':', linewidth=2)
    plt.ylim(ylim)

    # Plot image
    ax = plt.subplot(gs[1])
    plt.title('Image')
    plt.imshow(img)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    # Add information text to image
    info_txt = '\nThis is to {:.02f}% a {}!'.format(predicted_prob, predicted_label)
    plt.xlabel(info_txt, fontdict={'size': 21})

    plt.tight_layout()
    plt.show()