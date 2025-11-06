import numpy as np
from pathlib import Path
import librosa
import argparse

from scipy import stats
from tqdm import tqdm

from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

SUPPORTED_CLASSIFIERS = ['sgd', 'knn', 'svm']
MODEL_TYPE = 'sgd'
DATASET_PATH = './NSynth/'
FEATURES_PATH = './features/'

SKIP_FACTOR = 10


MIN_NOTE = 0
MAX_NOTE = 127

def train_model(train_X, train_Y, valid_X, valid_Y, clf_type = 'knn', **kwargs):
    '''
    args:
        train_X, train_Y: training inputs and labels
        valid_X, valid_Y: validation inputs and labels
    '''
    
    if clf_type not in SUPPORTED_CLASSIFIERS:
        raise ValueError('Unsupported classifier type: ' + clf_type)
    
    # choose classifier
    if clf_type == 'sgd':
        clf = SGDClassifier(verbose=0, loss='hinge', alpha=kwargs.get('alpha', 0.01), max_iter=kwargs.get('max_iter', 1000), penalty='l2', random_state=0)
    elif clf_type == 'knn':
        clf = KNeighborsClassifier(n_neighbors=kwargs.get('n_neighbors', 1))
    elif clf_type == 'svm':
        clf = svm.SVC(gamma='scale', kernel='linear')
    
    # train
    clf.fit(train_X, train_Y)
    
    valid_Y_hat = clf.predict(valid_X)
    
    accuracy = np.sum((valid_Y_hat == valid_Y)) / len(valid_Y_hat) * 100.0
    
    print('validation accuracy = ' + str(accuracy) + '%')
    
    return clf, accuracy

def load_data(fold = 'train'):
    data_list = Path(DATASET_PATH) / (fold + '_list.txt')
    
    with open(data_list, 'r') as f:
        audio_files = [ Path(DATASET_PATH) / line.rstrip('\n') for line in f ]
    
    return audio_files

def extract_spectrogram(audio):
    stft = librosa.stft(audio)
    S, phase = librosa.magphase(stft)
    return S

def extract_F0(y, sr):
    f0, voiced_flag, voiced_prob  = librosa.pyin(y, fmin=max(22.0, librosa.midi_to_hz(MIN_NOTE)), fmax=min(sr / 2.0, librosa.midi_to_hz(MAX_NOTE)))

    f0_filtered = np.fromiter((x for x in f0 if x > 0), dtype=f0.dtype)
    mode_result = stats.mode(f0_filtered, keepdims=True)
    
    return mode_result.mode[0]

def extract_features(file_list):
    """Extract features from audio files"""
    f0_list = []
    
    for f in tqdm(file_list, desc="Extracting features"):
        y, sr = librosa.load(f, sr=None)
        
        f0 = extract_F0(y, sr)
        
        f0_list.append(f0)
    
    features_dict = {
        'f0': np.array(f0_list),
    }
    
    for key, val in features_dict.items():
        print(f"{key}: {val.shape}")
    
    return features_dict

def save_features(train_features, valid_features):
    Path(FEATURES_PATH).mkdir(exist_ok=True)
    np.savez(Path(FEATURES_PATH) / 'train_features.npz', **train_features)
    np.savez(Path(FEATURES_PATH) / 'valid_features.npz', **valid_features)
    print(f"Features saved to {FEATURES_PATH}")
    print(f"Available features: {list(train_features.keys())}")

def load_features():
    train_data = np.load(Path(FEATURES_PATH) / 'train_features.npz')
    valid_data = np.load(Path(FEATURES_PATH) / 'valid_features.npz')
    
    # TODO: modify what features you include here
    feature_keys = ['f0']
    
    train_features = [train_data[key] for key in feature_keys]
    valid_features = [valid_data[key] for key in feature_keys]
    
    # Reshape 1D arrays to column vectors before concatenating
    train_features = [f.reshape(-1, 1) if f.ndim == 1 else f for f in train_features]
    valid_features = [f.reshape(-1, 1) if f.ndim == 1 else f for f in valid_features]
    
    train_X = np.concatenate(train_features, axis=1)
    valid_X = np.concatenate(valid_features, axis=1)
    
    # Transpose to match reference code format: (n_features, n_samples)
    train_X = train_X.T
    valid_X = valid_X.T
    
    print(f"train_X shape: {train_X.shape}, valid_X shape: {valid_X.shape}")
    
    return train_X, valid_X

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='extract', choices=['extract', 'train'])
    args = parser.parse_args()
    
    if args.mode == 'extract':
        # Extract and save features
        train_X_list, valid_X_list = load_data('train'), load_data('valid')
        
        # for fast testing, use a smaller subset
        train_X_list = train_X_list[::SKIP_FACTOR]
        valid_X_list = valid_X_list[::SKIP_FACTOR]
        
        # get features
        train_features, valid_features = extract_features(train_X_list), extract_features(valid_X_list)
        
        # save features
        save_features(train_features, valid_features)
        
    elif args.mode == 'train':
        # Load selected features
        train_X, valid_X = load_features()

        # generate labels
        cls = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        train_Y = np.repeat(cls, 110 // SKIP_FACTOR)
        valid_Y = np.repeat(cls, 30 // SKIP_FACTOR)
        
        # normalise features
        train_X = train_X.T
        train_X_mean = np.mean(train_X, axis=0)
        train_X = train_X - train_X_mean
        train_X_std = np.std(train_X, axis=0)
        train_X = train_X / (train_X_std + 1e-5)
        
        valid_X = valid_X.T
        valid_X = valid_X - train_X_mean
        valid_X = valid_X / (train_X_std + 1e-5)
        
        # train model
        alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
        
        models = []
        valid_acc = []
        for alpha in alphas:
            clf, acc = train_model(train_X, train_Y, valid_X, valid_Y, clf_type=MODEL_TYPE, alpha=alpha)
            models.append(clf)
            valid_acc.append(acc)
        
        # best model
        best_model = models[np.argmax(valid_acc)]
        
        # evaluate model with the test set
        valid_Y_hat = best_model.predict(valid_X)
        
        accuracy = np.sum((valid_Y_hat == valid_Y)) / len(valid_Y_hat) * 100.0
        print('Best validation accuracy = ' + str(accuracy) + '%')
