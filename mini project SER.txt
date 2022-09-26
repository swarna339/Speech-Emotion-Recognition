import soundfile
import numpy as np 
import librosa  
import glob 
import os 
from sklearn.model_selection import train_test_split  
from sklearn.neural_network import MLPClassifier  
from sklearn.metrics import accuracy_score 
def get_feature(file_name,mfccs,mel,chroma,contrast):
    
        data, sample_rate = librosa.load(file_name)
        stft = np.abs(librosa.stft(data))
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(data, sr=sample_rate).T,axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        
        
    
        return mfccs,mel,chroma,contrast
# emotions in dataset
list_emotion = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

classify_emotions = {
    "sad",
    "happy",
    "surprised"
    
    
}
def load_data(test_size=0.2):
    feature, y = [], []
    for file in glob.glob("C:\\Users\\Documents\\ravdess data\\Actor_*\\*.wav"):
        basename = os.path.basename(file)  
       
        emotion = list_emotion[basename.split("-")[2]]   
       
        if emotion not in classify_emotions:    
            try:
                mfccs,mel,chroma,contrast = get_feature(file)
            except Exception as e:
                print ("Error encountered while parsing file: ", file)
                continue
            ext_features = np.hstack([mfccs,mel,chroma,contrast])
            feature.append(ext_features)
            y.append(emotion)
        
    return train_test_split(np.array(feature), y, test_size=test_size, random_state=9)
feature_train, feature_test, y_train, y_test = load_data(test_size=0.25)
print("Number of samples in training data:", feature_train.shape[0])

print("Number of samples in testing data:", feature_test.shape[0])
print("Training the model.....")
clf=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500).fit(feature_train, y_train)

y_pred = clf.predict(feature_test)


accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Number of features:", feature_train.shape[1])
***********************************************IMPLEMENTATION********************************************************************************************************************8
#importing libraries
import soundfile as sf    
import librosa            
import os                 
import glob               
import pickle           
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import accuracy_score 
def feature_extraction(fileName,mfcc,chroma,mel):
    with sf.SoundFile(fileName) as file:
        sound = file.read(dtype='float32')
        sample_rate = file.samplerate     
        if chroma:         
            stft = np.abs(librosa.stft(sound))
        feature = np.array([])                
        if mfcc:
            mfcc = np.mean(librosa.feature.mfcc(y=sound,sr=sample_rate,n_mfcc=40).T,axis=0)
            feature =np.hstack((feature,mfcc))
        if chroma:
            chroma =  np.mean(librosa.feature.chroma_stft(S=stft,sr=sample_rate).T,axis=0)
            feature = np.hstack((feature,chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=sound,sr=sample_rate).T,axis=0)
            feature =np.hstack((feature,mel))
        return feature  

int_emotion = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}
EMOTIONS = {"happy","sad","neutral","angry"}
def train_test_data(test_size=0.3):
    features, emotions = [],[] #intializing features and emotions
    for file in glob.glob("data/Actor_*/*.wav"):
        fileName = os.path.basename(file)   
        emotion = int_emotion[fileName.split("-")[2]]
        if emotion not in EMOTIONS:
            continue
        feature=feature_extraction(file,mfcc=True,chroma=True,mel=True,) 
        features.append(feature)
        emotions.append(emotion)
    return train_test_split(np.array(features),emotions, test_size=test_size, random_state=7) 
X_train,X_test,y_train,y_test=train_test_data(test_size=0.3)
print("Total number of training sample: ",X_train.shape[0])
print("Total number of testing example: ",X_test.shape[0])
print("Feature extracted",X_train.shape[1])
#initializing the multi layer perceptron model
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(400,), learning_rate='adaptive', max_iter=1000)
#fitting the training data into model
print("__________Training the model__________")
model.fit(X_train,y_train)
#Predicting the output value for testing data
y_pred = model.predict(X_test)
print("Accuracy is: {:.2f}%".format(accuracy*100))
#calculating accuracy
accuracy = accuracy_score(y_true=y_test,y_pred=y_pred)
accuracy*=100
print("accuracy: {:.4f}%".format(accuracy))
#saving the model 
if not os.path.isdir("model"): 
   os.mkdir("model") 
pickle.dump(model, open("model/mlp_classifier.model", "wb"))