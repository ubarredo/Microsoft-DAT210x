import os
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
from sklearn.linear_model import LinearRegression

# How much of the audio file will be provided:
portion = 0.25
# Create regular python lists:
zero = list()
rate = list()
# Load up all 50 of the 0_jackson*.wav files:
wpath = 'datasets/fsdd/'
for f in os.listdir('datasets/fsdd'):
    wav = wavfile.read(wpath + f)
    if '0_jackson' in f:
        zero.append(wav[1])
        rate.append(wav[0])
# Convert zero into a DataFrame:
zero = pd.DataFrame(zero, dtype=np.int16)
# Do a dropna on the Y axis:
zero.dropna(axis=1, inplace=True)
# Convert back into an ndarray:
zero = zero.values
# Get the audio samples count:
n_audio_samples = zero.shape[1]
# Split into train & test data:
idx = np.random.RandomState(7).randint(zero.shape[0])
test = zero[idx].reshape(1, -1)
train = np.delete(zero, idx, axis=0)
X_test = test[:, :round(portion * n_audio_samples)]
Y_test = test[:, round(portion * n_audio_samples):]
X_train = train[:, :round(portion * n_audio_samples)]
Y_train = train[:, round(portion * n_audio_samples):]
# Check the shapes of train and test slices:
print(f"Train shape: {train.shape}")
print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")
print(f"Test shape: {test.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Y_test shape: {Y_test.shape}")
# Create the linear regression model:
lreg = LinearRegression()
# Fit your model using your training data:
lreg.fit(X_train, Y_train)
# Predict the 'label' of X_test:
pred = lreg.predict(X_test).astype(dtype=np.int16)
# Score how well your prediction would do:
score = lreg.score(X_test, Y_test)
print(f"Extrapolation R^2 Score: {score}")
# Save the original test clip:
wavfile.write(wpath + 'original.wav', rate[idx], test.ravel())
# Save the extrapolated test clip:
wavfile.write(wpath + 'extrapolated.wav', rate[idx],
              np.hstack((X_test, pred))[0])
