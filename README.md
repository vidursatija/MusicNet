# MusicNet (Python3)
#### Improved shuffle for iTunes
This app recognizes your pattern of listening music using recurrent neural networks and generates better shuffle predictions.
You have to run the app while iTunes is on.

## Install Dependencies
```
pip3 install -r requirements.txt
pip3 install git+https://github.com/PhilipTrauner/iTunesConnector.git
pip3 install git+https://github.com/liamks/libpytunes.git
```

## Setup
Open music_analyze.py and update the path of your iTunes XML directory at line 67
```
xmlPath = '/Users/<username>/Music/iTunes/iTunes Music Library.xml'
```
Install by:
```
python3 music_analyze.py
python3 input.py
python3 model.py
```
This might take some time to analyze and setup. (6 hours ~ 600 songs)

For best performance ensure your iTunes library has sorted albums

## Usage
```
python3 window.py
```
