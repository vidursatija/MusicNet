# MusicNet (Python3)
Improved shuffle for iTunes

## Setup
Open music_analyze.py and write the path of your iTunes XML directory
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

## Requirements
1. tensorflow (requirements.txt)
2. numpy (requirements.txt)
3. iTunesConnector (https://github.com/PhilipTrauner/iTunesConnector)
4. PyQt5 (requirements.txt)
5. Librosa (requirements.txt)
6. libpytunes (https://github.com/liamks/libpytunes)
