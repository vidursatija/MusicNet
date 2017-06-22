import sys
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QLabel, QSlider, QPushButton
from PyQt5.QtGui import QPainter, QColor, QPen, QIcon, QPixmap
import os
from iTunesConnector import iTunes
import pickle
from predict import PredictModel
import subprocess


class StateBtn(QWidget):
	def __init__(self, parent=None):
		super().__init__(parent=parent)
		self.playing = False
		self.drawUI()

	def drawUI(self):
		self.play = QPushButton(self)
		self.play.setIconSize(QSize(40, 40))
		self.play.setIcon(QIcon(QPixmap("play.png").scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
		self.play.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
		self.play.clicked.connect(lambda:self.playpause(self.play))
		self.resize(50, 50)

	def playpause(self, btn):
		if self.playing:
			os.system("""osascript -e 'tell application "iTunes" to pause'""")
			self.playing = False
			btn.setIcon(QIcon(QPixmap("play.png").scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
		else:
			os.system("""osascript -e 'tell application "iTunes" to play'""")
			self.playing = True
			btn.setIcon(QIcon(QPixmap("pause.png").scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)))

class ForwardBtn(QWidget):
	def __init__(self, parent=None):
		super().__init__(parent=parent)
		self.iTunes = iTunes()
		self.drawUI()

	def drawUI(self):
		self.play = QPushButton(self)
		self.play.setIconSize(QSize(40, 20))
		self.play.setIcon(QIcon(QPixmap("fastforward.png").scaled(40, 20, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
		self.play.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
		self.play.clicked.connect(lambda:self.forward())
		self.resize(80, 40)

	def forward(self):
		cTrackDuration = self.iTunes.current_track.duration
		self.iTunes.player_position = cTrackDuration - 2
		#os.system("""osascript -e 'tell application "iTunes" to next track'""") #change this to recommend

class RewindBtn(QWidget):
	def __init__(self, parent=None):
		super().__init__(parent=parent)

		self.drawUI()

	def drawUI(self):
		self.play = QPushButton(self)
		self.play.setIconSize(QSize(40, 20))
		self.play.setIcon(QIcon(QPixmap("rewind.png").scaled(40, 20, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
		self.play.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
		self.play.clicked.connect(lambda:self.rewind())
		self.resize(80, 40)

	def rewind(self):
		os.system("""osascript -e 'tell application "iTunes" to back track'""")

class SongWidget(QWidget):

	def __init__(self, parent=None):
		super().__init__(parent=parent)

	def drawUI(self, song_name="blank", song_artist="blank"):
		self.art = QLabel(self)
		self.art.resize(50, 50)
		#self.art.setPixmap(QPixmap("".join([song_name,'.jpg'])).scaled(50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation))

		self.song_label = QLabel(self)
		self.song_label.setText(song_name)
		self.song_label.setStyleSheet("border-image: rgba(0, 0, 0, 0); color: rgba(255, 255, 255); font-size: 12pt; font-weight: 600;")
		self.song_label.move(60, 10)
		self.song_label.resize(180, 15)

		self.artist_label = QLabel(self)
		self.artist_label.setText(song_artist)
		self.artist_label.setStyleSheet("border-image: rgba(0, 0, 0, 0); color: rgba(255, 255, 255); font-size: 12pt; font-weight: 200;")
		self.artist_label.move(60, 25)
		self.artist_label.resize(180, 15)

	def changeLabels(self, song_name=None, song_artist=None, song_loc='songs_raw/no_art.png'):
		if song_name != None:
			if len(song_name) > 30:
				song_name = "".join([song_name[:28], '..'])
			self.song_label.setText(song_name)

		if song_artist != None:
			if len(song_artist) > 30:
				song_artist = "".join([song_artist[:28], '..'])
			self.artist_label.setText(song_artist)

		if song_loc != None:
			self.art.setPixmap(QPixmap(song_loc).scaled(50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation))

class MainWindow(QWidget):
	def __init__(self, parent=None):
		super(MainWindow, self).__init__(parent)

		self.setGeometry(100, 100, 320, 480)
		self.setFixedSize(320, 480)
		self.setWindowTitle("ShuffleNet")
		#self.setWindowFlags(Qt.FramelessWindowHint)
		self.setAttribute(Qt.WA_TranslucentBackground, True)

		self.iTunes = iTunes()
		self.r_song = []

		self.currentSongID = '-1'
		self.songAdded = False

		pickle_file = "itl.p"
		self.itl = pickle.load(open(pickle_file, "rb"))

		self.pm = PredictModel()
		self.song_queue = []
		self.song_changing = False

		input_file = open(os.path.join("", 'input.p'), 'rb')
		input_file_data = pickle.load(input_file)
		self.int_to_key = input_file_data['itk']
		self.key_to_int = input_file_data['kti']

		self.best_recommend = None

		self.drawUI()

	def drawUI(self):

		self.alpha_rect = QLabel(self)
		self.alpha_rect.setPixmap(QPixmap("black_alpha.png").scaled(320, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation))

		self.current_song = SongWidget(self)
		self.current_song.drawUI()
		self.current_song.move(40, 40)

		self.line1 = QLabel(self)
		self.line1.setPixmap(QPixmap('line.png').scaled(300, 1))
		self.line1.move(10, 120)

		self.time_line = QSlider(Qt.Horizontal, self)
		self.time_line.setGeometry(40, 135, 240, 4)
		self.time_line.setMaximum(157)
		self.time_line.setMinimum(0)
		self.time_line.setValue(0)
		self.time_line.setStyleSheet("""QSlider::groove:horizontal 
			{background-color: rgba(128, 128, 128, 0.5); border: 1px solid rgba(128, 128, 128, 0.5); border-radius: 2px; height:4px;} 
			QSlider::handle:horizontal 
			{border: 1px solid #fff;width: 4px;height: 4px;border-radius: 2px; background-color: #fff;}
			QSlider::sub-page:horizontal
			{background-color: #fff;}
			""")
		self.time_line.sliderReleased.connect(self.changeTime)

		self.rewind = RewindBtn(self)
		self.rewind.move(50, 175)

		self.play = StateBtn(self)
		self.play.move(135, 165)

		self.fastforward = ForwardBtn(self)
		self.fastforward.move(220, 175)

		self.line2 = QLabel(self)
		self.line2.setPixmap(QPixmap('line.png').scaled(300, 1))
		self.line2.move(10, 240)

		self.r_song.append(SongWidget(self))
		self.r_song[0].drawUI()
		self.r_song[0].move(40, 270)

		self.r_song.append(SongWidget(self))
		self.r_song[1].drawUI()
		self.r_song[1].move(40, 330)

		self.r_song.append(SongWidget(self))
		self.r_song[2].drawUI()#, song_loc="/Users/vidursatija/Music/iTunes/iTunes Media/Music/Halsey/hopeless fountain kingdom (Deluxe)/01 The Prologue.mp3")
		self.r_song[2].move(40, 390)

		self.timer = QTimer(self)
		self.timer.timeout.connect(self.updateLabels)
		self.timer.start(1000)

	def changeTime(self):
		self.iTunes.player_position = self.time_line.value()

	def updateLabels(self):
		s = self.iTunes.current_track
		did_name = str(s.name)
		did_album = str(s.album)
		did_artist = str(s.artist)
		if did_name == "":
			did_name = "None"
		if did_album == "":
			did_album = "None"
		if did_artist == "":
			did_artist = "None"
		DID = "-".join([did_name[:5], did_album[:5], did_artist[:5], did_name[-5:], did_album[-5:]])#self.iTunes.current_track.database_id
		#print(DID)
		cTrackDuration = self.iTunes.current_track.duration
		playerPosition = self.iTunes.player_position
		self.time_line.setValue(playerPosition)

		if self.iTunes.playing != self.play.playing:
			self.play.playing = self.iTunes.playing
			if self.iTunes.playing:
				self.play.play.setIcon(QIcon(QPixmap("pause.png").scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
			else:
				self.play.play.setIcon(QIcon(QPixmap("play.png").scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)))


		if DID == self.currentSongID:
			if playerPosition >= 0.75 * cTrackDuration + 1:
				if self.songAdded == False:
					goodSong = True
					try:
						_ = self.key_to_int[DID]
					except:
						goodSong = False
					if goodSong:
						self.song_queue.append(DID)
						self.songAdded = True

						if len(self.song_queue) > 10:
							pickle.dump({"q": self.song_queue[-11:]}, open("update_queue.p", 'wb'))
							if os.path.isfile("temp.train") == False:
								print("Training start")
								subprocess.run(['python3', 'update_model.py'])
			else:
				self.songAdded = False

			if playerPosition <= cTrackDuration and playerPosition >= cTrackDuration-2:
				if self.song_changing == False:
					self.iTunes.pause()
					if self.best_recommend == None:
						self.iTunes.play()
					else:
						self.iTunes.play(self.best_recommend)
					self.song_changing = True
				return

			if self.song_changing == True:
				self.song_changing = False

		else:
			self.time_line.setMaximum(cTrackDuration)
			try:
				art = s.artworks[0].raw_data
				with open("".join(['songs_raw/', self.iTunes.current_track.name, '.raw']), 'wb') as f:
					f.write(art)
				self.current_song.changeLabels(song_name=s.name, song_artist=s.artist, song_loc="".join(['songs_raw/', s.name, '.raw']))
			except:
				self.current_song.changeLabels(song_name=s.name, song_artist=s.artist)
			
			self.currentSongID = DID
			self.top3 = ['-1', '-1', '-1']
			if DID == '':
				self.top3 = self.pm.predictNext(self.song_queue+['-1'])
			else:
				goodSong = True
				try:
					_ = self.key_to_int[DID]
				except Exception as e:
					print(e)
					print("Not a known song")
					#print(DID)
					goodSong = False
				if goodSong:
					self.top3 = self.pm.predictNext(self.song_queue+[DID, DID])
				else:
					return
			#print(self.top3)
			#SEARCH NAME AND ALBUM AND ARTIST in itl
			self.best_recommend = None
			for index, top in enumerate(self.top3):
				for id, song in self.itl.songs.items():

					s_name = str(song.name)
					s_artist = str(song.artist)
					s_album = str(song.album)
					if top == "-".join([s_name[:5], s_album[:5], s_artist[:5], s_name[-5:], s_album[-5:]]):
						if s_name == 'None':
							s_name = ""
						if s_artist == 'None':
							s_artist = ""
						if s_album == 'None':
							s_album = ""
						search_list = self.iTunes.search(name=s_name, album=s_album, artist=s_artist)
						if len(search_list) == 0:
							break
						best_track = search_list[0]
						if index == 0:
							self.best_recommend = best_track
						try:
							art = best_track.artworks[0].raw_data
							with open("".join(['songs_raw/', best_track.name, '.raw']), 'wb') as f:
								f.write(art)
							self.r_song[index].changeLabels(song_name=best_track.name, song_artist=best_track.artist, song_loc="".join(['songs_raw/', best_track.name, '.raw']))
						except:
							self.r_song[index].changeLabels(song_name=best_track.name, song_artist=best_track.artist)



if __name__ == '__main__':
	app = QApplication(sys.argv)
	window = MainWindow()
	window.show()
	sys.exit(app.exec_())


