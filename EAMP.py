# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 12:30:39 2018

@author: David
"""

import sys
import cv2
from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox, QFileDialog
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap, QImage
import dlib
from cv2 import WINDOW_NORMAL
import pickle
import numpy as np
import math
import random
import os
import subprocess
import validators
import pymongo
import datetime
#from moviepy.editor import VideoFileClip
#import time



class ProjectUI(QDialog):
    def __init__(self):
        super(ProjectUI, self).__init__()
        loadUi('projectUI.ui', self)
        self.vc_trailer = None
        self.vc = None
        self.timer = None
        self.video_restart = None
        self.activation_mood = None
        self.deactivation_mood = None
        self.pleasantness_mood = None
        self.unpleasantness_mood = None
        self.query_results_for_random = []
        self.emotions = ["anger", "disgust", "happy", "neutral", "surprise", "sadness", "fear"]
        self.stars = ['1','2','3','4','5']
        self.emotionsComboBox.addItems(self.emotions)
        self.angerStarComboBox.addItems(self.stars)
        self.fearStarComboBox.addItems(self.stars)
        self.disgustStarComboBox.addItems(self.stars)
        self.happinessStarComboBox.addItems(self.stars)
        self.neutralityStarComboBox.addItems(self.stars)
        self.surpriseStarComboBox.addItems(self.stars)
        self.sadnessStarComboBox.addItems(self.stars)
        self.set_stars()
        self.startDetectionButton.clicked.connect(self.start_detection)
        self.endDetectionButton.clicked.connect(self.end_detection)
        self.restartVideoStreamingButton.clicked.connect(self.restart_video_streaming)
        self.addMusicToCatalogButton.clicked.connect(self.add_music_to_catalog)
        self.selectRandomMediaButton.clicked.connect(self.select_random_media)
        self.selectPlayTrailerButton.clicked.connect(self.select_play_trailer)
        self.restartTrailerButton.clicked.connect(self.restart_trailer)
        self.endTrailerDetectionButton.clicked.connect(self.end_trailer_detection)
        self.saveStarsSettingButton.clicked.connect(self.save_stars_setting)
        self.saveReviewButton.clicked.connect(self.save_review)
        self.startTrailerFromURLButton.clicked.connect(self.start_trailer_from_url)
        self.resetAllButton.clicked.connect(self.reset_all)
        self.saveReviewFromURLButton.clicked.connect(self.save_review_url)
        self.mongodb_client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.mongodb_client["emotion_recognizer_db"]
        
    def reset_all(self):
        self.videoLabel_2.clear()
        self.resultsBrowser_2.setText("")
        self.trailerNameBrowser.setText("")
        self.trailerPathBrowser.setText("")
        self.angerBrowser.setText("")
        self.sadnessBrowser.setText("")
        self.happinessBrowser.setText("")
        self.disgustBrowser.setText("")
        self.fearBrowser.setText("")
        self.surpriseBrowser.setText("")
        self.neutralityBrowser.setText("")
        self.mostDetectedEmotionBrowser.setText("")
        self.reviewLabel.clear()
        self.trailerNameEditText.setText("")
        self.trailerURLEditText.setText("")
        self.vc_trailer = None
        
    def start_trailer_from_url(self):
        if self.vc_trailer is None:
            name = self.trailerNameEditText.toPlainText()
            url = self.trailerURLEditText.toPlainText()
            if validators.url(url):
                self.open_stuff(url)
                self.start_detection_trailer(data, self.emotions, window_size=(800, 600), window_name='WEBCAM (Press q to end detection)', update_time=8)
            else:
                if name == "" or url == "":
                    QMessageBox.critical(self, "Empty Field(s)", "Please enter both name and url.")
                elif not validators.url(url):
                    QMessageBox.critical(self, "Sintax incorrect", "Url sintax incorrect.")
        else:
            QMessageBox.critical(self,"Operation not allowed", "The webcam is already run. Please close the webcam's currently run and retry.")
    
    def save_review_url(self):
        review = self.reviewLabel.text()
        if review == "":
            QMessageBox.critical(self, "Empty Field(s)", "No emotion detection done.")
            return
        trailer_name = self.trailerNameEditText.toPlainText()
        trailer_url = self.trailerURLEditText.toPlainText()
        most_detected_emotion = self.mostDetectedEmotionBrowser.toPlainText()
        if trailer_url == "":
            QMessageBox.critical(self, "Empty Field(s)", "No emotion detection done for a video trailer from URL.")
            return
        if trailer_name == "":
            QMessageBox.critical(self, "Empty Field(s)", "Please enter a name for the movie trailer.")
            return
        timestamp = datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        self.coll = self.db["emotion_movie_trailer_review"]
        doc = {"trailerName": trailer_name, "trailerUrl": trailer_url, "review": review, "mostDetectedEmotion": most_detected_emotion,"emotionsDetected": emotions_detected, "timestamp": timestamp}
        self.coll.insert_one(doc)
        QMessageBox.information(self, "Success", "Correctly inserted.")
        
    def save_review(self):
        review = self.reviewLabel.text()
        if review == "":
            QMessageBox.critical(self, "Empty Field(s)", "No emotion detection done.")
            return
        trailer_name = self.trailerNameBrowser.toPlainText()
        most_detected_emotion = self.mostDetectedEmotionBrowser.toPlainText()
        if trailer_name == "":
            QMessageBox.critical(self, "Empty Field(s)", "No emotion detection done for a video trailer from path.")
            return
        else:
            timestamp = datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')
            self.coll = self.db["emotion_movie_trailer_review"]
            doc = {"trailerName": trailer_name, "review": review, "mostDetectedEmotion": most_detected_emotion, "emotionsDetected": emotions_detected, "timestamp": timestamp}
            self.coll.insert_one(doc)
            QMessageBox.information(self, "Success", "Correctly inserted.") 
            
    def set_saved_stars(self):
        settings_dict = {}
        with open("stars_setting.txt", "r") as ins:
            array = []
            for line in ins:
                array.append(line)
        for line in array:
            line_parts = line.split(':')
            settings_dict[line_parts[0]] = line_parts[1]
        self.angerStarComboBox.setCurrentText(settings_dict['anger'].strip('\n'))
        self.fearStarComboBox.setCurrentText(settings_dict['fear'].strip('\n'))
        self.disgustStarComboBox.setCurrentText(settings_dict['disgust'].strip('\n'))
        self.happinessStarComboBox.setCurrentText(settings_dict['happiness'].strip('\n'))
        self.neutralityStarComboBox.setCurrentText(settings_dict['neutrality'].strip('\n'))
        self.surpriseStarComboBox.setCurrentText(settings_dict['surprise'].strip('\n'))
        self.sadnessStarComboBox.setCurrentText(settings_dict['sadness'].strip('\n'))
            
    def set_stars(self):
        if os.path.exists("stars_setting.txt"):
            self.set_saved_stars()
        else:
            self.set_predefined_stars()

    def save_stars_setting(self):
        out_file = open("stars_setting.txt","w")        
        out_file = open("stars_setting.txt","a")
        out_file.write("anger" + ":" + self.angerStarComboBox.currentText() + "\n")
        out_file.write("fear" + ":" + self.fearStarComboBox.currentText() + "\n")
        out_file.write("disgust" + ":" + self.disgustStarComboBox.currentText() + "\n")
        out_file.write("happiness" + ":" + self.happinessStarComboBox.currentText() + "\n")
        out_file.write("neutrality" + ":" + self.neutralityStarComboBox.currentText() + "\n")
        out_file.write("surprise" + ":" + self.surpriseStarComboBox.currentText() + "\n")
        out_file.write("sadness" + ":" + self.sadnessStarComboBox.currentText() + "\n")
        out_file.close()
        QMessageBox.information(self, "Success", "Correctly saved.")
        
    def set_predefined_stars(self):
        self.angerStarComboBox.setCurrentText('2')
        self.fearStarComboBox.setCurrentText('3')
        self.disgustStarComboBox.setCurrentText('2')
        self.happinessStarComboBox.setCurrentText('4')
        self.neutralityStarComboBox.setCurrentText('3')
        self.surpriseStarComboBox.setCurrentText('4')
        self.sadnessStarComboBox.setCurrentText('2')

    def start_detection_trailer(self, model, emotions, window_size=None, window_name='window', update_time=10):
        emotions_detected.clear()
        emotions_detected_count.clear()
        cv2.namedWindow(window_name, WINDOW_NORMAL)
        if window_size:
            width, height = window_size
            cv2.resizeWindow(window_name, width, height)
    
        #Set up some required objects
        self.vc_trailer = cv2.VideoCapture(0) #Webcam objects
        self.vc_trailer.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.vc_trailer.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        
        if self.vc_trailer.isOpened():
            ret, frame = self.vc_trailer.read()
            
        else:
            print("webcam not found")
            return
    
        while ret:
            training_data = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            clahe_image = clahe.apply(gray)
    
            #Get Point and Landmarks
            landmarks_vectorised, frame = self.get_landmarks_with_point(clahe_image, frame)
            #print(landmarks_vectorised)
            if landmarks_vectorised == "error":
                pass
            else:
                #Predict emotion
                training_data.append(landmarks_vectorised)
                npar_pd = np.array(training_data)
                prediction_emo_set = model.predict_proba(npar_pd)
                if cv2.__version__ != '3.1.0':
                    prediction_emo_set = prediction_emo_set[0]
                #print(zip(model.classes_, prediction_emo_set))
                prediction_emo = model.predict(npar_pd)
                if cv2.__version__ != '3.1.0':
                    prediction_emo = prediction_emo[0]
                #print(emotions[prediction_emo])
                text = ""
                for i in range(len(prediction_emo_set)):
                    text += "{}: {} \n".format(emotions[i], prediction_emo_set[i])
                #cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                self.resultsBrowser_2.setText(text)
                print("Detected emotion: %s" %emotions[prediction_emo])
                """actionlist = [x for x in actions[emotions[prediction_emo]]] #get list of actions/files for detected emotion
                random.shuffle(actionlist) #Randomly shuffle the list
                open_stuff(actionlist[0]) #Open the first entry in the list"""
                emotions_detected.append(emotions[prediction_emo])
                if emotions[prediction_emo] in emotions_detected_count:
                    emotions_detected_count[emotions[prediction_emo]] += 1
                else:
                    emotions_detected_count[emotions[prediction_emo]] = 1
                
                
            self.displayImage(frame, self.videoLabel_2, 1)
            cv2.imshow(window_name, frame)  #Display the frame
            ret, frame = self.vc_trailer.read()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # do a bit of cleanup
                self.vc_trailer.release()
                cv2.destroyAllWindows()
                print("Exit")
                sys.exit(0) #Exit program when user press 'q'
    
    def get_trailer_results(self):
        #detected_emotions_mode = statistics.mode(emotions_detected)
        detected_emotions_mode = max(set(emotions_detected), key=emotions_detected.count)
        anger_count = emotions_detected_count.get('anger')
        if anger_count is None:
            anger_count = 0
        sadness_count = emotions_detected_count.get('sadness')
        if sadness_count is None:
            sadness_count = 0
        happiness_count = emotions_detected_count.get('happy')
        if happiness_count is None:
            happiness_count = 0
        fear_count = emotions_detected_count.get('fear')
        if fear_count is None:
            fear_count = 0
        disgust_count = emotions_detected_count.get('disgust')
        if disgust_count is None:
            disgust_count = 0
        surprise_count = emotions_detected_count.get('surprise')
        if surprise_count is None:
            surprise_count = 0
        neutrality_count = emotions_detected_count.get('neutral')
        if neutrality_count is None:
            neutrality_count = 0
        self.angerBrowser.setText(str(anger_count) + ' times')
        self.happinessBrowser.setText(str(happiness_count) + ' times')
        self.fearBrowser.setText(str(fear_count) + ' times')
        self.disgustBrowser.setText(str(disgust_count) + ' times')
        self.surpriseBrowser.setText(str(surprise_count) + ' times')
        self.sadnessBrowser.setText(str(sadness_count) + ' times')
        self.neutralityBrowser.setText(str(neutrality_count) + ' times')
        self.mostDetectedEmotionBrowser.setText(str(detected_emotions_mode))
        angerStar = int(self.angerStarComboBox.currentText())
        sadnessStar = int(self.sadnessStarComboBox.currentText())
        neutralityStar = int(self.neutralityStarComboBox.currentText())
        fearStar = int(self.fearStarComboBox.currentText())
        happinessStar = int(self.happinessStarComboBox.currentText())
        disgustStar = int(self.disgustStarComboBox.currentText())
        surpriseStar = int(self.surpriseStarComboBox.currentText())

        review = ((anger_count * angerStar) + (sadness_count * sadnessStar) +
                  (happiness_count * happinessStar) + (fear_count * fearStar) + 
                  (disgust_count * disgustStar) + (surprise_count * surpriseStar) +
                  (neutrality_count * neutralityStar)) / (anger_count + sadness_count + 
                  happiness_count + fear_count + disgust_count + surprise_count + neutrality_count)
        #review = round(review, 1)
        review = round(review * 2) / 2
        self.reviewLabel.setText(str(review))
        
        
    def end_trailer_detection(self):
        if self.vc_trailer is None:
            QMessageBox.critical(self, "Error", "No webcam streaming is currently running.")
            return
        self.vc_trailer.release()
        cv2.destroyAllWindows()
        self.get_trailer_results()
  
    def restart_trailer(self):
        filename = self.trailerPathBrowser.toPlainText()
        if filename is "":
            QMessageBox.critical(self, "Error", "No movie trailer to restart.")
            return
        self.open_stuff(filename)
        #clip = VideoFileClip(fname[0])
        #print( clip.duration )
        #max_time = clip.duration
        #start_time = time.time()  # remember when we started
        self.start_detection_trailer(data, self.emotions, window_size=(800, 600), window_name='WEBCAM (Press q to end detection)', update_time=8)
        
    def select_play_trailer(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file','c:\\',"Video files (*.mp4 *.avi *.webm *.mkv *.flv *.vob *.ogv *.ogg *.drc *.gif *.gifv *.mng *.mov *.qt *.wmv *.yuv *.rm *.rmvb *.asf *.amv *.m4p *.m4v *.mpg *.mpeg *.mpe *.mpv *.mp2 *.m2v *.m4v *.svi *.flv *.nsv *.mxf )")
        if fname[0] is "":
            QMessageBox.critical(self, "File not selected", "You don't have selected any movie trailer.")
            return
        self.open_stuff(fname[0])
        #clip = VideoFileClip(fname[0])
        #print( clip.duration )
        #max_time = clip.duration
        #start_time = time.time()  # remember when we started
        self.trailerPathBrowser.setText(fname[0])
        parts = fname[0].split('/')
        length = len(parts)
        name = parts[length -1]
        self.trailerNameBrowser.setText(name)
        self.start_detection_trailer(data, self.emotions, window_size=(800, 600), window_name='WEBCAM (Press q to end detection)', update_time=8)
        #if (time.time() - start_time) > max_time:
            #self.end_detection()
      
    def select_random_media(self):
        if not self.query_results_for_random:
            QMessageBox.critical(self, "Error", "Emotion detection not started.")
            return
        selected_media = random.choice(self.query_results_for_random) 
        url_or_path = selected_media["url_or_path"]
        if validators.url(url_or_path) or os.path.isabs(url_or_path):
            self.open_stuff(url_or_path) 
            self.video_restart = url_or_path
            self.music_file = selected_media
            self.mediaNameBrowser.setText(selected_media["name"])
            self.mediaUrlBrowser.setText(selected_media["url_or_path"])
        
    def add_music_to_catalog(self):
        name = self.musicNameEditText.toPlainText()
        url_or_path = self.musicUrlEditText.toPlainText()
        emotion = self.emotionsComboBox.currentText()
        if validators.url(url_or_path) or os.path.isabs(url_or_path):
            self.coll = self.db["emotion_aware_music_player"]
            doc = {"name": name, "url_or_path": url_or_path, "emotion": emotion}
            self.coll.insert_one(doc)
            QMessageBox.information(self, "Success", "Correctly inserted.")
        else:
            if name == "" or url_or_path == "":
                QMessageBox.critical(self, "Empty Field(s)", "Please enter both name and url (or path).")
            elif not (validators.url(url_or_path) or os.path.isabs(url_or_path)):
                QMessageBox.critical(self, "Sintax incorrect", "Url or path sintax incorrect.")
                
    def restart_video_streaming(self):
        if self.video_restart is None:
            QMessageBox.critical(self, "Error", "No video to restart.")
            return
        if validators.url(self.video_restart) or os.path.isabs(self.video_restart):
            self.open_stuff(self.video_restart) 
            
    def start_detection(self):
        # use learnt model
        #show_image_test(data, emotions)
        window_name = 'WEBCAM (Press q to end detection)'
        self.show_webcam_and_run(data, self.emotions, window_size=(800, 600), window_name=window_name, update_time=8)
        
    def get_landmarks_with_point(self, image, frame):
        detections = detector(image, 1)
        #For all detected face instances individually
        for k,d in enumerate(detections):
            #get facial landmarks with prediction model
            shape = model(image, d)
            xpoint = []
            ypoint = []
            for i in range(17, 68):
                if (i == 27) | (i == 30):
                    cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) #For each point, draw a red circle with thickness2 on the original frame
                xpoint.append(float(shape.part(i).x))
                ypoint.append(float(shape.part(i).y))
    
            #center points of both axis
            xcenter = np.mean(xpoint)
            ycenter = np.mean(ypoint)
            #Calculate distance between particular points and center point
            xdistcent = [(x-xcenter) for x in xpoint]
            ydistcent = [(y-ycenter) for y in ypoint]
    
            #prevent divided by 0 value
            if xpoint[11] == xpoint[14]:
                angle_nose = 0
            else:
                #point 14 is the tip of the nose, point 11 is the top of the nose brigde
                angle_nose = int(math.atan((ypoint[11]-ypoint[14])/(xpoint[11]-xpoint[14]))*180/math.pi)
    
            #Get offset by finding how the nose brigde should be rotated to become perpendicular to the horizontal plane
            if angle_nose < 0:
                angle_nose += 90
            else:
                angle_nose -= 90
    
            landmarks = []
            for cx,cy,x,y in zip(xdistcent, ydistcent, xpoint, ypoint):
                #Add the coordinates relative to the centre of gravity
                landmarks.append(cx)
                landmarks.append(cy)
    
                #Get the euclidean distance between each point and the centre point (the vector length)
                meanar = np.asarray((ycenter,xcenter))
                centpar = np.asarray((y,x))
                dist = np.linalg.norm(centpar-meanar)
    
                #Get the angle the vector describes relative to the image, corrected for the offset that the nosebrigde has when the face is not perfectly horizontal
                if x == xcenter:
                    angle_relative = 0
                else:
                    angle_relative = (math.atan(float(y-ycenter)/(x-xcenter))*180/math.pi) - angle_nose
                    #print(anglerelative)
                landmarks.append(dist)
                landmarks.append(angle_relative)
    
        if len(detections) < 1:
            #If no face is detected set the data to value "error" to catch detection errors
            landmarks = "error"
        return landmarks, frame

    def open_stuff(self, filename): #Open the file, credit to user4815162342, on the stackoverflow link in the text above
        if sys.platform == "win32":
            try:
                os.startfile(filename)
            except:
                QMessageBox.critical(self, "Failed", "File not found.")
        else:
            opener ="open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, filename])
            
    def exec_media_based_on_emotion(self, emotion):
        query_results = []
        self.coll = self.db["emotion_aware_music_player"]
        query = { "emotion": emotion }
        media = self.coll.find(query)
        for x in media:
            query_results.append(x)
        if not query_results:
            QMessageBox.critical(self, "Failed", "No media associated to detected emotion.")
        else:
            selected_media = random.choice(query_results)
            url_or_path = selected_media["url_or_path"]
            if validators.url(url_or_path) or os.path.isabs(url_or_path):
                self.open_stuff(url_or_path) 
                self.video_restart = url_or_path
                self.query_results_for_random = query_results
                self.music_file = selected_media
                self.mediaNameBrowser.setText(selected_media["name"])
                self.mediaUrlBrowser.setText(selected_media["url_or_path"])
    
    def exec_media_based_on_mood(self, mood):
        query_results = []
        self.coll = self.db["emotion_aware_music_player"]
        if mood == "activation":
            for emotion in ["happy", "surprise"]:
                query = { "emotion": emotion }
                media = self.coll.find(query)
                for x in media:
                    query_results.append(x)
        elif mood == "deactivation":
            for emotion in ["sadness"]:
                query = { "emotion": emotion }
                media = self.coll.find(query)
                for x in media:
                    query_results.append(x)
        elif mood == "pleasantness":
            for emotion in ["neutral"]:
                query = { "emotion": emotion }
                media = self.coll.find(query)
                for x in media:
                    query_results.append(x)
        elif mood == "unpleasantness":
            for emotion in ["fear", "disgust", "anger"]:
                query = { "emotion": emotion }
                media = self.coll.find(query)
                for x in media:
                    query_results.append(x)
        if not query_results:
            QMessageBox.critical(self, "Failed", "No media associated to detected emotion.")
        else:
            selected_media = random.choice(query_results)
            url_or_path = selected_media["url_or_path"]
            if validators.url(url_or_path) or os.path.isabs(url_or_path):
                self.open_stuff(url_or_path) 
                self.video_restart = url_or_path
                self.query_results_for_random = query_results
                self.music_file = selected_media
                self.mediaNameBrowser.setText(selected_media["name"])
                self.mediaUrlBrowser.setText(selected_media["url_or_path"])
        
    def show_emotions_detected_mode(self):
        #detected_emotions_mode = statistics.mode(emotions_detected)
        detected_emotions_mode = max(set(emotions_detected), key=emotions_detected.count)
        self.modeBrowser.setText(detected_emotions_mode)
        self.lastDetectedEmotionLabel.setText(emotions_detected[-1])
        #print("Mode of detected emotions: {}".format(detected_emotions_mode))
        #self.exec_media_based_on_emotion(detected_emotions_mode)
        delta = 10
        if self.activation_mood is None:
            self.activation_mood = 0
        if self.deactivation_mood is None:
            self.deactivation_mood = 0
        if self.pleasantness_mood is None:
            self.pleasantness_mood = 0
        if self.unpleasantness_mood is None:
            self.unpleasantness_mood = 0
        if self.activation_mood > self.deactivation_mood and self.activation_mood > self.pleasantness_mood and self.activation_mood > self.unpleasantness_mood and (self.activation_mood - self.deactivation_mood) >= delta and (self.activation_mood - self.pleasantness_mood) >= delta and (self.activation_mood - self.unpleasantness_mood) >= delta:
            self.moodLabel.setText("Positive")
            self.specificMoodLabel.setText("Activation")
            self.exec_media_based_on_mood("activation")
        elif self.deactivation_mood > self.activation_mood and self.deactivation_mood > self.pleasantness_mood and self.deactivation_mood > self.unpleasantness_mood and (self.deactivation_mood - self.activation_mood) >= delta and (self.deactivation_mood - self.pleasantness_mood) >= delta and (self.deactivation_mood - self.unpleasantness_mood) >= delta:
            self.moodLabel.setText("Negative")
            self.specificMoodLabel.setText("Deactivation")
            self.exec_media_based_on_mood("deactivation")
        elif self.pleasantness_mood > self.activation_mood and self.pleasantness_mood > self.deactivation_mood and self.pleasantness_mood > self.unpleasantness_mood and (self.pleasantness_mood - self.activation_mood) >= delta and (self.pleasantness_mood - self.deactivation_mood) >= delta and (self.pleasantness_mood - self.unpleasantness_mood) >= delta:
            self.moodLabel.setText("Positive")
            self.specificMoodLabel.setText("Pleasantness")
            self.exec_media_based_on_mood("pleasantness")
        elif self.unpleasantness_mood > self.activation_mood and self.unpleasantness_mood > self.pleasantness_mood and self.unpleasantness_mood > self.deactivation_mood and (self.unpleasantness_mood - self.activation_mood) >= delta and (self.unpleasantness_mood - self.pleasantness_mood) >= delta and (self.unpleasantness_mood - self.deactivation_mood) >= delta:
            self.moodLabel.setText("Negative")
            self.specificMoodLabel.setText("Unpleasantness")
            self.exec_media_based_on_mood("unpleasantness")
        else:
            self.moodLabel.setText("Mood not detected")
            self.specificMoodLabel.setText("Mood not detected")
            QMessageBox.information(self, "Info", "Mood not detected. A music based on the last emotion detected is starting.")
            self.exec_media_based_on_emotion(emotions_detected[-1])
        """else:
            self.moodLabel.setText("Neutral")
            self.exec_media_based_on_emotion("neutral")"""


    def displayImage(self, img, videoLabel, window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outImage = QImage(img,img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        if window == 1:
            videoLabel.setPixmap(QPixmap.fromImage(outImage))
            videoLabel.setScaledContents(True)
            
    def end_detection(self):
        if self.vc is None:
            QMessageBox.critical(self, "Error", "No webcam streaming is currently running.")
            return
        # do a bit of cleanup
        self.vc.release()
        cv2.destroyAllWindows()
        self.show_emotions_detected_mode();
        
    def show_webcam_and_run(self, model, emotions, window_size=None, window_name='webcam', update_time=10):
        emotions_detected.clear()
        cv2.namedWindow(window_name, WINDOW_NORMAL)
        if window_size:
            width, height = window_size
            cv2.resizeWindow(window_name, width, height)
    
        #Set up some required objects
        self.vc = cv2.VideoCapture(0) #Webcam objects
        self.vc.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        
        if self.vc.isOpened():
            ret, frame = self.vc.read()
            
        else:
            print("webcam not found")
            return
    
        while ret:
            training_data = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            clahe_image = clahe.apply(gray)
    
            #Get Point and Landmarks
            landmarks_vectorised, frame = self.get_landmarks_with_point(clahe_image, frame)
            #print(landmarks_vectorised)
            if landmarks_vectorised == "error":
                pass
            else:
                #Predict emotion
                training_data.append(landmarks_vectorised)
                npar_pd = np.array(training_data)
                prediction_emo_set = model.predict_proba(npar_pd)
                if cv2.__version__ != '3.1.0':
                    prediction_emo_set = prediction_emo_set[0]
                #print(zip(model.classes_, prediction_emo_set))
                prediction_emo = model.predict(npar_pd)
                if cv2.__version__ != '3.1.0':
                    prediction_emo = prediction_emo[0]
                #print(emotions[prediction_emo])
                text = ""
                for i in range(len(prediction_emo_set)):
                    text += "{}: {} \n".format(emotions[i], prediction_emo_set[i])
                #cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                self.resultsBrowser.setText(text)
                print("Detected emotion: %s" %emotions[prediction_emo])
                """actionlist = [x for x in actions[emotions[prediction_emo]]] #get list of actions/files for detected emotion
                random.shuffle(actionlist) #Randomly shuffle the list
                open_stuff(actionlist[0]) #Open the first entry in the list"""
                emotions_detected.append(emotions[prediction_emo])
                if emotions[prediction_emo] == "happy" or emotions[prediction_emo] == "surprise":
                    if self.activation_mood is None:
                        self.activation_mood = 1
                    else:
                        self.activation_mood += 1
                if emotions[prediction_emo] == "anger" or emotions[prediction_emo] == "fear" or emotions[prediction_emo] == "disgust":
                    if self.unpleasantness_mood is None:
                        self.unpleasantness_mood = 1
                    else:
                        self.unpleasantness_mood += 1
                if emotions[prediction_emo] == "neutral":
                    if self.pleasantness_mood is None:
                        self.pleasantness_mood = 1
                    else:
                        self.pleasantness_mood += 1
                if emotions[prediction_emo] == "sadness":
                    if self.deactivation_mood is None:
                        self.deactivation_mood = 1
                    else:
                        self.deactivation_mood += 1

            self.image = cv2.flip(frame, 1)
            self.displayImage(self.image, self.videoLabel, 1)
            cv2.imshow(window_name, frame)  #Display the frame
            ret, frame = self.vc.read()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # do a bit of cleanup
                self.vc.release()
                cv2.destroyAllWindows()
                print("Exit")
                sys.exit(0) #Exit program when user press 'q'
                
        #self.show_emotions_detected_mode();
    
if __name__ == '__main__':
    app=QApplication(sys.argv)
    window=ProjectUI()
    window.setWindowTitle('Emotion Detection')
    window.show()
    
    detector = dlib.get_frontal_face_detector() #Face detector
    #Landmark identifyier. Set the filename to whatever you named the downloaded filename
    model = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    emotions_detected = []
    emotions_detected_count = {}
    #emotions = ["happy", "neutral", "sadness","surprise"]
    #load models
    #joblib.load('models/emotion_detection_model.xml')
    pkl_file = open('models\model1.pkl', 'rb')
    data = pickle.load(pkl_file)
    #data.predict(X[0:1])
    pkl_file.close()
    sys.exit(app.exec_())





    

    
