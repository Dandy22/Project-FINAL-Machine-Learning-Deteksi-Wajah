import os
import mediapipe as mp
import numpy as np
import cv2
from keras.models import load_model
                                    
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.core.window import Window

import webbrowser
import subprocess

class HomePage(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = GridLayout(cols=1, spacing=10, padding=10)

        open_youtube_button = Button(text="Open Music Recommendation")
        open_youtube_button.bind(on_press=self.open_youtube_page)
        self.layout.add_widget(open_youtube_button)

        collect_data_button = Button(text="Collect Data from Camera")
        collect_data_button.bind(on_press=self.collect_data)
        self.layout.add_widget(collect_data_button)

        collect_dataset_button = Button(text="Collect Data from Dataset")
        collect_dataset_button.bind(on_press=self.collect_dataset)
        self.layout.add_widget(collect_dataset_button)

        self.add_widget(self.layout)

    def open_youtube_page(self, instance):
        self.manager.current = 'youtube'

    def collect_data(self, instance):
        self.manager.current = 'camera'

    def collect_dataset(self, instance):
        self.manager.current = 'dataset'

    def show_error_popup(self, title, message):
        layout = GridLayout(cols=1, padding=10)
        popup_label = Label(text=message)
        close_button = Button(text="Close")
        layout.add_widget(popup_label)
        layout.add_widget(close_button)
        popup = Popup(title=title, content=layout, size_hint=(None, None), size=(800, 600))
        close_button.bind(on_press=popup.dismiss)
        popup.open()

class YouTubePage(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = GridLayout(cols=1, spacing=10, padding=10)

        self.language_input = TextInput(hint_text='Input Language (Example: English)')
        self.layout.add_widget(Label(text='Language:'))
        self.layout.add_widget(self.language_input)

        self.artist_name_input = TextInput(hint_text='Input Artist Name (Example: Ariana Grande)')
        self.layout.add_widget(Label(text='Artist Name:'))
        self.layout.add_widget(self.artist_name_input)

        detect_face_button = Button(text="Open Camera for Face Detection")
        detect_face_button.bind(on_press=self.detect_face)
        self.layout.add_widget(detect_face_button)

        back_button = Button(text="Back to Home")
        back_button.bind(on_press=self.back_to_home)
        self.layout.add_widget(back_button)

        self.youtube_button = Button(text="Find Your Music Recommendation", disabled=True)
        self.youtube_button.bind(on_press=self.open_youtube_search)
        self.layout.add_widget(self.youtube_button)

        self.add_widget(self.layout)

        self.emotion = None
        self.cap = None  

    def detect_face(self, instance):
        self.language = self.language_input.text.strip()
        self.artist_name = self.artist_name_input.text.strip()
        if not self.language or not self.artist_name:
            self.show_error_popup("Error", "Input Your language and Artist Name")
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.show_error_popup("Error", "Camera not detected!")
            return

        holistic = mp.solutions.holistic
        holis = holistic.Holistic()
        drawing = mp.solutions.drawing_utils
        hands = mp.solutions.hands

        model = load_model("dataModel.h5")
        label = np.load("labels.npy")

        def find_emotion(frame):
            frm = cv2.flip(frame, 1)
            res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

            lst = []

            if res.face_landmarks:
                for i in res.face_landmarks.landmark:
                    lst.append(i.x - res.face_landmarks.landmark[1].x)
                    lst.append(i.y - res.face_landmarks.landmark[1].y)

                if res.left_hand_landmarks:
                    for i in res.left_hand_landmarks.landmark:
                        lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                        lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
                else:
                    for i in range(42):
                        lst.append(0.0)

                if res.right_hand_landmarks:
                    for i in res.right_hand_landmarks.landmark:
                        lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                        lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
                else:
                    for i in range(42):
                        lst.append(0.0)

                lst = np.array(lst).reshape(1, -1)
                pred = label[np.argmax(model.predict(lst))]

                cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

                np.save("datasetWajah.npy", np.array([pred]))

                drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                                    landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=1),
                                    connection_drawing_spec=drawing.DrawingSpec(thickness=1))
                drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
                drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

                return pred, frm
            return None, frm

        def update_frame(dt):
            ret, frame = self.cap.read()
            if not ret:
                return

            emotion, processed_frame = find_emotion(frame)
            cv2.imshow("window", processed_frame)
            if emotion:
                self.emotion = emotion

        def close_camera():
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            self.youtube_button.disabled = False
            self.show_confirmation_popup(f"Detected Emotion: {self.emotion}", f"{self.language} {self.artist_name} {self.emotion}")

        def on_close():
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()

        Clock.schedule_once(lambda dt: close_camera(), 10)  
        Clock.schedule_interval(update_frame, 1 / 30)  
        Window.bind(on_request_close=lambda *args: (on_close(), True))

    def open_youtube_search(self, instance):
        webbrowser.open(f"https://www.youtube.com/results?search_query={self.language}  +{self.artist_name}+{self.emotion}")

    def back_to_home(self, instance):
        self.manager.current = 'home'

    def show_error_popup(self, title, message):
        layout = GridLayout(cols=1, padding=10)
        popup_label = Label(text=message)
        close_button = Button(text="Close")
        layout.add_widget(popup_label)
        layout.add_widget(close_button)
        popup = Popup(title=title, content=layout, size_hint=(None, None), size=(800, 800))
        close_button.bind(on_press=popup.dismiss)
        popup.open()

    def show_confirmation_popup(self, title, message):
        layout = GridLayout(cols=1, padding=10)
        popup_label = Label(text=message)
        close_button = Button(text="Find your Music Recommendation")
        layout.add_widget(popup_label)
        close_button.bind(on_press=lambda x: self.open_youtube_search(message))
        layout.add_widget(close_button)
        popup = Popup(title=title, content=layout, size_hint=(None, None), size=(800, 600))
        close_button.bind(on_press=popup.dismiss) 
        popup.open()


class CameraPage(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = GridLayout(cols=1, spacing=10, padding=10)

        self.name_input = TextInput(hint_text='Input Face Gesture')
        self.layout.add_widget(Label(text='Face Gesture Name:'))
        self.layout.add_widget(self.name_input)

        collect_data_button = Button(text="Collect Data from Camera")
        collect_data_button.bind(on_press=self.collect_data)
        self.layout.add_widget(collect_data_button)

        back_button = Button(text="Back to Home")
        back_button.bind(on_press=self.back_to_home)
        self.layout.add_widget(back_button)

        self.add_widget(self.layout)

    def back_to_home(self, instance):
        self.manager.current = 'home'

    def collect_data(self, instance):
        name = self.name_input.text.strip()
        if not name:
            self.show_error_popup("Error", "Input your Face Gesture Name")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.show_error_popup("Error", "Camera not detected!")
            return

        holistic = mp.solutions.holistic
        hands = mp.solutions.hands
        holis = holistic.Holistic()
        drawing = mp.solutions.drawing_utils

        X = []
        data_size = 0

        while True:
            lst = []
            _, frm = cap.read()
            frm = cv2.flip(frm, 1)
            res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

            if res.face_landmarks:
                for i in res.face_landmarks.landmark:
                    lst.append(i.x - res.face_landmarks.landmark[1].x)
                    lst.append(i.y - res.face_landmarks.landmark[1].y)

                if res.left_hand_landmarks:
                    for i in res.left_hand_landmarks.landmark:
                        lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                        lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
                else:
                    for i in range(42):
                        lst.append(0.0)

                if res.right_hand_landmarks:
                    for i in res.right_hand_landmarks.landmark:
                        lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                        lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
                else:
                    for i in range(42):
                        lst.append(0.0)

                X.append(lst)
                data_size += 1

            drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
            drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
            drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

            cv2.putText(frm, str(data_size), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("window", frm)

            if cv2.waitKey(1) == 27 or data_size > 99:
                break

        cv2.destroyAllWindows()
        cap.release()

        np.save(f"{name}.npy", np.array(X))
        print(np.array(X).shape)

        subprocess.run(["python3", "./FolderAplikasi/TrainingAplikasi.py"])


    def show_error_popup(self, title, message):
        layout = GridLayout(cols=1, padding=10)
        popup_label = Label(text=message)
        close_button = Button(text="Close")
        layout.add_widget(popup_label)
        layout.add_widget(close_button)
        popup = Popup(title=title, content=layout, size_hint=(None, None), size=(800, 800))
        close_button.bind(on_press=popup.dismiss)
        popup.open()


class DatasetPage(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = GridLayout(cols=1, spacing=10, padding=10)

        collect_dataset_button = Button(text="Collect Data from Dataset")
        collect_dataset_button.bind(on_press=self.collect_dataset)
        self.layout.add_widget(collect_dataset_button)

        back_button = Button(text="Back to Home")
        back_button.bind(on_press=self.back_to_home)
        self.layout.add_widget(back_button)

        self.add_widget(self.layout)

    def back_to_home(self, instance):
        self.manager.current = 'home'

    def collect_dataset(self, instance):
        dataset_path = '/dataset/.'
        if not os.path.exists(dataset_path):
            self.show_error_popup("Error", f"Dataset path '{dataset_path}' does not exist!")
            return

        holistic = mp.solutions.holistic
        hands = mp.solutions.hands
        holis = holistic.Holistic()

        X = []
        y = []

        for folder in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    if file.endswith(".mp4") or file.endswith(".avi"):
                        cap = cv2.VideoCapture(file_path)
                        while cap.isOpened():
                            lst = []
                            ret, frm = cap.read()
                            if not ret:
                                break

                            frm = cv2.flip(frm, 1)
                            res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

                            if res.face_landmarks:
                                for i in res.face_landmarks.landmark:
                                    lst.append(i.x - res.face_landmarks.landmark[1].x)
                                    lst.append(i.y - res.face_landmarks.landmark[1].y)

                                if res.left_hand_landmarks:
                                    for i in res.left_hand_landmarks.landmark:
                                        lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                                        lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
                                else:
                                    for i in range(42):
                                        lst.append(0.0)

                                if res.right_hand_landmarks:
                                    for i in res.right_hand_landmarks.landmark:
                                        lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                                        lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
                                else:
                                    for i in range(42):
                                        lst.append(0.0)

                                X.append(lst)
                                y.append(folder)

                        cap.release()

        X = np.array(X)
        y = np.array(y)

        if os.path.isfile("datasetWajah.npy"):
            old = np.load("datasetWajah.npy")
            X = np.concatenate((old, X))

        if os.path.isfile("labels.npy"):
            old = np.load("labels.npy")
            y = np.concatenate((old, y))

        np.save("datasetWajah.npy", X)
        np.save("labels.npy", y)

    def show_error_popup(self, title, message):
        layout = GridLayout(cols=1, padding=10)
        popup_label = Label(text=message)
        close_button = Button(text="Close")
        layout.add_widget(popup_label)
        layout.add_widget(close_button)
        popup = Popup(title=title, content=layout, size_hint=(None, None), size=(800, 800))
        close_button.bind(on_press=popup.dismiss)
        popup.open()


class MyScreenManager(ScreenManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_widget(HomePage(name='home'))
        self.add_widget(YouTubePage(name='youtube'))
        self.add_widget(CameraPage(name='camera'))
        self.add_widget(DatasetPage(name='dataset'))


class MyApp(App):
    def build(self):
        return MyScreenManager()


if __name__ == '__main__':
    MyApp().run()
