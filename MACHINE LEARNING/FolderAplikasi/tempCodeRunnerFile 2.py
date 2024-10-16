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
import webbrowser

import subprocess


class HomePage(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = GridLayout(cols=1, spacing=10, padding=10)

        open_youtube_button = Button(text="Open YouTube")
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
        popup = Popup(title=title, content=layout, size_hint=(None, None), size=(400, 200))
        close_button.bind(on_press=popup.dismiss)
        popup.open()

class YouTubePage(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = GridLayout(cols=1, spacing=10, padding=10)

        self.language_input = TextInput(hint_text='Masukkan Bahasa (Contoh: Bahasa Inggris)')
        self.layout.add_widget(Label(text='Bahasa:'))
        self.layout.add_widget(self.language_input)

        self.artist_name_input = TextInput(hint_text='Masukkan Nama Artis')
        self.layout.add_widget(Label(text='Nama Artis:'))
        self.layout.add_widget(self.artist_name_input)

        detect_face_button = Button(text="Open Camera for Face Detection")
        detect_face_button.bind(on_press=self.detect_face)
        self.layout.add_widget(detect_face_button)

        open_youtube_button = Button(text="Open YouTube")
        open_youtube_button.bind(on_press=self.open_youtube)
        self.layout.add_widget(open_youtube_button)

        back_button = Button(text="Back to Home")
        back_button.bind(on_press=self.back_to_home)
        self.layout.add_widget(back_button)

        self.add_widget(self.layout)

    def detect_face(self, instance):
        self.language = self.language_input.text.strip()
        self.artist_name = self.artist_name_input.text.strip()

        if not self.language or not self.artist_name:
            self.show_error_popup("Error", "Masukkan bahasa dan nama artis!")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.show_error_popup("Error", "Camera not detected!")
            return

        holistic = mp.solutions.holistic
        holis = holistic.Holistic()
        drawing = mp.solutions.drawing_utils

        # Memuat model dan label
        model = load_model("dataModel.h5")
        label = np.load("labels.npy")

        def find_emotion(face_landmarks):
            def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        # Membalikkan citra secara horizontal
        frm = cv2.flip(frm, 1)

        # Memproses citra dengan detektor holistik dari MediaPipe
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        # Inisialisasi list untuk menyimpan posisi landmark
        lst = []

        # Deteksi landmark wajah dan tangan
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

            # Melakukan prediksi emosi berdasarkan landmark yang dideteksi
            lst = np.array(lst).reshape(1,-1)
            pred = label[np.argmax(model.predict(lst))]

            # Menampilkan prediksi emosi pada video
            print(pred)
            cv2.putText(frm, pred, (50,50), cv2.FONT_ITALIC, 1, (255,0,0),2)

            # Menyimpan data emosi
            np.save("datasetWajah.npy", np.array([pred]))

        # Menggambar landmark wajah dan tangan pada citra
        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                               landmark_drawing_spec=drawing.DrawingSpec(color=(0,0,255), thickness=-1, circle_radius=1),
                               connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        # Mengembalikan frame video dengan emosi yang ditampilkan
        return av.VideoFrame.from_ndarray(frm, format="bgr24")
        cap.release()

        # Open YouTube
        search_query = f"{self.language} {self.artist_name} {emotion}"
        webbrowser.open(f"https://www.youtube.com/results?search_query={search_query}")

    def open_youtube(self, instance):
        if hasattr(self, 'language') and hasattr(self, 'artist_name'):
            search_query = f"{self.language} {self.artist_name}"
            webbrowser.open(f"https://www.youtube.com/results?search_query={search_query}")
        else:
            self.show_error_popup("Error", "Masukkan bahasa dan nama artis!")

    def back_to_home(self, instance):
        self.manager.current = 'home'

    def show_error_popup(self, title, message):
        layout = GridLayout(cols=1, padding=10)
        popup_label = Label(text=message)
        close_button = Button(text="Close")
        layout.add_widget(popup_label)
        layout.add_widget(close_button)
        popup = Popup(title=title, content=layout, size_hint=(None, None), size=(400, 200))
        close_button.bind(on_press=popup.dismiss)
        popup.open()


class CameraPage(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = GridLayout(cols=1, spacing=10, padding=10)

        self.name_input = TextInput(hint_text='Masukkan Nama Mimik Gesture Wajah')
        self.layout.add_widget(Label(text='Nama Mimik Gesture Wajah:'))
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
            self.show_error_popup("Error", "Masukkan nama mimik gesture wajah!")
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

        subprocess.run(["python3", "./CollectDataset/TrainingAplikasi.py"])


    def show_error_popup(self, title, message):
        layout = GridLayout(cols=1, padding=10)
        popup_label = Label(text=message)
        close_button = Button(text="Close")
        layout.add_widget(popup_label)
        layout.add_widget(close_button)
        popup = Popup(title=title, content=layout, size_hint=(None, None), size=(400, 200))
        close_button.bind(on_press=popup.dismiss)
        popup.open()


class CameraDetectPage(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = GridLayout(cols=1, spacing=10, padding=10)

        detect_face_button = Button(text="Start Face Detection")
        detect_face_button.bind(on_press=self.detect_face)
        self.layout.add_widget(detect_face_button)

        back_button = Button(text="Back to YouTube Page")
        back_button.bind(on_press=self.back_to_youtube)
        self.layout.add_widget(back_button)

        self.add_widget(self.layout)

    def detect_face(self, instance):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.show_error_popup("Error", "Camera not detected!")
            return

        holistic = mp.solutions.holistic
        holis = holistic.Holistic()
        drawing = mp.solutions.drawing_utils

        while True:
            _, frm = cap.read()
            frm = cv2.flip(frm, 1)
            res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

            if res.face_landmarks:
                drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
                cv2.imshow("window", frm)
                break

            cv2.imshow("window", frm)

            if cv2.waitKey(1) == 27:
                break

        cv2.destroyAllWindows()
        cap.release()

        # Open YouTube
        self.manager.get_screen('youtube').open_youtube(None)

    def back_to_youtube(self, instance):
        self.manager.current = 'youtube'

    def show_error_popup(self, title, message):
        layout = GridLayout(cols=1, padding=10)
        popup_label = Label(text=message)
        close_button = Button(text="Close")
        layout.add_widget(popup_label)
        layout.add_widget(close_button)
        popup = Popup(title=title, content=layout, size_hint=(None, None), size=(400, 200))
        close_button.bind(on_press=popup.dismiss)
        popup.open()


class DatasetPage(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = GridLayout(cols=1, spacing=10, padding=10)

        self.dataset_name_input = TextInput(hint_text='Masukkan Nama File Dataset')
        self.layout.add_widget(Label(text='Nama File Dataset:'))
        self.layout.add_widget(self.dataset_name_input)

        load_dataset_button = Button(text="Load Dataset")
        load_dataset_button.bind(on_press=self.load_dataset)
        self.layout.add_widget(load_dataset_button)

        back_button = Button(text="Back to Home")
        back_button.bind(on_press=self.back_to_home)
        self.layout.add_widget(back_button)

        self.add_widget(self.layout)

    def load_dataset(self, instance):
        dataset_name = self.dataset_name_input.text.strip()

        if not dataset_name:
            self.show_error_popup("Error", "Masukkan nama file dataset!")
            return

        if os.path.exists(f"{dataset_name}.npy"):
            dataset = np.load(f"{dataset_name}.npy")
            self.dataset = dataset
            self.show_error_popup("Success", "Dataset berhasil dimuat.")
        else:
            self.show_error_popup("Error", "File dataset tidak ditemukan.")

    def back_to_home(self, instance):
        self.manager.current = 'home'

    def show_error_popup(self, title, message):
        layout = GridLayout(cols=1, padding=10)
        popup_label = Label(text=message)
        close_button = Button(text="Close")
        layout.add_widget(popup_label)
        layout.add_widget(close_button)
        popup = Popup(title=title, content=layout, size_hint=(None, None), size=(400, 200))
        close_button.bind(on_press=popup.dismiss)
        popup.open()


class DataCollectionApp(App):
    def build(self):
        self.screen_manager = ScreenManager()

        self.home_page = HomePage(name="home")
        self.screen_manager.add_widget(self.home_page)

        self.youtube_page = YouTubePage(name="youtube")
        self.screen_manager.add_widget(self.youtube_page)

        self.camera_page = CameraPage(name="camera")
        self.screen_manager.add_widget(self.camera_page)

        self.dataset_page = DatasetPage(name="dataset")
        self.screen_manager.add_widget(self.dataset_page)

        self.camera_detect_page = CameraDetectPage(name="camera_detect")
        self.screen_manager.add_widget(self.camera_detect_page)

        return self.screen_manager

if __name__ == "__main__":
    DataCollectionApp().run()
