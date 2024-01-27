from gtts import gTTS 
from playsound import playsound
import speech_recognition as sr
from transformers import pipeline
from datetime import datetime
import os 
import string
import random

REC = sr.Recognizer()

def printTTS(text):
    n = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".mp3"
    myobj = gTTS(text=text, lang="es", slow=False) 
    myobj.save(n) 
    playsound(n)
    os.remove(n)

def inputSTT():
    while True:
        try:
            printTTS("Ajustando")
            with sr.Microphone() as source:
                REC.adjust_for_ambient_noise(source, duration=2)
                printTTS("Grabando")
                recorded_audio = REC.listen(source, timeout=4)
            text = REC.recognize_google(recorded_audio, language="es-MX")
            return text
        
        except sr.UnknownValueError:
            printTTS("No te pudimos entender. Inténtalo de nuevo")
        except sr.RequestError:
            printTTS("No se pudo conectar con el servicio. Intentalo de nuevo")
        except Exception as ex:
            print("Error during recognition:", ex)

def main():
    printTTS("Bienvenido a nuestro sistema de la tienda patito. Yo te atenderé como tu asistente virtual. ¿Puede ofrecernos su nombre completo siendo primer nombre y apellido en ese orden por favor?")
    print(inputSTT())

if __name__ == "__main__":
    main()