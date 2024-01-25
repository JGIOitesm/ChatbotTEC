import speech_recognition as sr

recognizer = sr.Recognizer()

try:
    # List available microphones (optional)
    # print("Available microphones:")
    # print(sr.Microphone.list_microphone_names())

    # Select a specific microphone (optional)
    # with sr.Microphone(device_index=1) as source:

    with sr.Microphone() as source:
        print("Adjusting noise...")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Recording for a minimum of 4 seconds...")
        recorded_audio = recognizer.listen(source, timeout=4)
        print("Done recording.")

    print("Recognizing the text...")
    text = recognizer.recognize_google(recorded_audio, language="es-MX")
    print("Decoded Text: {}".format(text))
    
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand the audio.")
except sr.RequestError:
    print("Could not request results from Google Speech Recognition service.")
except Exception as ex:
    print("Error during recognition:", ex)