from gtts import gTTS 
from playsound import playsound

# This module is imported so that we can  
# play the converted audio 
import os 
  
# The text that you want to convert to audio 
mytext = 'Bienvenido!'
  
# Language in which you want to convert 
language = 'es'
  
# Passing the text and language to the engine,  
# here we have marked slow=False. Which tells  
# the module that the converted audio should  
# have a high speed 
myobj = gTTS(text=mytext, lang=language, slow=False) 
  
# Saving the converted audio in a mp3 file named 
# welcome  

name = mytext + ".mp3"

myobj.save(name) 
  
# Playing the converted file 
playsound(name)
os.remove(name)