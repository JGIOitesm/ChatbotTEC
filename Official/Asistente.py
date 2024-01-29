from gtts import gTTS 
from playsound import playsound
import speech_recognition as sr
from transformers import pipeline
from datetime import datetime
import os 
from transformers import pipeline, AutoTokenizer, Conversation

REC = sr.Recognizer()
NLPQA = pipeline("question-answering", model="nlp-en-es/roberta-base-bne-finetuned-sqac")
NLPSA = pipeline("text-classification", model="finiteautomata/beto-sentiment-analysis",top_k=None)

TOKENIZER = AutoTokenizer.from_pretrained('stabilityai/stablelm-2-zephyr-1_6b', trust_remote_code=True)
CHATBOT = pipeline("conversational", "stabilityai/stablelm-2-zephyr-1_6b",trust_remote_code=True,tokenizer=TOKENIZER)
CONV = Conversation([{"role": "system","content": "Es el asistente virtual de la tienda patito y resuelve cualquier duda de los clientes."}])

class Log:
    def __init__(self):
        self.start = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.end = ""
        self.logList = []

    def append(self,list):
        self.logList.append(' | '.join(map(str,list)))

    def generateLog(self):
        self.end = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        with open(self.start+"_"+self.end+'.txt', 'a') as fp:
            for i in self.logList:
                fp.write("%s\n" % i)

LOG = Log()

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
            printTTS("Hubo un problema, intente de nuevo")

def nlpqafunc(question,context):
    d = NLPQA(question, context)
    LOG.append(["NLPQA",question,context,d["score"],d["answer"]])
    return d["answer"]

def nlpsafunc(answer):
    d = NLPSA(answer)
    l = ["NLPSA",answer]
    for i in d[0]:
        l.extend(list(i.values()))
    LOG.append(l)
    if d[0][0]["label"] == "POS" or (d[0][0]["label"] == "NEU" and d[0][1]["label"] == "POS"):
        printTTS("Respuesta identificada como positiva")
        return True
    elif d[0][0]["label"] == "NEG" or (d[0][0]["label"] == "NEU" and d[0][1]["label"] == "NEG"):
        printTTS("Respuesta identificada como negativa")
        return False
    else:
        printTTS("Respuesta no identificada, intente de nuevo")
        return False
    
def askChatbot():
    global CONV, CHATBOT, TOKENIZER
    printTTS("¿Que duda tiene?")
    CONV.add_message({"role": "user", "content": inputSTT()})
    l = ["Chatbot","user",CONV.messages[-1]["content"]]
    printTTS("Espere su respuesta por favor")
    CONV = CHATBOT(CONV, pad_token_id=TOKENIZER.eos_token_id)
    printTTS(CONV.messages[-1]["content"])
    l.extend(["assistant",CONV.messages[-1]["content"]])
    LOG.append(l)

def main():
    global CONV
    try:
        printTTS("Bienvenido a nuestro sistema de la tienda patito. Yo te atendere como tu asistente virtual.")
        printTTS("Le informamos que esta llamada está siendo registrada con el fin de mejorar la calidad de nuestro servicio de atención al cliente.")
        printTTS("Agradecemos su comprensión y colaboración en este proceso.")
        b0 = False
        while not b0:
            printTTS("¿Puede ofrecernos su primer nombre y apellido?")
            name = nlpqafunc("¿Cuál es su nombre?",inputSTT())
            printTTS("¿Es "+name+" su nombre?")
            b0 = nlpsafunc(inputSTT())
        b0 = False
        while not b0:
            printTTS("¿Puede ofrecernos su número telefónico?")
            phone = inputSTT()
            l = ["Num. Detection", phone]
            phone = "".join([i for i in phone.split() if i.isdigit()])
            l.append(phone)
            LOG.append(l)
            printTTS("¿Es "+phone+" su número telefónico?")
            b0 = nlpsafunc(inputSTT())
        printTTS("Usuario Identificado")
        CONV.add_message({"role": "user", "content": "Mi nombre es "+name+" y siempre refierete a mi primer nombre de la manera más amable y respetuosa por favor"})
        b0 = True
        while b0:
            askChatbot()
            printTTS("¿Fue su respuesta útil?")
            nlpsafunc(inputSTT())
            printTTS("¿Quiere hacer más preguntas?")
            b0 = nlpsafunc(inputSTT())
        printTTS("Gracias por llamarnos. Esperamos volverlo a ver pronto")

    finally:
        LOG.generateLog()
if __name__ == "__main__":
    main()