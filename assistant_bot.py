import openai
import gtts
import os
import pyttsx3
from playsound import playsound  
import speech_recognition as sr


import config
r = sr.Recognizer()

while(1):

    with sr.Microphone() as source2:
       
        print("Recognizing...")
        audio_data = r.record(source2, duration=5)
        text = r.recognize_google(audio_data)
        MyText = text.lower()
        print("You said : ",MyText)

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=MyText,
    temperature=0.9,
    max_tokens=150,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.6,
    stop=[" Human:", " AI:"]
    )

    text=response["choices"][0]["text"]


    t1 = gtts.gTTS(text)

    t1.save("voice.mp3")

    playsound("voice.mp3")  

#engine = pyttsx3.init()
#engine.say(text)
#engine.runAndWait()#