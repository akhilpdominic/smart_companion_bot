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
        r.adjust_for_ambient_noise(source2, duration=0.2)
        audio2 = r.listen(source2)
        MyText = r.recognize_google(audio2)
        MyText = MyText.lower()
        print("Did you say ",MyText)

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt="hello , how are you",
    temperature=0.9,
    max_tokens=150,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.6,
    stop=[" Human:", " AI:"]
    )

    text=response["choices"][0]["text"]


    t1 = gtts.gTTS(text)

    t1.save("welcome.mp3")

    playsound("welcome.mp3")  

#engine = pyttsx3.init()
#engine.say(text)
#engine.runAndWait()#