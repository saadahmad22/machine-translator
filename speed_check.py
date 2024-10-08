import timeit
from langchain_app import LangChainApp
import warnings

# Define a function to initialize the LangChainApp

def init_langchain_app():
    app: LangChainApp = LangChainApp()

# Define a function to process an audio file
def process_audio_file():
    app: LangChainApp = LangChainApp()
    audio_file_path = "days-of-week.mp3"
    app.process_audio_file(audio_file_path)

def audio_pre_processing():
    audio_file_path = "days-of-week.mp3"
    LangChainApp.preprocess_audio(audio_file_path)

with warnings.catch_warnings(action="ignore"):
    # Measure the time taken to initialize the LangChainApp
    init_time = timeit.timeit("init_langchain_app()", setup="from __main__ import init_langchain_app", number=1)
    print(f"Initialization time: {init_time} seconds")

    # Measure the time taken to preprocess the audio file
    n = 10
    preprocess_time = timeit.timeit("audio_pre_processing()", setup="from __main__ import audio_pre_processing", number=n) / n
    print(f"Preprocessing time: {preprocess_time} seconds")

    # Measure the time taken to process the audio file
    n = 10
    process_time = timeit.timeit("process_audio_file()", setup="from __main__ import process_audio_file", number=n)
    print(f"Processing time: {(process_time - (init_time + preprocess_time) * n) / n} seconds")