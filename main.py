import os
import logging
import time
import threading

from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes
import requests
from playsound import playsound
from dotenv import load_dotenv


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

trick_start_time = None
trick_frame_count = 0
total_frame_count = 0
non_trick_frame_count = 0

TRICK_SECONDS = 3
TRICK_THRESHOLD = 0.80
NON_TRICK_GRACE_FRAMES = 30

class ClonedVoice:
    def __init__(self, api_key):
        self.CHUNK_SIZE = 1024
        self.url = "https://api.elevenlabs.io/v1/text-to-speech/rpZrpySS8GErxLLUYJ3p"
        self.headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }    
    

    def generate_audio(self, command, sentence):

        data = {
            "text": sentence,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        response = requests.post(self.url, json=data, headers=self.headers)

        with open(f"{command}.mp3", "wb") as f:
            for chunk in response.iter_content(chunk_size=self.CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
    
    
    def speak(self, command, blocking=False):

        if blocking:
            playsound(f"./{command}.mp3")
        else:
            threading.Thread(target=lambda: playsound(f"./{command}.mp3")).start()



def print_preds(predictions, video_frame, pipeline, cloned_voice, trick):
    render_boxes(predictions, video_frame)

    global trick_start_time, trick_frame_count, total_frame_count, non_trick_frame_count

    # If the dog is sitting, initialize tracking variables
    if trick_start_time is None and predictions["predictions"]:
        if predictions["predictions"][0]["class"] == trick:
            trick_start_time = time.time()
            trick_frame_count = 1
            total_frame_count = 1
            logging.info("First trick frame detected, starting timer for trick evaluation")
            return
        
    # If timer is running, add a frame and increment proper state variables
    if trick_start_time is not None:
        total_frame_count += 1

        if predictions["predictions"] and predictions["predictions"][0]["class"] == trick:
            trick_frame_count += 1
            non_trick_frame_count = 0

        else: 
            non_trick_frame_count += 1
            if non_trick_frame_count > NON_TRICK_GRACE_FRAMES:
                trick_start_time = None
                trick_frame_count = 0
                non_trick_frame_count = 0
                total_frame_count = 0
                logging.info("Dog is not doing trick, resetting trick evaluation")
                cloned_voice.speak("command")
                return
            
        
        if time.time() - trick_start_time >= TRICK_SECONDS:
            trick_percentage = trick_frame_count / total_frame_count
            if trick_percentage >= TRICK_THRESHOLD:
                
                trick_start_time = None
                trick_frame_count = 0
                non_trick_frame_count = 0
                total_frame_count = 0

                pipeline.terminate()
                cloned_voice.speak("affirmation")

            else:
                trick_start_time = None
                trick_frame_count = 0
                non_trick_frame_count = 0
                total_frame_count = 0
                cloned_voice.speak("command")
                logging.info("Percentage of tricks detected was too low during timeframe.")
            


if __name__=="__main__":

    # command line arg parser
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trick", help="The trick you want to train your dog to do. Either sitting or lying.")
    args = parser.parse_args()

    if args.trick not in ["sitting", "lying"]:
        parser.print_help()
        raise ValueError("Trick must be either sitting or lying") 
    
    # Load environment variables
    load_dotenv()
    roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")
    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")


    # Generate audio files from voice clone using eleven labs
    commands = {
        "sitting": """Do you want a treat? Ollie Sit!""",
        "lying": "Do you want a treat Ollie? Lay down!"
    }
    cloned_voice = ClonedVoice(elevenlabs_api_key)
    cloned_voice.generate_audio("boot", "Welcome to goldeneye A treat dispenser built with eyes and a voice.")
    cloned_voice.generate_audio("command", commands[args.trick])
    cloned_voice.generate_audio("affirmation", f"Good Boy for {args.trick} Ollie! Here's a treat!")

    # Init the roboflow inference pipeline
    pipeline = InferencePipeline.init(
        model_id="goldeneye/8",
        api_key=roboflow_api_key,
        video_reference=0,
        on_prediction=lambda preds, frame: print_preds(preds, frame, pipeline, cloned_voice, args.trick),
        confidence=0.70
    )

    # Welcome the user and issue trick command.
    cloned_voice.speak("boot", blocking=True)
    cloned_voice.speak("command")

    # Start the pipeline
    pipeline.start()