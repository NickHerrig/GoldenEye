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
    

    def generate_audio(self, sentence):

        data = {
            "text": sentence,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        response = requests.post(self.url, json=data, headers=self.headers)

        with open('output.mp3', 'wb') as f:
            for chunk in response.iter_content(chunk_size=self.CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

        return playsound('./output.mp3')
    
    
    def play_again(self):
        threading.Thread(target=lambda: playsound('./output.mp3')).start()



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
                cloned_voice.play_again()
                return
            
        
        if time.time() - trick_start_time >= TRICK_SECONDS:
            trick_percentage = trick_frame_count / total_frame_count
            if trick_percentage >= TRICK_THRESHOLD:
                
                trick_start_time = None
                trick_frame_count = 0
                non_trick_frame_count = 0
                total_frame_count = 0

                pipeline.terminate()
                cloned_voice.generate_audio("Good boy Ollie! Here's a treat!")


            else:
                trick_start_time = None
                trick_frame_count = 0
                non_trick_frame_count = 0
                total_frame_count = 0
                cloned_voice.play_again()
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
    
    load_dotenv()

    roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")
    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

    cloned_voice = ClonedVoice(elevenlabs_api_key)

    pipeline = InferencePipeline.init(
        model_id="goldeneye/6",
        api_key=roboflow_api_key,
        video_reference=0,
        on_prediction=lambda preds, frame: print_preds(preds, frame, pipeline, cloned_voice, args.trick),
        confidence=0.80
    )

    commands = {
        "sitting": "Ollie Sit! If you sit, you'll get a treat!",
        "lying": "Laydown Ollie, You'll get a treat if you lay down!"
    }

    cloned_voice.generate_audio("Welcome to goldeneye, the best golden retriever trainer.")
    cloned_voice.generate_audio(commands[args.trick])

    pipeline.start()