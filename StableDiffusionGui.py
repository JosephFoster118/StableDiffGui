import tkinter as tk
from tkinter import ttk
import time
from threading import Thread
from PIL import Image, ImageTk
import torch
import torchvision.transforms as T
from datetime import datetime
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionImg2ImgPipeline
import typing
import os



NUMBER_OF_FRAMES_DEFAULT = 50
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
#DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"
#DEFAULT_MODEL = "stabilityai/stable-diffusion-2-1"
#DEFAULT_MODEL = "stabilityai/control-lora"
#DEFAULT_MODEL = "hotshotco/SDXL-512"
DEFAULT_MODEL = "gsdf/Counterfeit-V2.5" #Good anime
#DEFAULT_MODEL = "segmind/SSD-1B"
#DEFAULT_MODEL = "dreamlike-art/dreamlike-photoreal-2.0"


class MainGui(tk.Frame):
    def __init__(self, root, **kwargs):
        super().__init__(**kwargs)
        self.image = ImageTk.PhotoImage(Image.new("RGB", (512,512), (0,0,0)))
        self.root = root
        self.progress = 10
        self.setupInputFrame()
        self.setupGenerationFrame()
        self.generateImageViewer()
        self.onGenerate = None
        
        

    def setupInputFrame(self):
        self.input_frame = tk.LabelFrame(self.root, width=30, text="Settings")

        #Prompt
        self.prompts_frame = tk.Frame(self.input_frame)
        
        self.prompt_frame = tk.Frame(self.prompts_frame)
        self.prompt_label = tk.Label(self.prompt_frame, text="Prompt", width=10)
        self.prompt_label.pack(side="left", padx=4)
        self.prompt_entry = tk.Entry(self.prompt_frame)
        self.prompt_entry.pack(side="right", padx=4, fill="x", expand=True)
        self.prompt_frame.pack(side="top", fill="x", expand=True, anchor="nw")

        self.negative_prompt_frame = tk.Frame(self.prompts_frame)
        self.negative_prompt_label = tk.Label(self.negative_prompt_frame, text="Neg-Prompt", width=10)
        self.negative_prompt_label.pack(side="left", padx=4)
        self.negative_prompt_entry = tk.Entry(self.negative_prompt_frame)
        self.negative_prompt_entry.pack(side="right", padx=4, fill="x", expand=True)
        self.negative_prompt_frame.pack(side="top", fill="x", expand=True, anchor="nw")


        self.prompts_frame.pack(side="left", fill="x", expand=True, anchor="nw")


        #Settings
        self.generation_settings_frame = tk.Frame(self.input_frame, width=60)
        
        self.number_of_steps_frame = tk.Frame(self.generation_settings_frame)
        self.number_of_steps_label = tk.Label(self.number_of_steps_frame, text="Steps")
        self.number_of_steps_entry = tk.Entry(self.number_of_steps_frame)
        self.number_of_steps_entry.insert(0,str(NUMBER_OF_FRAMES_DEFAULT))
        self.number_of_steps_label.pack(side="left", padx=4)
        self.number_of_steps_entry.pack(side="right", padx=4, fill="x", expand=True)
        self.number_of_steps_frame.pack(side="top")

        self.width_frame = tk.Frame(self.generation_settings_frame)
        self.width_label = tk.Label(self.width_frame, text="Width")
        self.width_entry = tk.Entry(self.width_frame)
        self.width_entry.insert(0,str(DEFAULT_WIDTH))
        self.width_label.pack(side="left", padx=4)
        self.width_entry.pack(side="right", padx=4, fill="x", expand=True)
        self.width_frame.pack(side="top")

        self.height_frame = tk.Frame(self.generation_settings_frame)
        self.height_label = tk.Label(self.height_frame, text="Height")
        self.height_entry = tk.Entry(self.height_frame)
        self.height_entry.insert(0,str(DEFAULT_WIDTH))
        self.height_label.pack(side="left", padx=4)
        self.height_entry.pack(side="right", padx=4, fill="x", expand=True)
        self.height_frame.pack(side="top")

        self.generation_settings_frame.pack(side="left")
        self.input_frame.pack(side="bottom", fill="x")

    def setupGenerationFrame(self):
        self.generation_frame = tk.LabelFrame(self.root, text="Generation")
        
        #Progress frame
        self.progress_frame = tk.Frame(self.generation_frame)
        self.status_label = tk.Label(self.progress_frame, text="Status: Idle")
        self.generation_progress_bar = ttk.Progressbar(self.progress_frame, orient="horizontal", mode="determinate", maximum=100)
        self.status_label.pack(side="top", anchor="nw")
        self.generation_progress_bar.pack(side="bottom", fill="x", expand=True)
        self.progress_frame.pack(side="left", fill="x", expand=True, anchor="nw")

        #Generate Button
        self.generate_button = tk.Button(self.generation_frame, text="Generate", command=self.generateClicked)
        self.generate_button.pack(side="right", fill="y", expand=False)

        self.generation_frame.pack(side="top", fill="x", expand=True, anchor="nw")
        self.generation_progress_bar.step(50)

    def generateImageViewer(self):
        self.output_frame = tk.Frame(self.root)

        self.output_image = tk.Label(self.output_frame, image=self.image)

        self.output_image.pack(side="top", fill="both", expand=True)

        self.output_frame.pack(side="top" , fill="both", expand=True)
        
    def setOnGenerate(self, callback: typing.Callable[[],None]):
        self.onGenerate = callback

    def generateClicked(self):
        print("Clicked")
        print(self.onGenerate)
        if self.onGenerate != None:
            self.onGenerate()

    def setStatus(self, text: str):
        self.status_label.config(text=f"Status: {text}")

    def setProgressIndeterminate(self):
        self.generation_progress_bar.configure(mode="indeterminate" , orient="horizontal",  maximum=100)
        self.generation_progress_bar.start(10)

    def setProgressbarStep(self, step: float):
        self.generation_progress_bar.configure(mode="determinate" , orient="horizontal",  maximum=100)
        self.generation_progress_bar.stop()
        self.generation_progress_bar.step(step)

    def getPrompt(self) -> str:
        return self.prompt_entry.get()
    
    def getNegativePrompt(self) -> str:
        return self.negative_prompt_entry.get()
    
    def getSteps(self) -> int:
        return int(self.number_of_steps_entry.get())
    
    def setImage(self, image: Image):
        self.image.paste(image)
        self.output_image.configure(image=self.image)
        
        


class ImageGenerator:
    def __init__(self, gui: MainGui):
        self.is_generating = False
        self.model_name = None
        self.model = None
        self.gui = gui
        self.generate_thread = None
        self.steps = 0
    
    def loadModel(self, model_name: str):
        self.model_name = model_name
        self.model = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
        self.model.set_progress_bar_config(leave=False)
        self.model.set_progress_bar_config(disable=True)
        self.model.requires_safety_checker = False
        self.model.safety_checker = None

    def startGeneration(self):
        if self.is_generating == False:
            self.is_generating = True
            self.generate_thread = Thread(target=self.generateImage)
            self.generate_thread.start()
            self.gui.generate_button.configure(state="disabled")

    def updateProgress(self, step: int, timestep: int, latents: torch.FloatTensor):
        self.gui.setProgressbarStep((step/self.steps)*100)
        self.gui.setStatus(f"Step {step}/{self.steps}")
        latents = 1 / 0.18215 * latents
        image = self.model.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        # convert to PIL Images
        image = self.model.numpy_to_pil(image)
        self.gui.setImage(image[0])

    def generateImage(self):
        print("Generating thread...")
        prompt = self.gui.getPrompt()
        neg_prompt = self.gui.getNegativePrompt()
        if self.model == None:
            self.gui.setStatus("Loading model")
            self.gui.setProgressIndeterminate()
            self.loadModel(DEFAULT_MODEL)
        self.gui.setStatus("Starting generation")
        self.gui.setProgressbarStep(0)
        self.steps = self.gui.getSteps()
        
        images = self.model(prompt,
            guidance_scale=6,
            num_images_per_prompt = 1,
            num_inference_steps = self.steps,
            negative_prompt=neg_prompt,
            callback=self.updateProgress,
            callback_steps=1
            ).images
        self.gui.setImage(images[0])
        timestamp = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
        images[0].save(f"Results/{timestamp}.png")
        self.gui.generate_button.configure(state="normal")

        self.gui.setStatus("Idle")
        self.gui.setProgressbarStep(100)
        self.is_generating = False



if __name__ == "__main__":
    root = tk.Tk()
    #root.resizable(False, False)
    main = MainGui(root)
    image_generator = ImageGenerator(main)
    main.setOnGenerate(image_generator.startGeneration)
    #root.wm_geometry("600x300")
    root.mainloop()

