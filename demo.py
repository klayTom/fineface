import torch
from fineface import FineFacePipeline

pipe = FineFacePipeline()

torch.manual_seed(2)

prompt = "Trump delivering a speech while wearing a Superman costume, laughing confidently at the crowd"
# Set AUs
# Possible AUs: [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
aus = {"AU1": 2.0, "AU2": 2.0, "AU6": 3.0, "AU12": 5.0, "AU25": 2.0}

image = pipe(prompt, aus).images[0]

image.save(f"results/{prompt} {str(aus)}.jpg")