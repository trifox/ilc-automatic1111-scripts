#
# ILC Animation Script v0.1
#
# Forked from https://github.com/Animator-Anon/Animator
#
# Must have ffmpeg installed in path.
#
# The main idea for the script was to smoothly only animate one parameter - the denoising parameter.
#
#
import os 
import time  
from scripts.ilc_mandelbrot import getILCBase64ImageSingleton, makePilILCMandelbrotLogo,ILC_LOGGER 
import modules.scripts as scripts
import gradio as gr
from modules import processing, shared, sd_samplers, images, sd_models
from modules.processing import Processed, process_images 
from modules.shared import opts, cmd_opts, state
import random
import subprocess
import numpy as np
import json
import cv2
import math 
from PIL import Image
import PIL.ImageOps    

gameAssets=[
    # {
    # "path":"C:\\PRIVAT\\godot-brickout-new\\stablediffusion-generator\\basics\\",

    #   "colorFilename":"black5612x512.png",
    #         "maskFilename":"white-512x512.png",
    #         "prompt":"A robotic leonardo da vinci"

    # },
    
    # {
    # "path":"C:\\PRIVAT\\godot-brickout-new\\stablediffusion-generator\\basics\\",

    #   "colorFilename":"black5612x512.png",
    #         "maskFilename":"white-512x512.png",
    #         "prompt":"A robot in front of a drawing cavas creating an oil painting of (mona lisa:0.1)"

    # },
    
    
    {
    "path":"C:\\PRIVAT\\godot-brickout-new\\stablediffusion-generator\\basics\\",
      "colorFilename":"black5612x512.png",
            "maskFilename":"black5612x512.png",
            "prompt":"a brick wall, atzek mayan style",
            "maskInvert":True, 
            "tileX":True,
            "tileY":True,
        "entry": [{
              "path":"C:\\PRIVAT\\godot-brickout-new\\stablediffusion-generator\\basics\\", 
            "maskFilename":"border-16px-512x512.png",
            "prompt":"a brick wall, atzek mayan style",
            "maskInvert":True,  
 "denoising_strength:":0.8,
 "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
        },{
              "path":"C:\\PRIVAT\\godot-brickout-new\\stablediffusion-generator\\basics\\", 
            "maskFilename":"border-16px-512x512.png",
            "prompt":"a brick wall, atzek mayan style",
            "maskInvert":True,  
 "denoising_strength:":0.7,
 "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
        },{
              "path":"C:\\PRIVAT\\godot-brickout-new\\stablediffusion-generator\\basics\\", 
            "maskFilename":"border-16px-512x512.png",
            "prompt":"a skull engraving in the wall",
            "maskInvert":True,  
 "denoising_strength:":0.4,
 "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
        },{
              "path":"C:\\PRIVAT\\godot-brickout-new\\stablediffusion-generator\\basics\\", 
            "maskFilename":"border-16px-512x512.png",
            "prompt":"an ancient rune engraving in the wall",
            "maskInvert":True,  
 "denoising_strength:":0.4,
 "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
        },{
              "path":"C:\\PRIVAT\\godot-brickout-new\\stablediffusion-generator\\basics\\", 
            "maskFilename":"border-16px-512x512.png",
            "prompt":"an cherry engraving in the wall",
            "maskInvert":True,  
 "denoising_strength:":0.4,
 "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
        }]},
    {
    "path":"C:\\PRIVAT\\godot-brickout-new\\stablediffusion-generator\\basics\\",
      "colorFilename":"black5612x512.png",
            "maskFilename":"black-mask-whitecenter-whiteborder-512x512.png",
            "prompt":"some a noble framer border",
            "maskInvert":True,
            "ref":"theFrame",
        "entry": [
    # an entry is a color image, a mask and a prompt, each sub entry image is initialised with the result of the entry, sub images only add masks to the created image
        {
            # each inner entries colorfilename is alpa added to the previous result 
            "maskFilename":"black-mask-whitecenter-512x512.png",
            "outFilename":"red.png",
            # "colorFilename":"diamond.png",
            "colorMode":"add",
            "maskInvert":False,
            "prompt":"a red diamond, ultra realistic, red velvet, dark red background",
           
        },

        {
            # each inner entries colorfilename is alpa added to the previous result 
            "maskFilename":"black-mask-whitecenter-512x512.png",
            "outFilename":"green.png",
            "maskInvert":False,
            "prompt":"a green diamond, super realistic, green velvet, dark green background",
        },
        {
            # each inner entries colorfilename is alpa added to the previous result 
            "maskFilename":"black-mask-whitecenter-512x512.png",
            "maskInvert":False,
            "prompt":"a blue diamond, super realistic, blue velvet, dark blue background",
            "outFilename":"blue.png",
        },
        {
            # each inner entries colorfilename is alpa added to the previous result 
            "maskFilename":"black-mask-whitecenter-512x512.png",
            "maskInvert":False,
            "prompt":"a purple diamond, super realistic, purple velvet, dark purple background",
            "outFilename":"purple.png",
        },
        {
            # each inner entries colorfilename is alpa added to the previous result 
            "maskFilename":"black-mask-whitecenter-512x512.png",
            "maskInvert":False,
            "outFilename":"yellow.png",
            "prompt":"a yellow diamond, super realistic, yellow velvet, dark yellow background",
        },
        {
            # each inner entries colorfilename is alpa added to the previous result 
            "maskFilename":"black-mask-whitecenter-512x512.png",
            "maskInvert":False,
            "outFilename":"cyan.png",
            "prompt":"a cyan diamond, super realistic, cyan velvet, dark cyan background",
        },
        {
            # each inner entries colorfilename is alpa added to the previous result 
            "maskFilename":"black-mask-whitecenter-512x512.png",
            "maskInvert":False,
            "outFilename":"bomb.png",
            "prompt":"a picture of a fusing bomb, dramatic, full shot, close up",
        },
        {
            # each inner entries colorfilename is alpa added to the previous result 
            "maskFilename":"black-mask-whitecenter-512x512.png",
            "maskInvert":False,
            "outFilename":"joker.png",
            "prompt":"a picture of a harlequin, ultra realistic, trending on artstation",
        },
        {
            # each inner entries colorfilename is alpa added to the previous result 
            "maskFilename":"black-mask-whitecenter-512x512.png",
            "maskInvert":False,
            "outFilename":"bonus.png",
            "prompt":"a picture of a pile of gold, ultra realistic, trending on artstation",
        }
            ],
         
},
{
    "path":"C:\\PRIVAT\\godot-brickout-new\\stablediffusion-generator\\basics\\",
      "maskFilename":"mask-person.png",
      "colorFilename":"black5612x512.png",
            "prompt":"""A portrait of Christian-001, normal, relaxed, equillibrium""",
            
            "ref":"thePersonNormal",
            "description":"""This is Normal Humpty Dumpty the Garlic Farmer
let's watch his story...""",
            "maskInvert":False,
             
            
            
        "entry": [
            {
            "prompt":"""A portrait of Christian-001, grief, sad, pensive""",
      "maskFilename":"mask-person.png",
            "maskInvert":False,
            "ref":"thePersonSad",
 "denoising_strength:":0.4,
 "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
            "description":"""INNER: TRANSFORMED 1 griefing""",

},
            {
            "prompt":"""A portrait of Christian-001, amazed, surprised, distracted""",
      "maskFilename":"mask-person.png",
            "maskInvert":False,
            "ref":"thePersonAmazed",
 "denoising_strength:":0.4,
 "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
            "description":"""INNER: TRANSFORMED 2 amaze""",

},
            {
            "prompt":"""A portrait of Christian-001, terror, fear, epprehension""",
      "maskFilename":"mask-person.png",
            "ref":"thePersonTerror",
            "maskInvert":False,
 "denoising_strength:":0.4,
 "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
            "description":"""INNER: TRANSFORMED 3 terror fear apprehension""",

},
            {
            "prompt":"""A portrait of Christian-001, admire trust, accept""",
      "maskFilename":"mask-person.png",
            "ref":"thePersonAdmire",
            "maskInvert":False,
 "denoising_strength:":0.4,
 "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
            "description":"""INNER: TRANSFORMED 5 admire""",

},
            {
            "prompt":"""A portrait of Christian-001, ecstatic, joy, serenity, smiling, teeth""",
      "maskFilename":"mask-person.png",
            "ref":"thePersonEcstatic",
            "maskInvert":False,
 "denoising_strength:":0.4,
 "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
            "description":"""INNER: TRANSFORMED 6 ecstatic""",

},
            {
            "prompt":"""A portrait of Christian-001, vigilant, anticipation, interest""",
      "maskFilename":"mask-person.png",
            "ref":"thePersonVigilant",
            "maskInvert":False,
 "denoising_strength:":0.4,
 "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
            "description":"""INNER: TRANSFORMED 7 vigilant""",

},
            {
            "prompt":"""A portrait of Christian-001, rage, anger, annoyed, open mouth""",
      "maskFilename":"mask-person.png",
            "maskInvert":False,
            "ref":"thePersonRage",
 "denoising_strength:":0.4,
 "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
            "description":"""INNER: TRANSFORMED 8 raging""",

},
            {
            "prompt":"""A portrait of Christian-001, loathing, disgusted, bored""",
      "maskFilename":"mask-person.png",
            "ref":"thePersonLoathing",
            "maskInvert":False,

 "denoising_strength:":0.4,
 "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
            "description":"""INNER: TRANSFORMED 9 loathing""",

},
{
    "path":"C:\\PRIVAT\\godot-brickout-new\\stablediffusion-generator\\basics\\",
      "maskFilename":"mask-person.png",
      "colorFilename":"black5612x512.png",
            "prompt":"""A portrait of Christian-001, rage, anger, annoyance""",
             
            "ref":"thePersonRaging",
            "description":"""this is raging humpty dumpty""",
            "maskInvert":False,
          "inpainting_fill":2, # 0=fill 1=original 2=latent noise 3=latent nothing
        "entry": [
    # an entry is a color image, a mask and a prompt, each sub entry image is initialised with the result of the entry, sub images only add masks to the created image
        {
            # each inner entries colorfilename is alpa added to the previous result 
    #   "maskFilename":"mask-person.png",
    #   "maskFilename":"mask-person.png", 
     
      "maskFilename":"black5612x512.png",
            "maskInvert":True,
            "prompt":"A tiny house on a green field with a tree in the foreground and mountains in the background, medieval",
            "description":"""He lives in a calm village...""", 
    #         "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
     "adds":[
                {"ref":"thePersonNormal"}
            ]
        }, 
        {
            # each inner entries colorfilename is alpa added to the previous result 
   
      "maskFilename":"black5612x512.png",
            "maskInvert":True,
            "prompt":"A path on hilly grassland terrain leading to the mountain range in the background, medieval",
            "description":"""But he has to go into the distance because
he heard of a strange temple""", "adds":[
                {"ref":"thePersonEcstatic"}
            ]
        },
           {
            # each inner entries colorfilename is alpa added to the previous result 
       "maskFilename":"black5612x512.png",
            "maskInvert":True,
            "prompt":"A path on hilly grassland terrain with a lake and trees leading to the mountain range in the background, medieval, trending on artstation",
            "description":"""The journey goes on along a lake""", "adds":[
               
                {"ref":"thePersonSad"}
            ]
        }, 
           {
            # each inner entries colorfilename is alpa added to the previous result 
       "maskFilename":"black5612x512.png",
            "maskInvert":True,
            "prompt":"A small cityscape, mountains,, medieval, trending on artstation",
            "description":"""The journey goes on along a small city""", "adds":[
                {"ref":"thePersonTerror"}
            ]
        }, 
           {
            # each inner entries colorfilename is alpa added to the previous result 
       "maskFilename":"black5612x512.png", 
            "maskInvert":True,
            "prompt":"a harbour city view, medieval, trending on artstation",
            "description":"""The journey goes on along a giant 
harbour city""", "adds":[
                {"ref":"thePersonAdmire"}
            ]
        }, 
        {
            # each inner entries colorfilename is alpa added to the previous result 
       "maskFilename":"black5612x512.png",
            "maskInvert":True,
            "prompt":"A path leading to a mayan temple on the foot of a mountain, medieval, trending on artstation",
            "description":"""Finally he arrives at the temple""", "adds":[
                {"ref":"thePersonAmazed"}
            ]
        },    {
            # each inner entries colorfilename is alpa added to the previous result 
    #   "maskFilename":"mask-person.png",
       "maskFilename":"black5612x512.png", 
            "maskInvert":True, 
            "prompt":"a wide range landscape mayan style pyramids, temples, mountain range",
            "description":"""He is delighted by the view""", "adds":[
                {"ref":"thePersonVigilant"}
            ]
        }, 
        {
            # each inner entries colorfilename is alpa added to the previous result 
       "maskFilename":"black5612x512.png",
            "maskInvert":True,
            "prompt":"a close shot of a mayan temple on the foot of a mountain, medieval, mayan, trending on artstation",
            "description":"""its a huge nice looking temple""", "adds":[
                {"ref":"thePersonRaging"}
            ]
        },    
        {
            # each inner entries colorfilename is alpa added to the previous result 
       "maskFilename":"black5612x512.png",
            "maskInvert":True,
            "prompt":"a view from a mayan temple into a mountain range, medieval, mayan, trending on artstation",
            "description":"""what a beautiful outlook it has""", "adds":[
                {"ref":"thePersonAmazed"}
            ]
        },  
        {
            # each inner entries colorfilename is alpa added to the previous result 
       "maskFilename":"black5612x512.png",
            "maskInvert":True,
            "prompt":"an entrance of an ancient temple with an altar in the middle, a banana lying on the right, stone wall, maya style, medieval, trending on artstation",
            "description":"""When he enters the temple he discovers....
.....""", "adds":[
                {"ref":"thePersonVigilant"}
            ]
        },    
        {
            # each inner entries colorfilename is alpa added to the previous result 
       "maskFilename":"black5612x512.png",
            "maskInvert":True,
            "prompt":"a sign standing in front of the temple with runes, stone wall, maya style, medieval, trending on artstation",
            "description":"""... a sign that reads
            ... you have to solve the puzzle""", "adds":[
                {"ref":"thePersonVigilant"}
            ]
        },  
    #     {
    #         # here we want to actually change the incoming character, so maskinvert is false, but we need to set denoising and fill values
    #   "maskFilename":"mask-person.png",
    #         "maskInvert":False,
    #         "denoising_strength:":0.1,
    #         "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
    #         "prompt":"A angry looking role playing character",
    #     }, 
            ],
         
}]}



]


ILC_PLUGIN_NAME="ILC-Game Art Generator"
ILC_PLUGIN_DESCRIPTION="""
A tool to create series of images following guides
"""
ILC_PLUGIN_VERSION=1.0
ILC_COMPLETE_NAME=ILC_PLUGIN_NAME+'@'+str(ILC_PLUGIN_VERSION)

LOGGER=ILC_LOGGER(ILC_PLUGIN_NAME,ILC_PLUGIN_VERSION )
from PIL import Image, ImageFilter, ImageDraw, ImageFont
 
 
class Script(scripts.Script):  
    def title(self):
        return ILC_COMPLETE_NAME

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img): 
 
        with gr.Blocks() as demo22: 
            with gr.Column(variant="panel"):
                    with gr.Row( ):
                    
                        gr.HTML(
                        """<a href="https://ilovechaos.org" target="chaos">
                            <img style="width:90%; border-radius: 1em; float:left;margin:1em" src="data:image/png;base64, """+getILCBase64ImageSingleton().decode('utf-8')+"""" alt="I-Love-Chaos Logo" />
                        </a>""")

                        gr.Markdown("""
### """+ILC_COMPLETE_NAME+"""

"""+ILC_PLUGIN_DESCRIPTION+"""

this particular plugin is derived and most code reused from
[Animator Script](https://github.com/Animator-Anon/Animator)

We follow a gui first directive here, so we reduce all possible parameters and provide sliders.

"""
                        )
                        # gr.Image(makePilILCMandelbrotLogo())
                 
          
                        globalprompt=gr.Text(value="" ,label="Global Prompt append")
        return [ globalprompt ]

    def run(self, p,   globalprompt  ): 
        usedMasks={} 
        refs={}
  
        # # output_images, info = None, None
        initial_seed = None
        initial_info = None
        initial_inpainting_fill=p.inpainting_fill
        initial_denoising_strength=p.denoising_strength
        processed=None
        imageIndex=0


        def renderText(image,text,position=(100,100)):
            draw=ImageDraw.Draw(image)
            font = ImageFont.truetype(r'arial.ttf', 25) 
            spacing = 28 
            print("Rendering text",text)
            pad=5

            txtsize = draw.multiline_textbbox(position, text, spacing = spacing,  font=font)
            draw.rounded_rectangle((txtsize[0] - pad,
                                txtsize[1] - pad,
                                txtsize[2] + 2 * pad,
                                txtsize[3] + 2 * pad), radius=pad, fill=(244,243,242))

            # drawing text size
            draw.text(position, text, fill ="black", font = font,  spacing = spacing, align ="right",stroke_width =1) 
            draw.text((position[0]+1,position[1]+1), text, fill ="red", font = font,  spacing = spacing, align ="right",stroke_width =1) 
            
        def countEntry(entry ):
            result=1
            if("entry" in entry):
                result+=countEntries(entry["entry"])
            return result
        def countEntries(entry ):
            result=0
            for centry in entry:
                result=result+countEntry(centry)

            return result
        def handleAdd(addd,currentImage):
            if(addd['ref'] in refs):
                currentImage.paste(refs[addd['ref']]['image'],(0,00),refs[addd['ref']]['mask'])
            else:
                print("ERROR: Ref not found",addd['ref'])
            return
        def handleEntry(entry,currentImage=None,path=None,depth=0):
            nonlocal initial_seed,initial_info,processed,resultPicture,imageIndex,refs
            imageIndex=imageIndex+1 
            print("Handling entry",entry)
            if(path==None):
                path=entry.get("path","")

            colorImage=None
            if("maskInvert" in entry):
                p.inpainting_mask_invert =entry["maskInvert"]
            else:
                p.inpainting_mask_invert=False

            if("inpainting_fill" in entry):
                p.inpainting_fill=entry["inpainting_fill"]
            else:
                p.inpainting_fill=initial_inpainting_fill

            if("denoising_strength" in entry):
                p.denoising_strength=entry["denoising_strength"]
            else:
                p.denoising_strength=initial_denoising_strength

            if("colorFilename" in entry):
                # either currentImage is present, then add colorFilename of entry to current
                # or the currentImage is not present, then load the colorFilename as full currentImage

                filename=entry["colorFilename"]
                if("@" in filename):
                    print("Using ref Image",filename[1:])
                    colorImage=refs[filename[1:]]['image']                   
                else:

                    if(currentImage==None):
                        colorImage =Image.open(os.path.join(path, entry["colorFilename"]))
                    else: 
                        newImage = Image.open(os.path.join(path, entry["colorFilename"])) 
                        colorImage = Image.alpha_composite(currentImage.convert("RGBA"), newImage.convert("RGBA"))

                    if(entry["colorFilename"] in usedMasks):
                        all_images_collection.append([ colorImage])
                        usedMasks[entry["colorFilename"]]=colorImage
            else:
                colorImage=currentImage

         

            maskImage=Image.open(os.path.join(path, entry["maskFilename"]))
            if(p.inpainting_mask_invert):
               if(not entry["maskFilename"]+"invert" in usedMasks):
                    all_images_collection.append([ PIL.ImageOps.invert( maskImage.convert('RGB'))])
                    usedMasks[entry["maskFilename"]+"invert"]=maskImage
            else:
               if(not entry["maskFilename"] in usedMasks):
                all_images_collection.append([ maskImage])
                usedMasks[entry["maskFilename"]]=maskImage
            p.prompt=entry["prompt"]+' '+globalprompt

            p.image_mask=maskImage
            p.init_images = [colorImage]
            p.extra_generation_params["entry-{imageIndex}-prompt"]=p.prompt
            
            if("seed" in entry):
                p.seed=entry["seed"]
            else:
                p.seed=initial_seed
            print("Rendering entry",p.image_mask,p.init_images)

            processed = processing.process_images(p)

            if state.interrupted:
                # Interrupt button pressed in WebUI
                return 

            post_processed_image = processed.images[0].copy()
            
            if("adds" in entry):
                    for addd in entry["adds"] :
                        handleAdd(addd,post_processed_image)


            if("ref" in entry):
                refs[entry["ref"]]={
                    "image":post_processed_image.copy(),
                    "mask":maskImage.copy(),
                }
               
            
            rowGap=128
            xPos=((imageIndex-1)%4)*512
            yPos=((imageIndex-1)//4)*(512+rowGap)
            resultPicture.paste(post_processed_image,(               xPos,yPos                )                )
            
            if("description" in entry):
                renderText(resultPicture,entry['description'],(xPos,yPos+512))

            if initial_seed is None:
                initial_seed = processed.seed
                initial_info = processed.info  

            for imga in processed.images:  
                        # imga.save(os.path.join(outpath, f"{index}-{outfilename}_{frame_no:05}.png"))                        
                        all_images_collection.append([imga,post_processed_image]) 

            if("entry" in entry):
                for childEntry in entry["entry"]:
                    handleEntry(childEntry,currentImage=post_processed_image,path=path,depth=depth+1) 
            return post_processed_image

        initial_seed_save=p.seed
         
        all_images_collection = [] 
        
        resultPicture=Image.new("RGB", 
        (
            512*4,
            (128+512)*((countEntries(gameAssets)//4) ),            
        )
        )

        # for step in range(0,p.n_iter): 
        # Fix variable types, i.e. text boxes giving strings.
        p.seed=initial_seed_save  
   

        processing.fix_seed(p)
        batch_count = countEntries(gameAssets)

        # Save extra parameters for the UI
        p.extra_generation_params = {
            # "Config Path": gameAssets["path"], 
            # "Config File": gameAssets["path"], 
        }

        p.batch_size = 1 
        state.job_count = 1* batch_count
        # for asset in gameAssets:
        #     handleEntry(asset)
        handleEntry(gameAssets[0])
        # handleEntry(gameAssets[2]) 
 
        # # Process current frame
        # processed = processing.process_images(p)

        # index=0
        # for imga in processed.images: 
        #     index=index+1
        #     # imga.save(os.path.join(outpath, f"{index}-{outfilename}_{frame_no:05}.png"))
            
        #     all_images_collection.append([])  
        #     all_images_collection[index-1].append(imga)

  
        dickies=[resultPicture]
        for images in all_images_collection:
            for image in images:
                dickies.append(image)

        processed = Processed(p, dickies, initial_seed, initial_info)
             

        return processed
