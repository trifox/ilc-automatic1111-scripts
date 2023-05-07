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
from scripts.ilc_mandelbrot import getILCBase64ImageSingleton,makePilILCMandelbrot, makePilILCMandelbrotLogo,ILC_LOGGER 
from scripts.asymmetric_tiling import Script as AsymmetricTiling
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
from PIL import Image, ImageOps,ImageMath

import PIL.ImageOps    
NEGATIVE_PROMPTS='''bad anatomy, bad proportions, blurry cloned face, cropped, deformed, dehydrated, disfigured, duplicate, error, gross proportions, jpeg artifacts, long neck, low quality, lowres, morbid
mutated hands, mutation, mutilated,out of frame, text, ugly, username, watermark, worst quality, horizontal lines, linearity'''
general='majestic, symbolic, intricate, mysterious, colorful, iconic, visually-striking, evocative, enigmatic, expressive, dynamic, surreal, fantastical, ancient, mystical, artistic.'
tarotCardSet=[  
     
     {
       "path":"C:\\PRIVAT\\godot-brickout-new\\stablediffusion-generator\\basics\\",
    #   "colorFilename":"white-512x768.png",
    "bgColorImage":"#ffffff",
    
            "maskFilename":"black-mask-whitecenter-512x768.png",
           "prompt": "An ornamental card frame with golden silver ornaments and flowers",
"description": "The Card Frame",
 
            "maskInvert":True, 
            # "tileX":False,
  "inpainting_fill":2, # 0=fill 1=original 2=latent noise 3=latent nothing
            # "tileY":False,
        "entry": [
                { 
            "maskFilename":"black-mask-whitelabel-round-512x768.png",
          "prompt": "A single cup",
"description": "The cup Card Class Symbol",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":2, # 0=fill 1=original 2=latent noise 3=latent nothing
  "entry":[
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A single large cup, adorned with intricate symbols and engravings. The cup appears radiant and gleaming.",
"description": "Ace of Cups: Emotional Renewal and Divine Love\nOutlook: Overflowing Emotions and Inner Harmony",
          "negative_prompt": "4 four 2 two 3 three 5 five 6 six 7 seven 8 eight 9 nine 10 ten",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "Two figures exchanging cups, representing the union of souls, emotional connection, and the potential for harmonious and loving partnerships.",
"description": "Two of Cups: Connection, partnership, harmony\nOutlook: Mutual support, balanced relationships, love blossoming",
          "negative_prompt": "4 four 1 one 3 three 5 five 6 six 7 seven 8 eight 9 nine 10 ten",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "Three figures raising their cups in celebration, symbolizing the joy and camaraderie that come from connections with others and the celebration of life's blessings.",
"description": "Three of Cups: Celebration, friendship, joy\nOutlook: Shared happiness, social gatherings, creative collaborations",
          "negative_prompt": "4 four 2 two 1 one 5 five 6 six 7 seven 8 eight 9 nine 10 ten",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }}
    ,
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A figure sitting and contemplating three cups while a fourth cup is being offered, signifying the need for introspection and a reassessment of one's emotional fulfillment.",
"description": "Four of Cups: Contemplation, introspection, reevaluation\nOutlook: Soul-searching, seeking inner fulfillment, new perspectives",
          "negative_prompt": "3 three 2 two 1 one 5 five 6 six 7 seven 8 eight 9 nine 10 ten",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }}
    
    
    ,
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A figure standing with their back turned to three spilled cups while two cups remain upright, representing the experience of loss and the opportunity for emotional healing and finding solace.",
"description": "Five of Cups: Loss, disappointment, grief\nOutlook: Acceptance, healing, finding silver linings",
          "negative_prompt": "3 three 2 two 1 one 4 four 6 six 7 seven 8 eight 9 nine 10 ten",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }}
    
    ,
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "Two figures, one older and one younger, exchanging cups filled with flowers, symbolizing the sweetness of nostalgia, fond memories, and the joy of reconnecting with the innocence of the past.",
"description": "Six of Cups: Nostalgia, childhood memories, innocence\nOutlook: Inner child, emotional connection, finding comfort in the past",
          "negative_prompt": "3 three 2 two 1 one 4 four 5 five 7 seven 8 eight 9 nine 10 ten",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }}
    
    
    ,
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A figure standing before seven cups containing various images, representing the need for discernment and choosing wisely among numerous options and potential illusions.",
"description": "Seven of Cups: Choices, illusions, possibilities\nOutlook: Clarity, discernment, focusing on priorities",
          "negative_prompt": "3 three 2 two 1 one 4 four 5 five 6 six 8 eight 9 nine 10 ten",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }}
    
    
    ,
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A figure walking away from eight stacked cups, symbolizing the journey of self-discovery, leaving behind emotional attachments, and embarking on a quest for deeper fulfillment.",
"description": "Eight of Cups: Departure, seeking deeper meaning, moving on\nOutlook: Inner exploration, spiritual growth, finding one's true path",
          "negative_prompt": "3 three 2 two 1 one 4 four 5 five 6 six 7 seven 9 nine 10 ten",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }}
    ,
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A figure sitting with arms crossed, surrounded by nine cups in a semicircle, symbolizing emotional fulfillment, contentment, and the realization of wishes and desires.",
"description": "Nine of Cups: Contentment, emotional satisfaction, wishes fulfilled\nOutlook: Abundance, gratitude, emotional well-being",
          "negative_prompt": "3 three 2 two 1 one 4 four 5 five 6 six 7 seven 8 eight 10 ten",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }}
    ,
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A joyful family gathered under a rainbow of ten cups, representing the ultimate emotional fulfillment, harmonious relationships, and the joy that comes from deep connections.",
"description": "Ten of Cups: Harmony, fulfillment, emotional bliss\nOutlook: Family joy, deep connections, domestic happiness",
          "negative_prompt": "3 three 2 two 1 one 4 four 5 five 6 six 7 seven 8 eight 9 nine",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }}
    
    ,
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A young person holding a cup, dressed in colorful attire, symbolizing the emergence of creative potential, emotional sensitivity, and the arrival of new experiences.",
"description": "Page of Cups: Creativity, intuitive messages, new beginnings\nOutlook: Curiosity, emotional exploration, embracing imaginative pursuits",
          
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }}
    ,
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A mounted knight holding a cup, wearing armor adorned with symbols of emotions, representing the pursuit of dreams, romantic endeavors, and the willingness to take risks for emotional fulfillment.",
"description": "Knight of Cups: Romance, charm, pursuing dreams\nOutlook: Passionate pursuit, emotional adventures, following the heart's desires",
          
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }}
    
    ,
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A queen seated with a cup, radiating warmth and serenity, symbolizing emotional wisdom, intuitive abilities, and the capacity to offer support and compassion to others.",
"description": "Queen of Cups: Compassion, emotional depth, intuition\nOutlook: Nurturing, empathy, emotional maturity",
          
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }}
    
    ,
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A king seated on a throne, holding a cup, representing emotional stability, wisdom, and the ability to navigate complex emotional situations with grace and understanding.",
"description": "King of Cups: Emotional balance, leadership, wisdom\nOutlook: Emotional intelligence, diplomacy, mastery of emotions",
          
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }}
    
    
    
    ]
    },
                { 
            "maskFilename":"black-mask-whitelabel-round-512x768.png",
      "prompt": "a single upright wand or staff emerging from a lush and fertile landscape with ripe fruit and flowing water, a vibrant and dynamic scene, a sturdy and flourishing branch, budding leaves or blooming flowers, lightning bolt or a burst of light,",
"description": "Wands Symbol",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":2, # 0=fill 1=original 2=latent noise 3=latent nothing
  "entry":[
        
           


        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "a single upright wand or staff emerging from a lush and fertile landscape with ripe fruit and flowing water, a vibrant and dynamic scene, a sturdy and flourishing branch, budding leaves or blooming flowers, lightning bolt or a burst of light",
"description": "Ace of Wands: Creative Potential and New Beginnings\nOutlook: Inspiration and Passion",
          "negative_prompt": "4 four 2 two 3 three 5 five 6 six 7 seven 8 eight 9 nine 10 ten",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A figure holding a globe and a wand, looking out into the horizon, representing the need to plan and make decisions for future growth and success.",
"description": "Two of Wands: Planning, personal power, future vision\nOutlook: Expansion, making choices, stepping into leadership roles",
          "negative_prompt": "4 four 1 one 3 three 5 five 6 six 7 seven 8 eight 9 nine 10 ten",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A figure standing on a cliff, looking out over the sea with three wands, symbolizing the anticipation of progress and the rewards of venturing beyond familiar",
"description": "Three of Wands: Progress, enterprise, expansion\nOutlook: Collaboration, exploring new horizons, reaping rewards",
          "negative_prompt": "4 four 1 one 2 two 5 five 6 six 7 seven 8 eight 9 nine 10 ten",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }}
    ,
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "Four wands forming a canopy with people celebrating underneath, representing the joy and harmony that comes from achieving a significant milestone or finding a sense of home.",
"description": "Four of Wands: Celebration, harmony, homecoming\nOutlook: Stability, joyful occasions, a sense of belonging",
          "negative_prompt": "3 three 1 one 2 two 5 five 6 six 7 seven 8 eight 9 nine 10 ten",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }} ,
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "Five figures holding wands in a dynamic and chaotic scene, signifying the need to work together, find common ground, and rise above conflicts and challenges.",
"description": "Five of Wands: Conflict, competition, challenges\nOutlook: Collaboration, finding common ground, overcoming obstacles",
          "negative_prompt": "3 three 1 one 2 two 4 four 6 six 7 seven 8 eight 9 nine 10 ten",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }} ,
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A figure riding a horse with a victory wreath, surrounded by onlookers holding wands, symbolizing the triumph and recognition that come from achieving a significant goal.",
"description": "Six of Wands: Victory, recognition, progress\nOutlook: Confidence, public acclaim, success",
          "negative_prompt": "3 three 1 one 2 two 4 four 5 five 7 seven 8 eight 9 nine 10 ten",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A figure standing on a hill, defending against six wands, symbolizing the need to stand firm, maintain beliefs, and face challenges with courage.",
"description": "Seven of Wands: Perseverance, standing tall, defending beliefs\nOutlook: Assertiveness, resilience, overcoming opposition",
          "negative_prompt": "3 three 1 one 2 two 4 four 5 five 6 six 8 eight 9 nine 10 ten",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "Eight wands in flight, representing the swift movement and the abundance of opportunities that bring accelerated progress and expansion.",
"description": "Eight of Wands: Swiftness, action, progress\nOutlook: Forward momentum, opportunities, rapid growth",
          "negative_prompt": "3 three 1 one 2 two 4 four 5 five 6 six 7 seven 9 nine 10 ten",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }}
    ,
    
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A figure standing with a bandaged head, holding a wand, surrounded by eight upright wands, symbolizing the resilience and determination to overcome challenges and reach the final stages of a project or goal.",
"description": "Nine of Wands: Resilience, perseverance, determination\nOutlook: Courage, strength, nearing completion",
          "negative_prompt": "3 three 1 one 2 two 4 four 5 five 6 six 7 seven 8 eight 10 ten",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }}
    ,
    
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A figure carrying a heavy bundle of wands, representing the weight of responsibilities and the need to find balance and release burdens after completing a demanding task.",
"description": "Ten of Wands: Burden, responsibility, hard work\nOutlook: Completion, releasing burdens, seeking balance",
          "negative_prompt": "3 three 1 one 2 two 4 four 5 five 6 six 7 seven 9 nine 8 eight",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }}
    
    ,
    
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A young person holding a wand, dressed in vibrant attire, symbolizing the emergence of creative potential, a spirit of adventure, and the arrival of exciting new opportunities.",
"description": "Page of Wands: Creative inspiration, exploration, new opportunities\nOutlook: Enthusiasm, curiosity, embracing new endeavors",
          
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }}
    
    ,
    
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A mounted knight holding a wand, dressed in armor and ready for action, representing the pursuit of passions, adventurous spirit, and the drive to achieve one's goals.",
"description": "Knight of Wands: Action, passion, adventure\nOutlook: Ambition, forward momentum, pursuing goals with determination",
          
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }}
    
    ,
    
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A queen seated with a wand, exuding confidence and grace, symbolizing leadership qualities, charisma, and the ability to inspire and motivate others.",
"description": "Queen of Wands: Confidence, leadership, charisma\nOutlook: Radiance, influence, harnessing personal power",
          
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }}
        
        ,
        
            { 
                "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
            "prompt": "A king seated on a throne, holding a wand, projecting strength and determination, representing authority, vision, and the ability to manifest success through strategic planning and decisive action.",
    "description": "King of Wands: Authority, vision, success\nOutlook: Command, strategic thinking, harnessing personal power",
            
                "maskInvert":False,  
    #  "denoising_strength":0.6, 
    "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
    "mandelbrotGenerate":{
            
        }}
        
    
    
    ]
    }
        ,
              { 
            "maskFilename":"black-mask-whitelabel-round-512x768.png",
          "prompt": "a single upright sword rising triumphantly from a clear and open sky",
"description": "The sword Card Class Symbol",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":2, # 0=fill 1=original 2=latent noise 3=latent nothing
  "entry":[
    
    
    
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A single large sword, adorned with intricate symbols and engravings. The sword appears radiant and gleaming.",
"description": "Ace of Swords: Mental Clarity and New Perspectives\nOutlook: Truth and Power",
          "negative_prompt": "4 four 2 two 3 three 5 five 6 six 7 seven 8 eight 9 nine 10 ten",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A figure seated with two crossed swords, blindfolded, representing the need to make a choice and find balance amidst opposing forces.",
"description": "Two of Swords\nDecision-making, balance, duality\nOutlook: Equilibrium, weighing options, finding middle ground",
          "negative_prompt": "4 four 1 one 3 three 5 five 6 six 7 seven 8 eight 9 nine 10 ten",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "Three swords piercing a heart, symbolizing emotional distress and the need to confront and heal emotional wounds.",
"description": "Three of Swords: Heartbreak, sorrow, emotional pain\nOutlook: Healing, release, finding solace",
          "negative_prompt": "4 four 1 one 2 two 5 five 6 six 7 seven 8 eight 9 nine 10 ten",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
    
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A figure lying on a tomb with four swords above, signifying the need for rest and rejuvenation to restore mental and physical well-being.",
"description": "Four of Swords: Rest, retreat, recuperation\nOutlook: Restoring energy, introspection, rejuvenation",
          "negative_prompt": "3 three 1 one 2 two 5 five 6 six 7 seven 8 eight 9 nine 10 ten",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A figure holding three swords while two others walk away, representing the aftermath of a conflict and the need for reflection and forgiveness.",
"description": " Five of Swords: Conflict, competition, betrayal\nOutlook: Learning from defeat, choosing battles wisely, forgiveness",
          "negative_prompt": "3 three 1 one 2 two 4 four 6 six 7 seven 8 eight 9 nine 10 ten",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
    
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A figure in a boat being ferried across water, symbolizing a journey away from difficulties towards a calmer and more peaceful future.",
"description": "Six of Swords\nTransition, moving on, finding peace\nOutlook: Healing, progress, leaving the past behind",
          "negative_prompt": "3 three 1 one 2 two 4 four 5 five 7 seven 8 eight 9 nine 10 ten",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
    
    
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A figure carrying five swords and sneaking away, implying the need for carefulness and avoiding deceitful situations.",
"description": "Seven of Swords: Deception, trickery, dishonesty\nOutlook: Caution, vigilance, protecting oneself",
          "negative_prompt": "3 three 1 one 2 two 4 four 6 six 5 five 8 eight 9 nine 10 ten",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
    
    
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A figure blindfolded and bound with eight swords, representing self-imposed restrictions and the need to overcome limitations.",
"description": "Eight of Swords: Restriction, feeling trapped, self-imposed limitations\nOutlook: Liberation, breaking free, self-empowerment",
          "negative_prompt": "3 three 1 one 2 two 4 four 6 six 5 five 7 seven 9 nine 10 ten",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
    
    
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A figure sitting up in bed with hands on the face, surrounded by nine swords, symbolizing nighttime worries and the importance of finding solace.",
"description": "Nine of Swords: Anxiety, worry, fear\nOutlook: Seeking support, finding peace of mind, releasing worries",
          "negative_prompt": "3 three 1 one 2 two 4 four 6 six 5 five 7 seven 8 eight 10 ten",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
    
    
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A figure lying face down with ten swords in the back, signifying the end of a difficult cycle and the opportunity for rebirth and renewal.",
"description": "Ten of Swords: Defeat, rock bottom, crisis\nOutlook: Embracing change, transformation, new beginnings",
          "negative_prompt": "3 three 1 one 2 two 4 four 6 six 5 five 7 seven 9 nine 8 eight",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
    
    
    
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A young person holding a sword, alert and ready for action, symbolizing curiosity, mental acuity, and the arrival of new ideas and perspectives.",
"description": "Page of Swords: Curiosity, mental agility, new ideas\nOutlook: Inquisitiveness, seeking knowledge, embracing intellectual growth",
          
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
      { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A mounted knight holding a sword, dressed in armor and charging forward, representing ambition, assertiveness, and the drive to overcome obstacles through strategic thinking and decisive action.",
"description": "Knight of Swords: Ambition, assertiveness, strategic thinking\nOutlook: Determination, focused action, pursuing intellectual challenges",
          
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A queen seated with a sword, radiating poise and intellect, symbolizing clarity of thought, wisdom, and the ability to make rational decisions while maintaining personal boundaries.",
"description": "Queen of Swords: Clarity, wisdom, intellectual strength\nOutlook: Discernment, independence, maintaining boundaries",
          
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
    
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A king seated on a throne, holding a sword, projecting strength and wisdom, representing authority, logical thinking, and the ability to wield intellectual mastery in leadership roles.",
"description": "King of Swords: Authority, logic, intellectual mastery\nOutlook: Leadership, strategic planning, harnessing mental power",
          
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
    
      
    
    
    
    
    ]
    },
               { 
            "maskFilename":"black-mask-whitelabel-round-512x768.png",
          "prompt": "A single gold coin symbol icon ",
"description": "The Coin Card Class Symbol",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":2, # 0=fill 1=original 2=latent noise 3=latent nothing
  "entry":[
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A single large coin, adorned with intricate symbols and engravings. The coin appears radiant and gleaming.",
"description": "Ace of Coins: New Beginnings\nOutlook: Positive",
          "negative_prompt": "4 four 2 two 3 three 5 five 6 six 7 seven 8 eight 9 nine 10 ten",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A figure juggling two coins, representing the need for balance and adaptability. The individual may be standing on uneven ground. The coins are in constant motion.",
"description": "Name: 2 Coins  Balance and Adaptability\n Outlook: Flexible",
          "negative_prompt": "1 one 4 four 3 three 5 five 6 six 7 seven 8 eight 9 nine 10 ten",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "three individuals working together on a project. They are often depicted in a workshop or construction site.",
"description": "Three of Coins: Collaboration and Mastery\nOutlook: Acknowledgment",
          "negative_prompt": "1 one 2 two 4 four  5 five 6 six 7 seven 8 eight 9 nine 10 ten",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A figure seated, tightly clutching four coins or pentacles. The person is closed offe. A fortified environment.",
          "negative_prompt": "1 one 2 two 3 three 5 five 6 six 7 seven 8 eight 9 nine 10 ten",
"description": "Four of Coins: Stability and Security\nOutlook: Resistance to Change",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A figure seated, tightly clutching four coins or pentacles. The person is closed offe. A fortified environment.",
          "negative_prompt": "1 one 2 two 3 three 5 five 6 six 7 seven 8 eight 9 nine 10 ten",
"description": "Five of Coins: Financial Hardship and Seeking Support\nOutlook: Temporary Setback",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
       
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "a figure, usually a well-dressed individual, offering coins to those in need. The visual shows a scale.",
          "negative_prompt": "1 one 2 two 3 three 4 four 5 five 7 seven 8 eight 9 nine 10 ten",
"description": "Six of Coins: Generosity and Receiving Assistance\nOutlook: Mutual Support",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "a person leaning on a tool gazing to a vineyard, carefully cultivating plants. The plants are in various stages of growth.",
          "negative_prompt": "1 one 2 two 3 three 4 four 5 five 6 six 8 eight 9 nine 10 ten",
"description": "Seven of Coins: Patience and Investment\nOutlook: Delayed Rewards",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "An individual diligently working on a coin. A quiet study room. Intricate details of the coin being crafted.",
          "negative_prompt": "1 one 2 two 3 three 4 four 5 five 6 six 7 seven eight 9 nine 10 ten",
"description": "Eight of Coins: Craftsmanship and Skill Development\nOutlook: Skill Enhancement",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "a figure, o elegantly dressed, standing in a lavish garden. The person exudes an air of confidence and self-sufficiency. The visual showcase abundant harvest or ripe fruits.",
          "negative_prompt": "1 one 2 two 3 three 4 four 5 five 6 six 7 seven eight  8 eight  10 ten",
"description": "Nine of Coins: Self-Sufficiency and Luxury\nAbundance and Independence",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "a gathering of generations.  The visual depicts a family crest, a grand estate and a joyful celebration.",
          "negative_prompt": "1 one 2 two 3 three 4 four 5 five 6 six 7 seven eight  8 eight  9 nine",
"description": "Ten of Coins: Wealth and Ancestral Blessings\n Prosperity and Legacy",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A young person holding a pentacle, dressed in earthy attire, symbolizing the willingness to explore practical matters, learn new skills, and seize opportunities for material and financial growth.",
        
"description": "Page of Pentacles: Exploration, practicality, new opportunities\nOutlook: Eagerness to learn, growth in skills, embracing new ventures",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A mounted knight holding a pentacle, dressed in sturdy armor, representing a strong work ethic, responsibility, and the determination to achieve stability and long-term success through diligent efforts.",
        
"description": "Knight of Pentacles: Hard work, responsibility, stability\nOutlook: Dedication, patience, perseverance in pursuit of goals",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A queen seated with a pentacle, radiating warmth and stability, symbolizing nurturing qualities, practical wisdom, and the ability to create abundance in both material and emotional realms.",
        
"description": "Queen of Pentacles: Nurturing, abundance, practicality\nOutlook: Groundedness, generosity, creating a harmonious home",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
        { 
            "maskFilename":"black-mask-whitelabel-round-content-512x768.png",
          "prompt": "A king seated on a throne, holding a pentacle, projecting a sense of prosperity and authority, representing practical mastery, the ability to manage resources, and the potential for financial success.",
        
"description": "King of Pentacles: Prosperity, wealth, practical mastery\nOutlook: Financial security, leadership, utilizing resources wisely",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    }},
  ]
        },
      
             { 
            "maskFilename":"black-mask-whitecenter-512x768.png",
          "prompt": "Vibrant jester-like figure, standing amidst a colorful landscape. Wearing a motley outfit, holding a jingling staff, and gazing mischievously at the viewer.",
"description": "0: The Fool: New Beginnings and Spontaneity\nOutlook: Optimism and Adventure",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
   "mandelbrotGenerate":{
         
    },
        }
        ,
        { 
            "maskFilename":"black-mask-whitecenter-512x768.png",
            "prompt": "A central man, shrouded in a flowing robe, stands amidst a mystical landscape. They hold a staff, emanating vibrant energy, while their eyes gleam with ancient wisdom. Surrounding them, ethereal creatures dance in harmonious patterns.",
"description": "I: The Magician: Manifestation and Mastery\nOutlook: Empowerment and Action",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
  "mandelbrotGenerate":{
         
    },
        },
        { 
            "maskFilename":"black-mask-whitecenter-512x768.png",
          "prompt": "Majestic figure, adorned in flowing robes, holds a sacred scepter. Crowned with intricate headdress, surrounded by ethereal aura.",
"description": "II: The High Priestress: Intuition and Hidden Knowledge\nOutlook: Inner Wisdom and Spiritual Guidance",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing 
  "mandelbrotGenerate":{
         
    },
        }
        ,
        { 
            "maskFilename":"black-mask-whitecenter-512x768.png",
           "prompt": "Elegant woman seated on a golden throne, wearing a crown and holding a scepter, surrounded by lush vegetation and a serene blue sky.",
"description": "III: The Empress: Nurturing and Abundance\nOutlook: Fertility and Creativity",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
  "mandelbrotGenerate":{
         
    },
        }
        ,
        { 
            "maskFilename":"black-mask-whitecenter-512x768.png",
          "prompt": "Regal man sitting on a grand throne, adorned with a golden crown. Robes flowing with intricate patterns, symbolizing authority. Serene expression, holding a scepter and an orb, signifying power and dominion.",
"description": "IV: The Emperor: Authority and Structure\nOutlook: Leadership and Stability",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
  "mandelbrotGenerate":{
         
    },
        }
        ,
        { 
            "maskFilename":"black-mask-whitecenter-512x768.png",
        "prompt": "A man sits on a throne between two pillars, holding a staff in one hand and a raised hand gesture in the other. He wears a ceremonial robe adorned with crosses and other symbols. Two acolytes kneel before him, and a symbolic key and scroll lay at his feet. The background is a yellowish-brown hue with a faint pattern.",
"description": "V: The Hierophant: Tradition and Spiritual Guidance\nOutlook: Conformity and Education",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
  "mandelbrotGenerate":{
         
    },
        }
        ,
        { 
            "maskFilename":"black-mask-whitecenter-512x768.png",
        "prompt": "A Man and a Woman entwined in a passionate embrace, surrounded by lush vegetation, adorned with blooming flowers.",
"description": "VI: The Lovers: Union and Harmony\n Outlook: Connection and Choice",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
  "mandelbrotGenerate":{
         
    },
        }
        ,
        { 
            "maskFilename":"black-mask-whitecenter-512x768.png",
       "prompt": "A golden majestic chariot, adorned with stars the chariot is pulled by two horses of contrasting colors.",
"description": "VII: The Chariot: Willpower and Triumph\nOutlook: Determination and Victory",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
  "mandelbrotGenerate":{
         
    },
        }
        ,
        { 
            "maskFilename":"black-mask-whitecenter-512x768.png",
      "prompt": "A regal woman sits on a stone throne, draped in a flowing robe. They hold a balanced scale in one hand and a double-edged sword in the other. The background displays pillars and a serene blue sky, representing order and impartiality.",
"description": "VIII: The Justice: Fairness and Balance\nOutlook: Accountability and Truth",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
  "mandelbrotGenerate":{
         
    },
        }
        ,
        { 
            "maskFilename":"black-mask-whitecenter-512x768.png",
     "prompt":"A solitary man sits in contemplation, surrounded by a serene landscape. A cloak envelops them, accentuating their seclusion. A lantern's warm glow illuminates their path, while a staff rests by their side.",
"description":"IX: The Hermit: Solitude and Inner Guidance\nOutlook: Reflection and Wisdom",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
  "mandelbrotGenerate":{
         
    },
        }
        
        ,
        { 
            "maskFilename":"black-mask-whitecenter-512x768.png",
    "prompt": "A circular wheel divided into four equal parts, each quadrant representing a distinct season. The outer rim is adorned with various symbols and images depicting nature's cycles, intertwined with celestial motifs. In the center, a hub with intricate gears and spokes symbolizes the passage of time.",
"description": "X: The Wheel of Fortune: Destiny and Change\nOutlook: Cycles and Opportunities",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
  "mandelbrotGenerate":{
         
    },
        }
        
        ,
        { 
            "maskFilename":"black-mask-whitecenter-512x768.png",
   "prompt":"A towering figure adorned in armor and a regal cloak, wielding a mighty sword, exuding strength and authority.",
"description":"XI: The Strength:Inner Power and Courage\nOutlook: Resilience and Triumph",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
  "mandelbrotGenerate":{
         
    },
        }
        
        ,
        { 
            "maskFilename":"black-mask-whitecenter-512x768.png",
      "prompt": "A man hanging upside down from a tree. A body of water and mountains in the background.",
"description": "XII: The Hanged Man: Surrender and Perspective\nOutlook: Release and Enlightenment",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
  "mandelbrotGenerate":{
         
    },
        }
        
        
        ,
        { 
            "maskFilename":"black-mask-whitecenter-512x768.png",
     "prompt": "A skeletal figure, black robes and armor, holding a flag with a white rose on a black background. riding a white horse, a rising sun in the background.",
"description": "XIII: Death: Transformation and Renewal\nOutlook: Endings and New Beginnings",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
  "mandelbrotGenerate":{
         
    },
        }
        
        
        ,
        { 
            "maskFilename":"black-mask-whitecenter-512x768.png",
   "prompt": "An angelic figure standing with one foot on land and the other in a body of water. The angel pours water from one cup to another. The sun rises behind the angel, with mountains in the background and flowers blooming in the foreground.",
"description": "XIV: Temperance: Balance and Harmony\nOutlook: Moderation and Integration",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
  "mandelbrotGenerate":{
         
    },
        }
        ,
        { 
            "maskFilename":"black-mask-whitecenter-512x768.png",
   "prompt": "A horned, goat-like creature with a male and female figure chained to his throne-like seat. The devil has a bat-like wings and holds a pentagram in his hand. The man and woman are naked.",
"description": "XV: The Devil: Temptation and Shadow Self\nOutlook: Bondage and Liberation",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
  "mandelbrotGenerate":{
         
    },
        }
        
        
        ,
        { 
            "maskFilename":"black-mask-whitecenter-512x768.png",
  "prompt": "A tall stone tower being struck by lightning, causing flames and debris to burst out from the top. The sky is dark and stormy, with bolts of lightning and black clouds.",
"description": "XVI: The Tower: Sudden Change and Revelation\nOutlook: Destruction and Rebirth",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
  "mandelbrotGenerate":{
         
    },
        }
        
        
        ,
        { 
            "maskFilename":"black-mask-whitecenter-512x768.png",
"prompt":"A Star shines with a radiant celestial figure, surrounded by a vast midnight-blue sky. Her graceful, outstretched arms hold two water vessels. Glittering stars twinkle in the background, casting their soft glow upon the landscape",
"description":"XVII: The Star: Hope and Inspiration\nOutlook: Healing and Renewal",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
  "mandelbrotGenerate":{
         
    },
        }
        
        ,
        { 
            "maskFilename":"black-mask-whitecenter-512x768.png",
            "prompt":"A Moon scene on a pond, a dog and wolf howl at the moon in the background.",
"description":"XVIII: The Moon: Intuition and Mystery\nOutlook: Inner Reflection and Illusion",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
  "mandelbrotGenerate":{
         
    },
        }
        
        ,
        { 
            "maskFilename":"black-mask-whitecenter-512x768.png",
"prompt":"The sun, a Radiant sunburst, a golden disk encircled by vibrant rays. A confident, triumphant figure stands at its center, arms outstretched, bathed in warm illumination.",
"description":"XIX: Sun: Joy and Vitality\nOutlook: Radiance and Optimism",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
  "mandelbrotGenerate":{
         
    },
        }
        
        ,
        { 
            "maskFilename":"black-mask-whitecenter-512x768.png",
"prompt":"A women, surrounded by a glowing aura, stands upright in the center. In their right hand, they hold a balanced scale, delicately weighing two objects. Their left hand extends outward, revealing an open palm. Behind them, a large curtain drapes, revealing a vibrant landscape with rolling hills and a bright sun. The figure wears a flowing robe adorned with intricate symbols, and above their head hovers a crown",
"description":"XX: Judgment: Awakening and Rebirth\nOutlook: Renewal and Transformation",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
  "mandelbrotGenerate":{
         
    },
        }
        
        ,
        { 
            "maskFilename":"black-mask-whitecenter-512x768.png",
"prompt": "A globe adorned with intricate illustrations of diverse landscapes, interconnectedness of all places and cultures.",
"description": "XXI: The World: Completion and Wholeness\nOutlook: Harmony and Fulfillment",
            "maskInvert":False,  
#  "denoising_strength":0.6, 
  "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
  "mandelbrotGenerate":{
         
    },
        }
        
        
        ]
}]
gameAssetsLaserGames=[ {
    "path":"C:\\PRIVAT\\laser-api\\public\\media\\img\\",
    #   "colorFilename":"logo.png",
            # "maskFilename":"black5612x512.png",
            "prompt": "the interior of the Mandelbrot Museum. The dimly lit space is adorned with colorful displays of fractal patterns. The walls are lined with interactive exhibits, infinite complexity of the Mandelbrot set. In the center of the room, a large sculpture of the set stands proudly, its intricate curves and shapes reflecting the infinite possibilities.",
                      "description":"Image for game mandelbrot",
            "tileX":True,
            "name":"laser-mandelbrot",
            # "expand":8
    
    }, 
    {
    "path":"C:\\PRIVAT\\laser-api\\public\\media\\img\\",
    #   "colorFilename":"logo.png",
            # "maskFilename":"black5612x512.png",
            "prompt": "A solitary canvas stands tall in the museum entry hall, its vibrant colors and intricate brushstrokes capturing the essence of the artist's laser-pointer.",
                                  "description":"Image for game fire",
            "tileX":True,
            "name":"laser-fire",
          
          
    }, {
    "path":"C:\\PRIVAT\\laser-api\\public\\media\\img\\",
    #   "colorFilename":"logo.png",
            # "maskFilename":"black5612x512.png",
            "prompt": "This photograph captures a paradise-like setting, with lush greenery and a single forbidden fruit hanging from a tree, tempting the viewer with its alluring presence.",
            "description":"Image for game paradise",
            "tileX":True,
            "name":"laser-paradise",
          
          
    }, {
    "path":"C:\\PRIVAT\\laser-api\\public\\media\\img\\",
    #   "colorFilename":"logo.png",
            # "maskFilename":"black5612x512.png",
            "prompt": "A fierce torpedo battle ensues between two bases, each armed with laser defenses. The explosive clash of firepower illuminates the dark, ominous waters.",
                      "description":"Image for game Torpedo",
            "tileX":True,
            "name":"laser-torpedo",
          
          
    },{
    "path":"C:\\PRIVAT\\laser-api\\public\\media\\img\\",
    #   "colorFilename":"logo.png",
            # "maskFilename":"black5612x512.png",
            "prompt": "A lone cowboy stands tall, wielding a futuristic laser gun, amidst a stampede of cattle thundering through the rugged grassland terrain, flanked by towering rocky outcroppings.",
                    "description":"Image for game Cowhorde",
            "tileX":True,
            "name":"laser-cowhorde",
          
          
    },
    {
    "path":"C:\\PRIVAT\\laser-api\\src\\img\\gamelogos\templates\\",
    #   "colorFilename":"black5612x512.png",
            # "maskFilename":"black5612x512.png",
            "prompt": "A surreal underwater scene captured in a photograph, showcasing a tranquil pool with small fish swimming amongst dangerous sharks, surrounded by floating pieces of fodder.",  
                "description":"Image for game Shark-Pool",
            "tileX":True,
            "name":"laser-sharkpool",
          
          
    },
    {
    "path":"C:\\PRIVAT\\laser-api\\src\\img\\gamelogos\templates\\",
    #   "colorFilename":"black5612x512.png",
            # "maskFilename":"black5612x512.png",
            "prompt": "A mesmerizing underwater photography of life in a pool. The larvae in the foreground, while schools of fish glide past. And looming in the background, a menacing shark surveys the scene, creating an atmosphere of danger and intrigue.",
              "description":"Image for game Shark",
            "tileX":True,
            "name":"laser-shark"
          
    },
    {
    "path":"C:\\PRIVAT\\laser-api\\src\\img\\gamelogos\templates\\",
    #   "colorFilename":"black5612x512.png",
            # "maskFilename":"black5612x512.png",
            "prompt": "A flock of plump chubby fat thick and vibrant parrots soar through a scenic landscape of rugged mountain ranges. They navigate through tree obstacles, their rainbow wings flapping gracefully in a side view. The scene is presented in pixel art, with 2D obstacles adding to the challenge of the parrots' flight.",
                  "description":"Image for game Flapping",
            "tileX":True,
            "name":"laser-flapping"
          
    },
    {
    "path":"C:\\PRIVAT\\laser-api\\src\\img\\gamelogos\templates\\",
    #   "colorFilename":"black5612x512.png",
            # "maskFilename":"black5612x512.png",
            "prompt": "A dynamic photograph captures the intense laser torpedo battle between four bases, with beams of light crisscrossing in a mesmerizing display of power and strategy.",
                "description":"Image for game Basefight",
            "tileX":True,
            "name":"laser-basefight"
          
    },
    {
    "path":"C:\\PRIVAT\\laser-api\\src\\img\\gamelogos\templates\\",
    #   "colorFilename":"black5612x512.png",
            # "maskFilename":"black5612x512.png",
            "prompt": "A tower defense scene in 2D art style set in a forest with towers, archers, goblins, trolls, a castle in the distance, and intricate foliage. The sky is dramatic with a mix of blues and purples, and the scene is highly detailed and trending on ArtStation.",
            "tileX":True,
            "name":"laser-defense"
          
    },
     {
    "path":"C:\\PRIVAT\\laser-api\\src\\img\\gamelogos\templates\\",
    #   "colorFilename":"black5612x512.png",
            # "maskFilename":"black5612x512.png",
            "prompt": "a complex machine with a series of chutes and conveyor belts, humming with activity. On the right side of the machine, buckets of different colors are lined up, each labeled with a corresponding color. As the camera zooms in, brightly colored cubes begin to tumble down the chutes and are sorted into their designated buckets with impressive speed and precision.",
            "description":"Image for game Sorter",
            "tileX":True,
           
            "name":"laser-sorter"
    }
,
     {
    "path":"C:\\PRIVAT\\laser-api\\src\\img\\gamelogos\templates\\",
    #   "colorFilename":"black5612x512.png",
            # "maskFilename":"black5612x512.png",
            "prompt": "a vintage styled illuminated arena, consisting of various square-shaped obstacles arranged in a symmetrical pattern across the plane. Beams of colored lasers emerge from the barriers, adding an extra layer of difficulty. At the center of the scene, a luminous soccer ball floats, emitting a soft glow.",
            "description":"Image for game Pong",
            "tileX":True,
            "name":"laser-pong"
           
    }

]

theSetupLaserGames={
    
"sampler": "Euler",
"negative_prompts":" ",
"entries": gameAssetsLaserGames,
"additionalPrompts":""

}
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
            "prompt": "a mayan stone wall",
            "description":"Base Tile for Background ",
            "maskInvert":True, 
            "tileX":True,
            "tileY":True,
        "entry": [{
              "path":"C:\\PRIVAT\\godot-brickout-new\\stablediffusion-generator\\basics\\", 
            "maskFilename":"border-16px-512x512.png",
            "prompt":"a mayan stone wall",
            "description":"Base Tile Variation #1 for Background ",
            "maskInvert":True,  
 "denoising_strength":0.6, 
 "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
        },{
              "path":"C:\\PRIVAT\\godot-brickout-new\\stablediffusion-generator\\basics\\", 
            "maskFilename":"border-16px-512x512.png",
            "prompt":"a mayan stone wall",
            "description":"Base Tile Variation #2 for Background ",
            "maskInvert":True,   
 "denoising_strength":0.7,
 "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
        },{
              "path":"C:\\PRIVAT\\godot-brickout-new\\stablediffusion-generator\\basics\\", 
            "maskFilename":"border-16px-512x512.png",
            "prompt":"a mayan stone wall",
            "description":"Base Tile Variation #2 for Background ",
            "maskInvert":True,   
 "denoising_strength":0.8,
 "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
        },{
              "path":"C:\\PRIVAT\\godot-brickout-new\\stablediffusion-generator\\basics\\", 
            "maskFilename":"border-16px-512x512.png",
            "prompt":"a mayan stone wall",
            "description":"Base Tile Variation #2 for Background ",
            "maskInvert":True,   
 "denoising_strength":0.9,
 "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
        },{
              "path":"C:\\PRIVAT\\godot-brickout-new\\stablediffusion-generator\\basics\\", 
            "maskFilename":"border-16px-512x512.png",
            "prompt":"a skull engraving in a mayan brick wall",
            "description":"Base Tile Variation skull",
            "maskInvert":True,  
 "denoising_strength":0.8,
 "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
        },{
              "path":"C:\\PRIVAT\\godot-brickout-new\\stablediffusion-generator\\basics\\", 
            "maskFilename":"border-16px-512x512.png",
            "prompt":"a sun engraving in a mayan brick wall",
            "description":"Base Tile Variation sun",
            "maskInvert":True,  
 "denoising_strength":0.8,
 "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
        },{
              "path":"C:\\PRIVAT\\godot-brickout-new\\stablediffusion-generator\\basics\\", 
            "maskFilename":"border-16px-512x512.png",
            "prompt":"an ancient rune engraving in a mayan brick wall",
            "description":"Base Tile Variation Rune ",
            "maskInvert":True,  
 "denoising_strength":0.9,
 "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
        },{
              "path":"C:\\PRIVAT\\godot-brickout-new\\stablediffusion-generator\\basics\\", 
            "maskFilename":"border-16px-512x512.png",
            "prompt":"an cherry engraving in a mayan brick wall",
            "description":"Base Tile Variation Cherry ",
            "maskInvert":True,  
 "denoising_strength":0.9,
 "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
        }]
        
    },
    {
    "path":"C:\\PRIVAT\\godot-brickout-new\\stablediffusion-generator\\basics\\",
      "colorFilename":"black5612x512.png",
            "maskFilename":"black-mask-whitecenter-whiteborder-512x512.png",
            "prompt":"a yellow-orange plastic border frame game token",
            "maskInvert":True,
            "ref":"theFrame",
        "entry": [
    # an entry is a color image, a mask and a prompt, each sub entry image is initialised with the result of the entry, sub images only add masks to the created image
        {
            # each inner entries colorfilename is alpa added to the previous result 
            "maskFilename":"black-mask-whitecenter-512x512.png",
            "outFilename":"red.png",
            # "colorFilename":"diamond.png",
            "maskInvert":False,
            "prompt":"a red playing token, dark red background",
           
        },

        {
            # each inner entries colorfilename is alpa added to the previous result 
            "maskFilename":"black-mask-whitecenter-512x512.png",
            "outFilename":"green.png",
            "maskInvert":False,
            "prompt":"a green playing token, dark green background",
        },
        {
            # each inner entries colorfilename is alpa added to the previous result 
            "maskFilename":"black-mask-whitecenter-512x512.png",
            "maskInvert":False,
            "prompt":"a blue playing token, dark blue background",
            "outFilename":"blue.png",
        },
        {
            # each inner entries colorfilename is alpa added to the previous result 
            "maskFilename":"black-mask-whitecenter-512x512.png",
            "maskInvert":False,
            "prompt":"a purple playing token, dark purple background",
            "outFilename":"purple.png",
        },
        {
            # each inner entries colorfilename is alpa added to the previous result 
            "maskFilename":"black-mask-whitecenter-512x512.png",
            "maskInvert":False,
            "outFilename":"yellow.png",
            "prompt":"a yellow playing token, dark yellow background",
        },
        {
            # each inner entries colorfilename is alpa added to the previous result 
            "maskFilename":"black-mask-whitecenter-512x512.png",
            "maskInvert":False,
            "outFilename":"cyan.png",
            "prompt":"a cyan playing token, dark cyan background",
        },
        {
            # each inner entries colorfilename is alpa added to the previous result 
            "maskFilename":"black-mask-whitecenter-512x512.png",
            "maskInvert":False,
            "outFilename":"bomb.png",
            "prompt":"a playing token of a fusing bomb",
        },
        {
            # each inner entries colorfilename is alpa added to the previous result 
            "maskFilename":"black-mask-whitecenter-512x512.png",
            "maskInvert":False,
            "outFilename":"joker.png",
            "prompt":"a playing token of a harlequin",
        },
        {
            # each inner entries colorfilename is alpa added to the previous result 
            "maskFilename":"black-mask-whitecenter-512x512.png",
            "maskInvert":False,
            "outFilename":"bonus.png",
            "prompt":"a playing token of a pile of gold",
        }
            ],
         
},
{
    "path":"C:\\PRIVAT\\godot-brickout-new\\stablediffusion-generator\\basics\\",
      "maskFilename":"mask-person.png",
      "colorFilename":"person-color.png",
            "prompt":"""A portrait of player character, normal, relaxed, equillibrium""",
            
            "ref":"thePersonNormal",
            "description":"""This is Normal Humpty Dumpty the Garlic Farmer
let's watch his story...""",
            "maskInvert":False,
             
            
 "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
            
        "entry": [
            {
            "prompt":"""A portrait of player character, grief, sad, pensive""",
      "maskFilename":"mask-person.png",
            "maskInvert":False,
            "ref":"thePersonSad",
 "denoising_strength":0.6,
 "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
            "description":"""INNER: TRANSFORMED 1 griefing""",

},
            {
            "prompt":"""A portrait of player character, amazed, surprised, distracted""",
      "maskFilename":"mask-person.png",
            "maskInvert":False,
            "ref":"thePersonAmazed",
 "denoising_strength":0.6,
 "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
            "description":"""INNER: TRANSFORMED 2 amaze""",

},
            {
            "prompt":"""A portrait of player character, terror, fear, epprehension""",
      "maskFilename":"mask-person.png",
            "ref":"thePersonTerror",
            "maskInvert":False,
 "denoising_strength":0.6,
 "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
            "description":"""INNER: TRANSFORMED 3 terror fear apprehension""",

},
        
            {
            "prompt":"""A portrait of player character, ecstatic, joy, serenity, smiling, teeth""",
      "maskFilename":"mask-person.png",
            "ref":"thePersonEcstatic",
            "maskInvert":False,
 "denoising_strength":0.6,
 "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
            "description":"""INNER: TRANSFORMED 6 ecstatic""",

}, 
{
    "path":"C:\\PRIVAT\\godot-brickout-new\\stablediffusion-generator\\basics\\",
      "maskFilename":"mask-person.png", 
            "prompt":"""A portrait of player character, rage, anger, annoyance""",
             
            "ref":"thePersonRaging",
            "description":"""this is raging humpty dumpty""",
            "maskInvert":False,
 "denoising_strength":0.6,
 "inpainting_fill":1, # 0=fill 1=original 2=latent noise 3=latent nothing
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
    #         "denoising_strength":0.1,
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
        asi=AsymmetricTiling()
        # # output_images, info = None, None
        initial_seed = p.seed 
        initial_info = ""
        initial_cfg_scale = p.cfg_scale
        initial_inpainting_fill=p.inpainting_fill
        initial_denoising_strength=p.denoising_strength
        processed=None
        imageIndex=0
        
        outfilename =  time.strftime('%Y%m%d%H%M%S')
        outpath =  os.path.join(p.outpath_samples, 'ilc-gameart-gen')
        if not os.path.exists(outpath):
                    os.mkdir(outpath)
        outpath =  os.path.join(p.outpath_samples, 'ilc-gameart-gen',outfilename)
        if not os.path.exists(outpath):
                    os.mkdir(outpath)
        def renderText(image,text,position=(100,100)):
            draw=ImageDraw.Draw(image)
            font = ImageFont.truetype(r'arial.ttf', 18) 
            spacing = 20 
            print("Rendering text",text)
            pad=5

            txtsize = draw.multiline_textbbox(position, text, spacing = spacing,  font=font)
            draw.rounded_rectangle((txtsize[0] - pad,
                                txtsize[1] - pad,
                                txtsize[2] + 2 * pad,
                                txtsize[3] + 2 * pad), radius=pad, fill=(244,243,242))

            # drawing text size
            draw.text((position[0]+1,position[1]+1), text, fill ="black", font = font,  spacing = spacing, align ="left",stroke_width =1) 
            draw.text( position,text, fill ="red", font = font,  spacing = spacing, align ="left",stroke_width =1) 
            
        def expandCurrentImage(expandCount,processed):

            orig_image_mask=p.image_mask
            orig_latent_mask=p.latent_mask
            orig_init_images=p.init_images
            orig_seed=p.seed

            rangedExpandedImage= Image.new("RGB",(p.width*expandCount//2+p.width,p.height),"black")
            rangedExpandedImage.paste(processed.images[0])
            for i in range(expandCount):


                 rangedLatentMask=Image.new("L",(p.width,p.height))
                #  /** ok, what is latent mask?
                #  **/
                 draw = ImageDraw.Draw(rangedLatentMask) 
                 draw.rectangle(( p.width//2-p.mask_blur*2  ,0,p.width ,p.height),'white')

                 rangedMask=Image.new("L",(p.width,p.height))
                 draw = ImageDraw.Draw(rangedMask) 
                 draw.rectangle(( p.width//2-p.mask_blur*2 ,0,p.width ,p.height),'white')

                 rangedCollor=Image.new("RGB",(p.width,p.height))
                 rangedCollor.paste(processed.images[0],(-p.width//2,0))

                 p.image_mask=rangedMask
                 p.latent_mask=rangedLatentMask
                 p.init_images=[rangedCollor]
                 p.seed=initial_seed

                 processed = processing.process_images(p)
                 rangedExpandedImage.paste(processed.images[0],(i*p.width//2+p.width//2,0))
                    
                 if(rangedExpandedImage != None):
                    rangedExpandedImage.save(outpath+"/expanded_"+str(imageIndex)+".png")

            

            p.image_mask=orig_image_mask
            p.latent_mask=orig_latent_mask
            p.init_images=orig_init_images
            p.seed=orig_seed
        def countEntry(entry ):
            result=1
            if("entry" in entry):
                result+=countEntries(entry["entry"])
            return result
        def countEntries(entry ):
            result=0
            for centry in entry:
                result=result+countEntry(centry)
                if("expand" in entry):
                     result+=entry["expand"]    

            return result
        def handleAdd(addd,currentImage):
            if(addd['ref'] in refs):
                currentImage.paste(refs[addd['ref']]['image'],(0,00),refs[addd['ref']]['mask'])
            else:
                print("ERROR: Ref not found",addd['ref'])
            return
        

        negative_prompt=p.negative_prompt
        def handleEntry(entry,currentImage=None,path=None,depth=0):   
            if state.interrupted:
                # Interrupt button pressed in WebUI
                return 
            expand=0
            nonlocal initial_seed,initial_info,processed,resultPicture,imageIndex,refs,initial_cfg_scale,outpath,all_images_collection
            imageIndex=imageIndex+1 
            print("Handling entry",entry)
            if(path==None):
                path=entry.get("path","")

            colorImage=None
            if("maskInvert" in entry):
                p.inpainting_mask_invert =entry["maskInvert"]
            else:
                p.inpainting_mask_invert=False   
            if("expand" in entry):
                 expand=entry["expand"]

            if("inpainting_fill" in entry):
                p.inpainting_fill=entry["inpainting_fill"]
            else:
                p.inpainting_fill=initial_inpainting_fill
            if("seed" in entry):
                p.seed=entry["seed"]
            else:
                p.seed=initial_seed  
                # if(p.seed!=None):
                #     p.seed=p.seed+0;#imageIndex

            if("cfg_scale" in entry):
                p.cfg_scale=entry["cfg_scale"]
            else:
                p.cfg_scale=initial_cfg_scale
            if("negative_prompt" in entry):
                p.negative_prompt=negative_prompt+' '+entry["negative_prompt"]
            else:
                p.negative_prompt=negative_prompt 

            print('entry denoising strenhgth is',  entry)
            print('entry denoising strenhgth is',"denoising_strength" in entry)
            if("denoising_strength" in entry):
                p.denoising_strength=entry["denoising_strength"]
                print("Setting denoising_strength to",p.denoising_strength)
            else:
                p.denoising_strength=initial_denoising_strength
                print("Default denoising_strength to",p.denoising_strength)

            if("colorFilename" in entry):
                # either currentImage is present, then add colorFilename of entry to current
                # or the currentImage is not present, then load the colorFilename as full currentImage

                filename=entry["colorFilename"]
                if("@" in filename):
                    print("Using ref Image",filename[1:])
                    colorImage=refs[filename[1:]]['image']                   
                else:

                    if(currentImage==None):
                        colorImage=Image.new("RGB",size=(p.width,p.height),color='black')
                        newImage =Image.open(os.path.join(path, entry["colorFilename"])).resize(p.width,p.height),
                        colorImage = Image.alpha_composite(colorImage.convert("RGBA"), newImage.convert("RGBA")).convert("RGB")
                    else: 
                        newImage = Image.open(os.path.join(path, entry["colorFilename"])) .resize(p.width,p.height),
                        colorImage = Image.alpha_composite(currentImage.convert("RGBA"), newImage.convert("RGBA")).convert("RGB")

                    if(entry["colorFilename"] in usedMasks):
                        # all_images_collection.append([ colorImage])
                        usedMasks[entry["colorFilename"]]=colorImage
            else: 
                    colorImage=currentImage
                    if colorImage == None:
                        colorImage=p.init_images[0] 

            if("bgColorImage" in entry):
                colorImage=Image.new("RGB",size=(p.width,p.height),color=entry['bgColorImage'])
            
            
            if("maskFilename" in entry):
                maskImage=Image.open(os.path.join(path, entry["maskFilename"])).resize((p.width,p.height)).convert('L')
                
                # if(p.inpainting_mask_invert):
                #     if(not entry["maskFilename"]+"invert" in usedMasks):
                #             all_images_collection.append([ PIL.ImageOps.invert( maskImage.convert('RGB'))])
                #             usedMasks[entry["maskFilename"]+"invert"]=maskImage
                # else:
                if(not entry["maskFilename"] in usedMasks):
                    all_images_collection.append([ maskImage])
                    usedMasks[entry["maskFilename"]]=maskImage
            else:
                maskImage=Image.new("L",size=(p.width,p.height),color='white')


            if("mandelbrotGenerate" in entry):
                 mandelbrot=makePilILCMandelbrot(imageSize=(p.width,p.height), scale=0.25,seed= p.seed)
                
                 colorImage.paste(mandelbrot,(0,0),maskImage)
                #  colorImage.paste(mandelbrot,(0,0),maskImage)
                
                 all_images_collection.append([ mandelbrot.copy()])

                 if("name" in entry):
                        mandelbrot.save(outpath+"/"+str(entry["name"])+ '-'+str(counterimages)+"_mandelbroti.png")
                 else:
                        mandelbrot.save(outpath+"/file_"+str(imageIndex)+"_mandelbroti.png")

            if("text" in entry):
                #  // generate some nixe text on all ;)
                    renderText(colorImage,entry['text']['text'],(entry['text']['x'],entry['text']['y']))

# //                colorImage= 

            p.prompt=entry["prompt"]+' '+globalprompt

            p.image_mask=maskImage
            # p.latent_mask=maskImage
            print("Rendering entry",p.prompt) 
            print("Rendering entry",p.image_mask,p.init_images) 
            p.init_images = [colorImage]
            p.extra_generation_params["entry-{imageIndex}-prompt"]=p.prompt
             
            print("Rendering entry",p.image_mask,p.init_images)


            try:
                tileX=False
                tileY=False
                if("tileX" in entry):
                    tileX=entry['tileX']
                if("tileY" in entry):
                    tileY=entry['tileY']
                asi.__hijackConv2DMethods(tileX, tileY)

                processed = processing.process_images(p)
            finally:
            # Restore model behaviour to normal, even if something went wrong during processing.
                asi.__restoreConv2DMethods()
            if state.interrupted:
                # Interrupt button pressed in WebUI
                return 

            post_processed_image = processed.images[0].copy()

            # save single image to file
            if("name" in entry):
                post_processed_image.save(outpath+"/"+str(entry["name"])+ '-postprocessed-'+str(imageIndex)+".png")
            else:
                post_processed_image.save(outpath+"/file_"+str(imageIndex)+".png")
                 
            
            if("adds" in entry):
                    for addd in entry["adds"] :
                        handleAdd(addd,post_processed_image)


            if("ref" in entry):
                refs[entry["ref"]]={
                    "image":post_processed_image.copy(),
                    "mask":maskImage.copy(),
                }
               
            
            rowGap=128
            xPos=((imageIndex-1)%4)*p.width
            yPos=((imageIndex-1)//4)*(p.height+rowGap)
            resultPicture.paste(post_processed_image,(               xPos,yPos                )                )
            # resultPictureFlat.paste(post_processed_image,(               ((imageIndex-1)%6)*p.width,((imageIndex-1)//6)*(p.height)                )                )
            
            if("description" in entry):
                renderText(resultPicture,entry['description'],(xPos,yPos+p.height))

            if initial_seed is None:
                initial_seed = processed.seed
                initial_info = processed.info  
            counterimages=0
            for imga in processed.images:  
                        counterimages=counterimages+1;
                        # imga.save(os.path.join(outpath, f"{index}-{outfilename}_{frame_no:05}.png"))                        
                        all_images_collection.append([imga]) 

                        
                        if("name" in entry):
                            imga.save(outpath+"/"+str(entry["name"])+ '-'+str(counterimages)+".png")
                        else:
                            imga.save(outpath+"/file_"+str(imageIndex)+".png")

            expandCurrentImage(expand,processed)

            if("entry" in entry):
                for childEntry in entry["entry"]:
                    handleEntry(childEntry,currentImage=post_processed_image,path=path,depth=depth+1) 
            return post_processed_image

        initial_seed_save=p.seed

        setToUse=tarotCardSet
         
        all_images_collection = [] 
        
        resultPicture=Image.new("RGB", 
        (
            p.width*4,
            (128+p.height)*(1+(countEntries(setToUse)//4) ),            
        )
        )
        
        # resultPictureFlat=Image.new("RGB", 
        # (
        #     p.width*6,
        #     (p.height)*((countEntries(gameAssetsLaserGames)//6) ),            
        # )
        # )

        # for step in range(0,p.n_iter): 
        # Fix variable types, i.e. text boxes giving strings.
        p.seed=initial_seed_save  
   

        processing.fix_seed(p)
        batch_count = countEntries(setToUse)*p.n_iter

        # Save extra parameters for the UI
        p.extra_generation_params = {
            # "Config Path": gameAssets["path"], 
            # "Config File": gameAssets["path"], 
        }

        p.batch_size = 1 
        state.job_count = 1* batch_count
        for asset in setToUse[0:]:
            handleEntry(asset)
        # handleEntry(gameAssets[0])
        # handleEntry(gameAssets[2]) 
 
        # # Process current frame
        # processed = processing.process_images(p)

        # index=0
        # for imga in processed.images: 
        #     index=index+1
        #     # imga.save(os.path.join(outpath, f"{index}-{outfilename}_{frame_no:05}.png"))
            
        #     all_images_collection.append([])  
        #     all_images_collection[index-1].append(imga)

  
        # save result map image to file
        resultPicture.save(outpath+"/all_comic.png")
        # resultPictureFlat.save(outpath+"/all_flat.png")
        dickies=[resultPicture]
        for images in all_images_collection:
            for image in images:
                dickies.append(image)

        processed = Processed(p, dickies, initial_seed, initial_info,comments="no comments") 

        return processed
