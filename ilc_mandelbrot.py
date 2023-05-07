import matplotlib.pyplot as plt
import random
from datetime import datetime
import time
import sys
from perlin_noise import PerlinNoise
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from colour import Color
class Storage:
    cache=None 

def getTimeString():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S") 
    return current_time
#
# I-Love-Chaos Logo implementation for python, the seeds are alternated in each iteration step
# returns a pil image with text in arial font written to it

def ILC_LOGGER(ilc_name='ILC Name' ,ilc_version=1.0):
     def log(message='',*args,**kwargs):
            print("[i-love-chaos",ilc_name+"@"+str(ilc_version),"]",getTimeString(), ">>",message,*args,**kwargs)
     return log
def getILCBase64ImageSingleton():
        buff = BytesIO()
        makePilILCMandelbrotLogo(128).save(buff, format="PNG")
        img_str = base64.b64encode(buff.getvalue())
        return img_str
  
     
     
def makePilILCMandelbrotLogo(
    imageSize=128,
    maxIters=32,
    color1Inner=(255, 0, 0),
    color2Outer=(255, 255, 255)
):  
    # if Storage.cache!=None:
    #     print("Cache HIT#")
    #     return Storage.cache

    width = imageSize
    height = imageSize

    # Create a new image
    im = Image.new('RGB', (width, height))

    # Maximum number of iterations
    max_iterations = maxIters

    def lerp(a: float, b: float, t: float) -> float:
        return (1.0 - t) * a + t * b
    randomstate=random.getstate()
    
    random.seed()
    seed1 = 0.15*complex(
        random.random()*2-1,
        random.random()*2-1
    )
    seed2 = 0.15*complex(
        random.random()*2-1,
        random.random()*2-1
    )
    seed3 = 0.15*complex(
        random.random()*2-1,
        random.random()*2-1
    )
    random.setstate(randomstate) 

    # Loop over the image pixels
    for x in range(width):
        for y in range(height):
            # Initialize the complex number z
            z = complex(0, 0)
            c = complex(-1*(y/height*4-2), x/width*4-2)+complex(-1, -0.25)
            # Compute the Mandelbrot set

            seeds = [c, c+seed1, c+seed2,c+seed3]
            for i in range(max_iterations+1):
                z = z*z + seeds[i % len(seeds)]

                # If the magnitude of z is greater than 2, it is not part of the Mandelbrot set
                if abs(z) > 2:
                    break

            normalizedI = 1-i/max_iterations
            # Set the pixel color inner and outer separately
            if i == max_iterations:
                im.putpixel((x, y), (
                    int((z.real*z.real+.85)*color1Inner[0]),
                    int((z.imag*z.imag)*color1Inner[1]),
                    int(color1Inner[2]),
                ))
            else:
                im.putpixel((x, y), (
                    int(lerp(color1Inner[0], color2Outer[0], normalizedI)),
                    int(lerp(color1Inner[1], color2Outer[1], normalizedI)),
                    int(lerp(color1Inner[2], color2Outer[2], normalizedI)),
                ))

    # Save the image
    # im.save('mandelbrot.png')
    font = ImageFont.truetype('arial', size=width-width//3)
    font2 = ImageFont.truetype('arial', size=width//4)

    textChaos = 'CHΑΩS'
    draw = ImageDraw.Draw(im)
    size = draw.textbbox(text=textChaos, font=font2, xy=(0, 0))
    draw.text(
        ((width-size[2])//2, 0),  # Coordinates
        'I',  # Text
        (0, 0, 0), font=font,
    )
    print('Size is', width, size)
    print('Size is', (width-size[2]))
    draw.text((
        (width-size[2])//2,
        width-width//3),  # Coordinates
        textChaos,  # Text
        (0, 0, 0),
        font=font2
    )
    draw.rounded_rectangle((0, 0, width-2, height-2), fill=None, outline="white",
                           width=4, radius=imageSize//10)
    # im.show()
    # # Plot the results using matplotlib
    # plt.imshow(im)
    # plt.show()
    Storage.cache=im
    return im


     
def makePilILCMandelbrot(
    imageSize=(128,128), 
    maxIters=32,
    color1InnerIn=(255, 0, 0),
    color2OuterIn=(255, 255, 255),
    scale=1,
    seed=None,radius=-0.15
):  
    log=ILC_LOGGER("makePilILCMandelbrot",1.0)
    # if Storage.cache!=None:
    #     print("Cache HIT#")
    #     return Storage.cache

    color1Inner=color1InnerIn 
    color2Outer=color2OuterIn
    width,height = imageSize 

    # Create a new image
    im = Image.new('RGB', (width, height))

    # Maximum number of iterations
    max_iterations = maxIters

    def lerp(a: float, b: float, t: float) -> float:
        return (1.0 - t) * a + t * b
    randomstate=random.getstate() 
    if(seed!=None):
         random.seed( )
    else:
        random.seed(seed) 
    log("Making Mandelbrot")
    seed1 = radius*complex(
        random.random()*2-1,
        random.random()*2-1
    )
    seed2 =radius*complex(
        random.random()*2-1,
        random.random()*2-1
    )
    seed3 = radius*complex(
        random.random()*2-1,
        random.random()*2-1
    )
    perlin1=PerlinNoise(octaves=10, seed=random.randrange(sys.maxsize))
    perlin2=PerlinNoise(octaves=10, seed=random.randrange(sys.maxsize))
    perlin3=PerlinNoise(octaves=10, seed=random.randrange(sys.maxsize))
    perlin4=PerlinNoise(octaves=10, seed=random.randrange(sys.maxsize))
    perlin5=PerlinNoise(octaves=10, seed=random.randrange(sys.maxsize))
    perlin6=PerlinNoise(octaves=10, seed=random.randrange(sys.maxsize))
    random.setstate(randomstate) 
    # Loop over the image pixels
    for x in range(width):
        for y in range(height):
            # Initialize the complex number z
            z = complex(0, 0)
            c = complex(1*(y/height*4-2)*scale, (x/width*4-2)*scale)+complex(-1, -0.25)
            # Compute the Mandelbrot set

            seeds = [c, c+seed1, c+seed2,c+seed3]
            for i in range(max_iterations+1):
                z = z*z + seeds[i % len(seeds)]

                # If the magnitude of z is greater than 2, it is not part of the Mandelbrot set
                if abs(z) >4:
                    break

            normalizedI = 1-i/max_iterations
            normalizedII = i/max_iterations
            # color1Inner=(
            #         random.randrange(0,255),
            #         random.randrange(0,255),
            #         random.randrange(0,255)
            #                          )
        #     color1Inner=(
        #    color1InnerIn[0]+      int(perlin1([x/width*10,y/height*10])*255-128), 
        #       color1InnerIn[1]+     int(perlin2([x/width*10,y/height*10])*255-128),
        #       color1InnerIn[2]+     int(perlin3([x/width*10,y/height*10])*255-128),
        #     )        
        #     color2Outer=(
        #     color2OuterIn[0]+       int(perlin4([x/width*10,y/height*10])*255-128), 
        #      color2OuterIn[1]+        int(perlin5([x/width*10,y/height*10])*255-128),
        #       color2OuterIn[2]+      int(perlin6([x/width*10,y/height*10])*255-128),
        #     )

            # Set the pixel color inner and outer separately
            if i == max_iterations:
                im.putpixel((x, y), (
                    int((z.real*z.real )*color1Inner[0]),
                    int((z.imag*z.imag)*color1Inner[1]),
               int( (abs(z)/4)* int(color1Inner[2])),
                ))  
                im.putpixel((x, y), (
                   color1Inner[0],
                   color1Inner[1],
                color1Inner[2],
                ))
            else:
               
                if(i%2==0):
                    im.putpixel((x, y), (
                        int(lerp(color1Inner[0], color2Outer[0], normalizedI)),
                        int(lerp(color1Inner[1], color2Outer[1], normalizedI)),
                        int(lerp(color1Inner[2], color2Outer[2], normalizedI)),
                    ))
                else:
                    im.putpixel((x, y), (
                        int(lerp(color2Outer[0], color1Inner[0], normalizedI)),
                        int(lerp(color2Outer[1], color1Inner[1], normalizedI)),
                        int(lerp(color2Outer[2], color1Inner[2], normalizedI)),
                    ))

                # im.putpixel((x, y), (
                #     round(normalizedI*255),
                #     round((z.real/abs(z))*255),
                #    round( (z.imag/abs(z))*255)
                    
                # ))
 
    log("Making Mandelbrot Finish")
    return im


# makePilILCMandelbrotLogo(128)
