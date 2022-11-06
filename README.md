Update 6/11/22: finnally Version 2.0 is out!
I tried to make it more like a package. More of methods and object oriented.

Check out on the main branch. I moved the other one branch: TB+read_data, I will keep that for a while just in case!

Now, you may run like this:
python input_script.py
##########

# TB-twist-bilayer
To my best knowledge there is no TB code publicly available for twisted bilayer and 2d materials. 
I would like to publish this library at some point and get citations, please keep it private for now.

This code has potential to turn into a smarter tool, for others who just want to input any desired input structre for twisted bi-layer, like XYZ and get the levels output...  without worryng about the indeces of atoms, orientations, local normal vercor (to be developed).

Moreover, if we combine with Jin's code -the one makes bilayer-. We can make something for absolute idiot people, who don't want to do any effort but just entering any desired misfit angle and get a picture.

I am having efforts toward obect oriented programming, please advise your ideas on this code.

As the state of the code untill now -24/10-
please just read and run it for yourself, aviod commiting, still there are some crucial fixes and ideas I have which can make it much faster. 
right now, 50 K-point of misfit=1.05 takes around 40 minutes... I believe that can get reduced to ~20 minutes.

You can run the code like:

python TB.py unrelax_105.data
