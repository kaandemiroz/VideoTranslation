#VideoTranslation

##Replicating the code:
###Downloads
####Julia
Download and install Julia here: http://julialang.org/downloads/
####Knet
In Julia's console, run 
"Pkg.clone("git://github.com/denizyuret/Knet.jl.git")" and then "Pkg.build("Knet")"
####VGG-16 Model
Download the pre-trained VGG-16 Model here: http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat
######It is necessary that this file is downloaded to the same directory as the .jl files.

###Run the preliminary scripts
####Run these three Julia scripts:
#####vgg16.jl
#####lstm.jl
#####lstm_data.jl
######It is recommended that you are in the project folder while running these files
These programs will create the necessary .jld files in your current workspace

###Run the translation script
######videotranslation.jl
This file is the main code in which the translation occurs.
