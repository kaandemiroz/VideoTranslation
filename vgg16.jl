using Knet
using MAT
using JLD

# convolution, bias and activation function
@knet function cbf(x; f = :relu, o...)
	y = wconv(x; o...)
	z = bias4(y; o...)
	return f(z; o...)
end

# Reduced repetitions
@knet function vgg16(data; weights = nothing, pdrop = 0.5, cwin = 3, pwin = 2, pad = 1, o...)
	# pool() functions have stride = window & MAXPOOL by default
	# convolution strides are 1 by default
	conv1_1 = cbf(data; out = 64, cinit = map(Float64,weights[1]["weights"][1]), binit = reshape(map(Float64,weights[1]["weights"][2]), (1,1,64,1)), window = cwin, padding = pad, o...)
	conv1_2 = cbf(conv1_1; out = 64, cinit = map(Float64,weights[3]["weights"][1]), binit = reshape(map(Float64,weights[3]["weights"][2]), (1,1,64,1)), window = cwin, padding = pad, o...)
	pool1 = pool(conv1_2; pwindow = pwin, o...)
	conv2_1 = cbf(pool1; out = 128, cinit = map(Float64,weights[6]["weights"][1]), binit = reshape(map(Float64,weights[6]["weights"][2]), (1,1,128,1)), window = cwin, padding = pad, o...)
	conv2_2 = cbf(conv2_1; out = 128, cinit = map(Float64,weights[8]["weights"][1]), binit = reshape(map(Float64,weights[8]["weights"][2]), (1,1,128,1)), window = cwin, padding = pad, o...)
	pool2 = pool(conv2_2; pwindow = pwin, o...)
	conv3_1 = cbf(pool2; out = 256, cinit = map(Float64,weights[11]["weights"][1]), binit = reshape(map(Float64,weights[11]["weights"][2]), (1,1,256,1)), window = cwin, padding = pad, o...)
	conv3_2 = cbf(conv3_1; out = 256, cinit = map(Float64,weights[13]["weights"][1]), binit = reshape(map(Float64,weights[13]["weights"][2]), (1,1,256,1)), window = cwin, padding = pad, o...)
	conv3_3 = cbf(conv3_2; out = 256, cinit = map(Float64,weights[15]["weights"][1]), binit = reshape(map(Float64,weights[15]["weights"][2]), (1,1,256,1)), window = cwin, padding = pad, o...)
	pool3 = pool(conv3_3; pwindow = pwin, o...)
	conv4_1 = cbf(pool3; out = 512, cinit = map(Float64,weights[18]["weights"][1]), binit = reshape(map(Float64,weights[18]["weights"][2]), (1,1,512,1)), window = cwin, padding = pad, o...)
	conv4_2 = cbf(conv4_1; out = 512, cinit = map(Float64,weights[20]["weights"][1]), binit = reshape(map(Float64,weights[20]["weights"][2]), (1,1,512,1)), window = cwin, padding = pad, o...)
	conv4_3 = cbf(conv4_2; out = 512, cinit = map(Float64,weights[22]["weights"][1]), binit = reshape(map(Float64,weights[22]["weights"][2]), (1,1,512,1)), window = cwin, padding = pad, o...)
	pool4 = pool(conv4_3; pwindow = pwin, o...)
	conv5_1 = cbf(pool4; out = 512, cinit = map(Float64,weights[25]["weights"][1]), binit = reshape(map(Float64,weights[25]["weights"][2]), (1,1,512,1)), window = cwin, padding = pad, o...)
	conv5_2 = cbf(conv5_1; out = 512, cinit = map(Float64,weights[27]["weights"][1]), binit = reshape(map(Float64,weights[27]["weights"][2]), (1,1,512,1)), window = cwin, padding = pad, o...)
	conv5_3 = cbf(conv5_2; out = 512, cinit = map(Float64,weights[29]["weights"][1]), binit = reshape(map(Float64,weights[29]["weights"][2]), (1,1,512,1)), window = cwin, padding = pad, o...)
	pool5 = pool(conv5_3; pwindow = pwin, o...)
	# VGG uses convolutional layers in fully connected as well
	fc6 = cbf(pool5; out = 4096, winit = map(Float64, weights[32]["weights"][1]), binit = reshape(map(Float64, weights[32]["weights"][2]), (1,1,4096,1)), window = 7, o...)
	# fc6 = wbf( pool5; out = 4096, winit = transpose( reshape(map(Float64,weights[32]["weights"][1]), (size(weights[32]["weights"][1],1)*size(weights[32]["weights"][1],2)*size(weights[32]["weights"][1],3), 4096) ) ), binit = reshape(map(Float64,weights[32]["weights"][2]), (1,1,4096,1)), f = :relu, o...)
	fc6 = drop(fc6; pdrop = pdrop, o...)
	fc7 = cbf(fc6; out = 4096, winit = map(Float64, weights[34]["weights"][1]), binit = reshape(map(Float64, weights[34]["weights"][2]), (1,1,4096,1)), window = 1, o...)
	# fc7 = wbf(fc6; out = 4096, winit = transpose( reshape(map(Float64,weights[34]["weights"][1]), (size(weights[34]["weights"][1],1)*size(weights[34]["weights"][1],2)*size(weights[34]["weights"][1],3), 4096) ) ), binit = reshape(map(Float64,weights[34]["weights"][2]), (1,1,4096,1)), f = :relu, o...)
	return fc7
end

# Compile the VGG-16 Model to vgg16.jld
function main()
	matpath = "vgg16.mat"
	weights = matread(matpath)["layers"]
	vgg16 = compile(:vgg16; weights = weights)
	JLD.save("vgg16.jld", "model", clean(vgg16))
end

main()
