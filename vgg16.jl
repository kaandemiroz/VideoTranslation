using Knet
using MAT
using JLD

# convolution, bias and activation function
@knet function cbf(x; out = 0, window = 3, padding = 1, f = :relu, o...)
	y = wconv(data; out = out, window = window, padding = padding, o...)
	z = bias4(y; o...)
	return f(z; o...)
end

# Reduced repetitions
@knet function vgg16(data; weights = nothing, pdrop = 0.5, win = 3, pad = 1, o...)
	# pool() functions have stride = window=2 MAXPOOL by default
	# convoulution strides are 1 by default
	conv1 = cbf(data; out = 64, cinit = weights[1]["weights"][1], binit = weights[1]["weights"][2], window = win, padding = pad, o...)
	pool1 = cbfp(conv1; out = 64, cinit = weights[3]["weights"][1], binit = weights[3]["weights"][2], cwindow = win, padding = pad, o...)
	conv2 = cbf(pool1; out = 128, cinit = weights[6]["weights"][1], binit = weights[6]["weights"][2], window = win, padding = pad, o...)
	pool2 = cbfp(conv2; out = 128, cinit = weights[8]["weights"][1], binit = weights[8]["weights"][2], cwindow = win, padding = pad, o...)
	conv3_1 = cbf(pool2; out = 256, cinit = weights[11]["weights"][1], binit = weights[11]["weights"][2], window = win, padding = pad, o...)
	conv3_2 = cbf(conv3_1; out = 256, cinit = weights[13]["weights"][1], binit = weights[13]["weights"][2], window = win, padding = pad, o...)
	pool3 = cbfp(conv3_2; out = 256, cinit = weights[15]["weights"][1], binit = weights[15]["weights"][2], cwindow = win, padding = pad, o...)
	conv4_1 = cbf(pool3; out = 512, cinit = weights[18]["weights"][1], binit = weights[18]["weights"][2], window = win, padding = pad, o...)
	conv4_2 = cbf(conv4_1; out = 512, cinit = weights[20]["weights"][1], binit = weights[20]["weights"][2], window = win, padding = pad, o...)
	pool4 = cbfp(conv4_2; out = 512, cinit = weights[22]["weights"][1], binit = weights[22]["weights"][2], cwindow = win, padding = pad, o...)
	conv5_1 = cbf(pool4; out = 512, cinit = weights[25]["weights"][1], binit = weights[25]["weights"][2], window = win, padding = pad, o...)
	conv5_2 = cbf(conv5_1; out = 512, cinit = weights[27]["weights"][1], binit = weights[27]["weights"][2], window = win, padding = pad, o...)
	pool5 = cbfp(conv5_2; out = 512, cinit = weights[29]["weights"][1], binit = weights[29]["weights"][2], cwindow = win, padding = pad, o...)
	fc6 = wbf(pool5; out = 4096, winit = weights[32]["weights"][1], binit = weights[32]["weights"][2], f = :relu, o...)
	fc6 = drop(fc6; pdrop = pdrop, o...)
	fc7 = wbf(fc6; out = 4096, winit = weights[34]["weights"][1], binit = weights[34]["weights"][2], f = :relu, o...)
	return fc7
end

# Compile the VGG-16 Model to vgg16.jld
function main()
	matpath = "E:/Datasets/vgg16.mat"
	weights = matread(matpath)["layers"]
	vgg16 = compile(:vgg16; weights = weights)
	JLD.save("vgg16.jld", "model", clean(vgg16))
end

main()

# Older version of vgg16 for reference purposes:

# @knet function vgg16(data; pdrop = 0.5, win = 3, pad = 1)
# 	# pool() functions have stride = window=2 MAXPOOL by default
# 	# convoulution strides are 1 by default
# 	conv1_1 = wconv(data; out = 64, window = win, padding = pad)
# 	conv1_1 = relu(conv1_1)
# 	conv1_2 = wconv(conv1_1; out = 64, window = win, padding = pad)
# 	conv1_2 = relu(conv1_2)
# 	pool1 = pool(conv1_2)
# 	conv2_1 = wconv(pool1; out = 128, window = win, padding = pad)
# 	conv2_1 = relu(conv2_1)
# 	conv2_2 = wconv(conv2_1; out = 128, window = win, padding = pad)
# 	conv2_2 = relu(conv2_2)
# 	pool2 = pool(conv2_2)
# 	conv3_1 = wconv(pool2; out = 256, window = win, padding = pad)
# 	conv3_1 = relu(conv3_1)
# 	conv3_2 = wconv(conv3_1; out = 256, window = win, padding = pad)
# 	conv3_2 = relu(conv3_2)
# 	conv3_3 = wconv(conv3_2; out = 256, window = win, padding = pad)
# 	conv3_3 = relu(conv3_3)
# 	pool3 = pool(conv3_3)
# 	conv4_1 = wconv(pool3; out = 512, window = win, padding = pad)
# 	conv4_1 = relu(conv4_1)
# 	conv4_2 = wconv(conv4_1; out = 512, window = win, padding = pad)
# 	conv4_2 = relu(conv4_2)
# 	conv4_3 = wconv(conv4_2; out = 512, window = win, padding = pad)
# 	conv4_3 = relu(conv4_3)
# 	pool4 = pool(conv4_3)
# 	conv5_1 = wconv(pool4; out = 512, window = win, padding = pad)
# 	conv5_1 = relu(conv5_1)
# 	conv5_2 = wconv(conv5_1; out = 512, window = win, padding = pad)
# 	conv5_2 = relu(conv5_2)
# 	conv5_3 = wconv(conv5_2; out = 512, window = win, padding = pad)
# 	conv5_3 = relu(conv5_3)
# 	pool5 = pool(conv5_3)
# 	fc6 = wf(pool5; out = 4096, f = :relu)
# 	fc6 = drop(fc6; pdrop = pdrop)
# 	fc7 = wf(fc6; out = 4096, f = :relu)
# 	return fc7
# end
