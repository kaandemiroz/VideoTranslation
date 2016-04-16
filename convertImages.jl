using Knet
using Images
using Colors

function main()
	#============================
	MNIST.xtrn, MNIST.ytrn
	MNIST.xtst, MNIST.ytst
	============================#
	path = "E:/Datasets/Flickr30k/flickr30k-images/"
	outpath = "E:/Datasets/Flickr30k/images.txt"
	key = ".jpg"
	imageSize = 224
	batchsize = 10

	outfile = open(outpath, "w")

	image_names = filter(x->contains(x,key), readdir(path))

	#Intialize x matrix
 	x = zeros( imageSize, imageSize, 3, batchsize )

	index = 1
 	for image_name in image_names
		print(index)
  	#Read image file
  	imgPath = "$(path)$(image_name)"
  	img = load(imgPath)
		img = Images.imresize(img, (imageSize, imageSize))

		#Convert img to float values
		r = map(Float32,red(img))
		g = map(Float32,green(img))
		b = map(Float32,blue(img))

		x[:,:,1,index] = r
		x[:,:,2,index] = g
		x[:,:,3,index] = b

		#Transform image matrix to a vector and store
		#it in data matrix
		r = reshape(r, 1, imageSize^2)
		g = reshape(g, 1, imageSize^2)
		b = reshape(b, 1, imageSize^2)
		# write(outfile,"$(image_name)\r\n")
		# write(outfile,"$(r)\r\n")
		# write(outfile,"$(g)\r\n")
		# write(outfile,"$(b)\r\n")


		index = index + 1
		if(index>10)
			break
		end
	 end
	 close(outfile)


end

function main2()
	#============================
	MNIST.xtrn, MNIST.ytrn
	MNIST.xtst, MNIST.ytst
	============================#
	path = "E:/Datasets/Flickr30k/flickr30k-images/"
	outpath = "E:/Datasets/Flickr30k/images.txt"
	key = ".jpg"
	imageSize = 256

	outfile = open(outpath, "w")

	img = load("$(path)65567.jpg")
	img = Images.imresize(img, (imageSize, imageSize))
	r = float32(red(img))
	g = float32(green(img))
	b = float32(blue(img))
	r = reshape(r, 1, imageSize^2)
	g = reshape(g, 1, imageSize^2)
	b = reshape(b, 1, imageSize^2)
	write(outfile,"$r")
	write(outfile,"$g")
	write(outfile,"$b")
	close(outfile)

end

@knet function softmax(x)
	# Old Code
	#w = par(init=Gaussian(0,0.001), dims=(10,784))
	#b = par(init=Constant(0), dims=(10,1))
	#y = w * x + b
	#return soft(y)

	return wbf(x; out = 10, winit = Gaussian(0,0.001), f = :soft)
end

@knet function mlp30(x)
	# Old Code
	#w1 = par(init=Gaussian(0,0.001), dims=(30,784))
	#b1 = par(init=Constant(0), dims=(30,1))
	#y1 = relu(w1 * x + b1)
	#w2 = par(init=Gaussian(0,0.001), dims=(10,30))
	#b2 = par(init=Constant(0), dims=(10,1))
	#return soft(w2 * y1 + b2)

	y1 = wbf(x; out = 30, winit = Gaussian(0,0.001), f = :relu)
	return wbf(y1; out = 10, winit = Gaussian(0,0.001), f = :soft)
end

@knet function mlp100(x)
	# Old Code
	#w1 = par(init=Gaussian(0,0.001), dims=(100,784))
	#b1 = par(init=Constant(0), dims=(100,1))
	#y1 = relu(w1 * x + b1)
	#w2 = par(init=Gaussian(0,0.001), dims=(10,100))
	#b2 = par(init=Constant(0), dims=(10,1))
	#return soft(w2 * y1 + b2)

	y1 = wbf(x; out = 100, winit = Gaussian(0,0.001), f = :relu)
	return wbf(y1; out = 10, winit = Gaussian(0,0.001), f = :soft)
end

@knet function mlp2_30(x)
	# Old Code
	#w1 = par(init=Gaussian(0,0.001), dims=(30,784))
	#b1 = par(init=Constant(0), dims=(30,1))
	#y1 = relu(w1 * x + b1)
	#w2 = par(init=Gaussian(0,0.001), dims=(30,30))
	#b2 = par(init=Constant(0), dims=(30,1))
	#y2 = relu(w2 * y1 + b2)
	#w3 = par(init=Gaussian(0,0.001), dims=(10,30))
	#b3 = par(init=Constant(0), dims=(10,1))
	#return soft(w3 * y2 + b3)

	y1 = wbf(x; out = 30, winit = Gaussian(0,0.001), f = :relu)
	y2 = wbf(y1; out = 30, winit = Gaussian(0,0.001), f = :relu)
	return wbf(y2; out = 10, winit = Gaussian(0,0.001), f = :soft)
end

@knet function cnn(x)
	y = cbfp(x; out = 3, f = :sigm, cwindow = 5, pwindow = 2)
	return wbf(y; out = 10, winit = Gaussian(0,0.001), f = :soft)
end

@knet function mysmallest(x)
	y1 = wbf(x; out = 35, winit = Gaussian(0,0.001), f = :tanh)
	y2 = wbf(y1; out = 25, winit = Gaussian(0,0.001), f = :relu)
	return wbf(y2; out = 10, winit = Gaussian(0,0.001), f = :soft)
end

function train(f, data, loss)
	for (x,y) in data
		forw(f, x)
		back(f, y, loss)
		update!(f)
	end
end

function test(f, data, loss)
	sumloss = numloss = 0
	for (x,ygold) in data
		ypred = forw(f, x)
		sumloss += loss(ypred, ygold)
		numloss += 1
	end
	sumloss / numloss
end

main()
