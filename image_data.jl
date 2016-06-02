using Knet, JLD, Images, Colors, CUDArt

function main()
	global imageSize = 224
	global batchsize = 50

	# Load VGG-16 Model
	global vgg16 = JLD.load("vgg16.jld", "model")

	ftrn_desc = JLD.load("flickr_data.jld","ftrn")
	ftrn_names = JLD.load("flickr_data.jld","ftrn_names")
	info("Forwarding Flickr30k training images to VGG-16...")
	xtrn = parseFlickrImages(ftrn_desc, ftrn_names)

	fval_desc = JLD.load("flickr_data.jld","fval")
	fval_names = JLD.load("flickr_data.jld","fval_names")
	info("Forwarding Flickr30k validation images to VGG-16...")
	xval = parseFlickrImages(fval_desc, fval_names)

	info("Saving Flickr30k Data...")
	JLD.save(	"flickr_image_data.jld",
				"xtrn", xtrn,
				"xval", xval	)


	ctrn_path = "COCO2014/train2014/"
	ctrn_names = JLD.load("coco_data.jld","ctrn_names")
	info("Forwarding COCO2014 training images to VGG-16...")
	xtrn = parseCOCOImages(ctrn_path, ctrn_names)

	cval_path = "COCO2014/val2014/"
	cval_names = JLD.load("coco_data.jld","cval_names")
	info("Forwarding COCO2014 validation images to VGG-16...")
	xval = parseCOCOImages(cval_path, cval_names)

	info("Saving COCO2014 Data...")
	JLD.save(	"coco_image_data.jld",
				"xtrn", xtrn,
				"xval", xval	)

end

function parseFlickrImages(desc, names)

	fpath = "Flickr30k/flickr30k-images/"
	numimages = div( size(desc, 2), 5 )
	numbatches = div( numimages, batchsize )
	result = zeros(4096, numimages)

	info("$(numimages) images, $(numbatches) batches, batchsize = $(batchsize)")

	for batch = 0:numbatches-1
		#Initialize x matrix
		info("batch: $(batch+1)")
		x = zeros( imageSize, imageSize, 3, batchsize )

		for i = 1:batchsize
			index = batch * batchsize + i * 5
			#Read image file
			imgPath = "$(fpath)$(names[desc[1,index]]).jpg"
			img = load(imgPath)
			#Resize image to 224x224
			img = Images.imresize(img, (imageSize, imageSize))

			#Convert img to float values for RGB
			r = map(Float32, red(img))
			g = map(Float32, green(img))
			b = map(Float32, blue(img))

			x[:,:,1,i] = r
			x[:,:,2,i] = g
			x[:,:,3,i] = b
		end

		result[:,batch * batchsize + 1:(batch+1) * batchsize] = reshape(to_host(forw(vgg16, x)), (4096, batchsize))

	end

	return result
end

function parseCOCOImages(path, names)

	numimages = length(names)
	numbatches = div( numimages, batchsize )
	result = zeros(4096, numbatches * batchsize)

	info("$(numimages) images, $(numbatches) batches, batchsize = $(batchsize)")

	key = collect(keys(names))

	for batch = 0:numbatches-1
		#Initialize x matrix
		info("batch: $(batch+1)")
		x = zeros( imageSize, imageSize, 3, batchsize )

		for i = 1:batchsize
			index = batch * batchsize + i
			#Read image file
			imgPath = "$(path)$(names[key[index]])"
			img = load(imgPath)
			#Resize image to 224x224
			img = Images.imresize(img, (imageSize, imageSize))

			#Convert img to float values for RGB
			r = map(Float32, red(img))
			g = map(Float32, green(img))
			b = map(Float32, blue(img))

			x[:,:,1,i] = r
			x[:,:,2,i] = g
			x[:,:,3,i] = b
		end

		result[:,batch * batchsize + 1:(batch+1) * batchsize] = reshape(to_host(forw(vgg16, x)), (4096, batchsize))

	end

	return result
end

main()
