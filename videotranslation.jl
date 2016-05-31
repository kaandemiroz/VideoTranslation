using Knet, JLD, Images, Colors, CUDArt, JSON
using Knet: regs, getp, setp, stack_length, stack_empty!, params

function main()
	info("initializing...")
	imageSize = 224
	nepochs = 20
	batchsize = 20
	lr = 0.01

	# Load VGG-16 Model
	global vgg16 = JLD.load("vgg16.jld", "model")

	model = 2

	if model == 2
		# Load the LSTM Model
		global lstm = JLD.load("lstm.jld", "lstmYTFlickr")
		# Load caption vocabulary
		global word2int = JLD.load("flickr_data.jld","word2intFlickr")
		global int2word = JLD.load("flickr_data.jld","int2wordFlickr")
		trainLSTMFlickr(lstm, lr, imageSize, batchsize)
		lr /= 2
	elseif model == 3
		# Load the LSTM Model
		global lstm = JLD.load("lstm.jld", "lstmYTCOCO")
		# Load caption vocabulary
		global word2int = JLD.load("coco_data.jld","word2intCOCO")
		global int2word = JLD.load("coco_data.jld","int2wordCOCO")
		global ctrn_desc = JLD.load("coco_data.jld","ctrn_descCOCO")
		global cval_desc = JLD.load("coco_data.jld","cval_descCOCO")
		trainLSTMCOCO(lstm, lr, imageSize, batchsize)
		lr /= 2
	elseif model == 4
		# Load the LSTM Model
		global lstm = JLD.load("lstm.jld", "lstmYTCOCOFlickr")
		# Load caption vocabulary
		global word2int = JLD.load("coco_data.jld","word2intCOCOFlickr")
		global int2word = JLD.load("coco_data.jld","int2wordCOCOFlickr")
		global ctrn_desc = JLD.load("coco_data.jld","ctrn_descCOCOFlickr")
		global cval_desc = JLD.load("coco_data.jld","cval_descCOCOFlickr")
		trainLSTMFlickr(lstm, lr, imageSize, batchsize)
		trainLSTMCOCO(lstm, lr, imageSize, batchsize)
		lr /= 2
	else
		# Load the LSTM Model
		global lstm = JLD.load("lstm.jld", "lstmYT")
		# Load caption vocabulary
		global word2int = JLD.load("youtube_data.jld","word2int")
		global int2word = JLD.load("youtube_data.jld","int2word")
	end

	trainLSTMYT(lstm, lr, nepochs, batchsize)

	JLD.save("lstm_trained2.jld", "model", clean(lstm))

	# sent = "";
	# for i = 1:27
	# 	ypred = to_host(sforw(lstm, xtrn[i,:] ))
	# 	m = findmax(ypred,1)[2] % length(int2word)
	# 	sent = string(sent, int2word[m[1]], " ")
	# end

	# print(sent)

end

function trainLSTMFlickr(lstm, lr, imageSize, batchsize)
	info("training with Flickr30k...")
	fpath = "Flickr30k/flickr30k-images/"
	fdesc = JLD.load("flickr_data.jld","fdesc")
	key = ".jpg"
	val = rand(1:length(fdesc),1000)

	image_names = filter(x->contains(x,key), readdir(fpath))
	numbatches = 1#div(size(image_names,1), batchsize)

	for batch = 0:numbatches-1
		#Initialize x matrix
		x = zeros( imageSize, imageSize, 3, batchsize )

		for i = 1:batchsize
			index = batch * batchsize + i
			info("batch: $(index)")
			#Read image file
			imgName = image_names[val[index]]
			imgPath = "$(fpath)$(imgName)"
			img = load(imgPath)
			#Resize image to 224x224
			img = Images.imresize(img, (imageSize, imageSize))

			#Convert img to float values for RGB
			r = map(Float32,red(img))
			g = map(Float32,green(img))
			b = map(Float32,blue(img))

			x[:,:,1,i] = r
			x[:,:,2,i] = g
			x[:,:,3,i] = b

			setp(lstm; lr = lr)
			l = zeros(2); m = zeros(2)

			xtrn = reshape(to_host(forw(vgg16, x)), (4096, batchsize))
			train(lstm, (xtrn[:,1], fdesc[imgName]), softloss; gclip = 10, losscnt = fill!(l,0), maxnorm = fill!(m,0))
			test(lstm, (xtrn[:,1], fdesc[imgName]), softloss, int2word)
		end

	end

end

function transLSTMCOCO(lstm, lr, imageSize, batchsize)
	info("training with COCO2014...")
	cpath = "COCO2014/val2014/"
	key = ".jpg"
	val = rand(1:length(fdesc),1000)

	image_names = filter(x->contains(x,key), readdir(cpath))
	numbatches = 1#div(size(image_names,1), batchsize)

	for batch = 0:numbatches-1
		#Initialize x matrix
		x = zeros( imageSize, imageSize, 3, batchsize )

		for i = 1:batchsize
			index = batch * batchsize + i
			info("batch: $(index)")
			#Read image file
			imgName = image_names[val[index]]
			imgPath = "$(fpath)$(imgName)"
			img = load(imgPath)
			#Resize image to 224x224
			img = Images.imresize(img, (imageSize, imageSize))

			#Convert img to float values for RGB
			r = map(Float32,red(img))
			g = map(Float32,green(img))
			b = map(Float32,blue(img))

			x[:,:,1,i] = r
			x[:,:,2,i] = g
			x[:,:,3,i] = b

			setp(lstm; lr = lr)
			l = zeros(2); m = zeros(2)

			xtrn = reshape(to_host(forw(vgg16, x)), (4096, batchsize))
			train(lstm, (xtrn[:,1], cval_desc[imgName]), softloss; gclip = 10, losscnt = fill!(l,0), maxnorm = fill!(m,0))
			test(lstm, (xtrn[:,1], cval_desc[imgName]), softloss, int2word)
		end

	end

end

function trainLSTMYT(lstm, lr, nepochs, batchsize)
	xtrn = JLD.load("youtube_data.jld","xtrn")
	ytrn = JLD.load("youtube_data.jld","ytrn")

	idx = shuffle(collect(1:size(ytrn,2)))

	setp(lstm; lr = lr)
	l = zeros(2); m = zeros(2)

	for epoch = 1:nepochs
		print("epoch: $epoch\n")
		train(lstm, (xtrn, ytrn), idx, batchsize, softloss, gclip = 10, losscnt = fill!(l,0), maxnorm = fill!(m,0))
		test(lstm, (xtrn, ytrn), idx, batchsize, softloss, int2word)
	end

end

reset_trn!(f;o...)=reset!(f, keepstate=true)

function reset_tst!(f; keepstate=false)
    for p in regs(f)
        p.out = keepstate && isdefined(p,:out0) ? p.out0 : nothing
    end
end

function mask(ybatch)
	mask = ones(Cuchar, size(ybatch,2))
	mask[find(ybatch[12594,:] .== 1)] = 0
	return mask
end

function train(f, data, idx, batchsize, loss; gcheck=false, gclip=0, maxnorm=nothing, losscnt=nothing)
	info("training...")
	reset_trn!(f)
    ystack = Any[]
	(xtrn, ytrn) = data

	for i = 1:batchsize:size(ytrn,2)-batchsize
		print("batch: $(ceil(i/batchsize)) of $(ceil(size(ytrn,2)/batchsize))\n")
		flush(STDOUT)
		x = xtrn[:,collect(ytrn[1,idx[i:min(i+batchsize-1,size(ytrn,2))]])]
		s = ytrn[2:end,idx[i:min(i+batchsize-1,size(ytrn,2))]]

		words = sparse(map(Int64,ones(batchsize)),collect(1:batchsize),ones(batchsize),12594,batchsize)
		for i = 1:size(s,1)
			y = map(Float64,s[i,:])
			ygold = sparse(map(Int64,collect(y)),collect(1:batchsize),ones(batchsize),12594,batchsize)
			ypred = sforw(f, x, words)
			words = ygold
			# Knet.netprint(f); error(:ok)
			losscnt != nothing && (losscnt[1] += loss(ypred, ygold; mask = mask(ygold)); losscnt[2] += 1)
			push!(ystack, copy(ygold))
	    end

		while !isempty(ystack)
			ygold = pop!(ystack)
			sback(f, ygold, loss; mask = mask(ygold))
		end
		#error(:ok)
		gcheck && break # return losscnt[1] leave the loss calculation to test # the parameter gradients are cumulative over the whole sequence
		g = (gclip > 0 || maxnorm!=nothing ? gnorm(f) : 0)
		# global _update_dbg; _update_dbg +=1; _update_dbg > 1 && error(:ok)
		update!(f; gscale=(g > gclip > 0 ? gclip/g : 1))
		if maxnorm != nothing
			w=wnorm(f)
			w > maxnorm[1] && (maxnorm[1]=w)
			g > maxnorm[2] && (maxnorm[2]=g)
		end
		reset_trn!(f)
	end
	# losscnt[1]/losscnt[2]       # this will give per-token loss, should we do per-sequence instead?
end

function test(f, data, idx, batchsize, loss, int2word; gcheck=false)
    info("testing...")
    sumloss = numloss = 0.0
    reset_tst!(f)
	(xtrn, ytrn) = data

	for i = 1:batchsize:size(ytrn,2)-batchsize
		print("batch: $(ceil(i/batchsize)) of $(ceil(size(ytrn,2)/batchsize))\n")
		x = xtrn[:,collect(ytrn[1,idx[i:min(i+batchsize-1,size(ytrn,2))]])]
		s = ytrn[2:end,idx[i:min(i+batchsize-1,size(ytrn,2))]]

		sent = ""
		words = sparse(map(Int64,ones(batchsize)),collect(1:batchsize),ones(batchsize),12594,batchsize)
		for i = 1:size(s,1)
			y = s[i,:]
			ygold = sparse(map(Int64,collect(y)),collect(1:batchsize),ones(batchsize),12594,batchsize)
			ypred = forw(f, x, words)
			words = ypred
			m = findmax(to_host(ypred),1)[2] % length(int2word)
			m[1] = (m[1] == 0 ? 12594 : m[1])
			sent = string(sent, int2word[m[1]], " ")
			# @show (hash(x),hash(ygold),vecnorm0(ypred))
			l = loss(ypred,ygold; mask = mask(ygold))
			sumloss += l
			numloss += 1
	    end
		gcheck && return sumloss
		reset_tst!(f; keepstate=true)
		print("sent = $sent\n")
		flush(STDOUT)
	end

	print("sumloss / numloss = $(sumloss/numloss)\n")
    return sumloss/numloss
end

main()
