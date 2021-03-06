using Knet, JLD, CUDArt, JSON
using Knet: regs, getp, setp, stack_length, stack_empty!, params

function main()
	info("initializing...")
	global imageSize = 224
	global nepochs = 20
	global batchsize = 20
	lr = 0.01

	# Load VGG-16 Model
	global vgg16 = JLD.load("vgg16.jld", "model")

	model = 3

	if model == 2
		# Load the LSTM Model
		global lstm = JLD.load("lstm.jld", "lstmYTFlickr")
		# Load caption vocabulary
		global word2int = JLD.load("flickr_data.jld","word2intFlickr")
		global int2word = JLD.load("flickr_data.jld","int2wordFlickr")
		trainLSTMFlickr(lstm, lr)
		lr /= 2
	elseif model == 3
		# Load the LSTM Model
		global lstm = JLD.load("lstm.jld", "lstmYTCOCO")
		# Load caption vocabulary
		global word2int = JLD.load("coco_data.jld","word2intCOCO")
		global int2word = JLD.load("coco_data.jld","int2wordCOCO")
		global ctrn = JLD.load("coco_data.jld","ctrn_COCO")
		global cval = JLD.load("coco_data.jld","cval_COCO")
		trainLSTMCOCO(lstm, lr)
		lr /= 2
	elseif model == 4
		# Load the LSTM Model
		global lstm = JLD.load("lstm.jld", "lstmYTCOCOFlickr")
		# Load caption vocabulary
		global word2int = JLD.load("coco_data.jld","word2intCOCOFlickr")
		global int2word = JLD.load("coco_data.jld","int2wordCOCOFlickr")
		global ctrn = JLD.load("coco_data.jld","ctrn_COCOFlickr")
		global cval = JLD.load("coco_data.jld","cval_COCOFlickr")
		trainLSTMFlickr(lstm, lr)
		trainLSTMCOCO(lstm, lr)
		lr /= 2
	else
		# Load the LSTM Model
		global lstm = JLD.load("lstm.jld", "lstmYT")
		# Load caption vocabulary
		global word2int = JLD.load("youtube_data.jld","word2int")
		global int2word = JLD.load("youtube_data.jld","int2word")
	end

	trainLSTMYT(lstm, lr)

	JLD.save("lstm_trained$(model).jld", "model", clean(lstm))

	# sent = "";
	# for i = 1:27
	# 	ypred = to_host(sforw(lstm, xtrn[i,:] ))
	# 	m = findmax(ypred,1)[2] % length(int2word)
	# 	sent = string(sent, int2word[m[1]], " ")
	# end

	# print(sent)

end

function trainLSTMFlickr(lstm, lr)
	info("training with Flickr30k...")
	xtrn = JLD.load("flickr_image_data.jld", "xtrn")
	ftrn = JLD.load("flickr_data.jld", "ftrn")[:,1:size(xtrn,2)*5]

	idx = shuffle(collect(1:size(ftrn,2)))

	setp(lstm; lr = lr)
	l = zeros(2); m = zeros(2)

	for epoch = 1:nepochs
		print("epoch: $epoch\n")

		train(lstm, (xtrn, ftrn), idx, softloss; gclip = 10, losscnt = fill!(l,0), maxnorm = fill!(m,0))
		test(lstm, (xtrn, ftrn), idx, softloss)
	end

end

function trainLSTMCOCO(lstm, lr)
	info("training with COCO2014...")
	xtrn = JLD.load("coco_image_data.jld", "xtrn")
	ctrn = ctrn[:,find( x -> x <= size(xtrn,2), ctrn[1,:])]

	idx = shuffle(collect(1:size(ctrn,2)))

	setp(lstm; lr = lr)
	l = zeros(2); m = zeros(2)

	for epoch = 1:nepochs
		print("epoch: $epoch\n")

		train(lstm, (xtrn, ctrn), idx, softloss; gclip = 10, losscnt = fill!(l,0), maxnorm = fill!(m,0))
		test(lstm, (xtrn, ctrn), idx, softloss)
	end

end

function trainLSTMYT(lstm, lr)
	info("training with YouTube Video Dataset...")
	xtrn = JLD.load("youtube_data.jld","xtrn")
	ytrn = JLD.load("youtube_data.jld","ytrn")

	idx = shuffle(collect(1:size(ytrn,2)))

	setp(lstm; lr = lr)
	l = zeros(2); m = zeros(2)

	for epoch = 1:nepochs
		print("epoch: $epoch\n")

		train(lstm, (xtrn, ytrn), idx, softloss, gclip = 10, losscnt = fill!(l,0), maxnorm = fill!(m,0))
		test(lstm, (xtrn, ytrn), idx, softloss)
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
	mask[find(ybatch[word2int[""],:] .== 1)] = 0
	return mask
end

function train(f, data, idx, loss; gcheck=false, gclip=0, maxnorm=nothing, losscnt=nothing)
	info("training...")
	reset_trn!(f)
    ystack = Any[]
	(xtrn, ytrn) = data

	for i = 1:batchsize:size(ytrn,2)-batchsize
		print("batch: $(ceil(i/batchsize)) of $(ceil(size(ytrn,2)/batchsize))\n")
		flush(STDOUT)
		x = xtrn[:,collect(ytrn[1,idx[i:min(i+batchsize-1,size(ytrn,2))]])]
		s = ytrn[2:end,idx[i:min(i+batchsize-1,size(ytrn,2))]]

		words = sparse(map(Int64,ones(batchsize)),collect(1:batchsize),ones(batchsize),length(word2int),batchsize)
		for i = 1:size(s,1)
			y = map(Float64,s[i,:])
			ygold = sparse(map(Int64,collect(y)),collect(1:batchsize),ones(batchsize),length(word2int),batchsize)
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

function test(f, data, idx, loss; gcheck=false)
    info("testing...")
    sumloss = numloss = 0.0
    reset_tst!(f)
	(xtrn, ytrn) = data

	for i = 1:batchsize:size(ytrn,2)-batchsize
		print("batch: $(ceil(i/batchsize)) of $(ceil(size(ytrn,2)/batchsize))\n")
		x = xtrn[:,collect(ytrn[1,idx[i:min(i+batchsize-1,size(ytrn,2))]])]
		s = ytrn[2:end,idx[i:min(i+batchsize-1,size(ytrn,2))]]

		sent = ""
		words = sparse(map(Int64,ones(batchsize)),collect(1:batchsize),ones(batchsize),length(word2int),batchsize)
		for i = 1:size(s,1)
			y = s[i,:]
			ygold = sparse(map(Int64,collect(y)),collect(1:batchsize),ones(batchsize),length(word2int),batchsize)
			ypred = forw(f, x, words)
			words = ypred
			m = findmax(to_host(ypred),1)[2] % length(int2word)
			m[1] = (m[1] == 0 ? length(word2int) : m[1])
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
