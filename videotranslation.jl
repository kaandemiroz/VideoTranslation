using Knet, JLD, Images, Colors, CUDArt
using Knet: regs, getp, setp, stack_length, stack_empty!, params

function main()
	info("initializing...")
	fpath = "flickr30k-images/"
	cpath = "train2014/"
	key = ".jpg"
	imageSize = 224

	nepochs = 1
	batchsize = 10

	# image_names = filter(x->contains(x,key), readdir(fpath))
	# numbatches = 1#div(size(image_names,1), batchsize)

	# Load VGG-16 Model
	# vgg16 = JLD.load("vgg16.jld", "model")
	# Load the LSTM Model
	lstm = JLD.load("lstm.jld", "model")
	# # Load caption vocabulary
	word2int = JLD.load("lstm_data.jld","word2int")
	int2word = JLD.load("lstm_data.jld","int2word")
	xval = JLD.load("lstm_data.jld","xval")
	yval = JLD.load("lstm_data.jld","yval")

	setp(lstm; lr = 0.01)
	l = zeros(2); m = zeros(2)

	for epoch = 1:nepochs
		# for i = 1:batchsize:length(yval)
		# 	train(lstm, (xval[:,i:i+batchsize-1], yval[i:i+batchsize-1]), softloss; gclip = 10, losscnt = fill!(l,0), maxnorm = fill!(m,0))
		# 	test(lstm, (xval[:,i:i+batchsize-1], yval[i:i+batchsize-1]), softloss, int2word)
		# end
		for i = 1:10#length(yval)
			train(lstm, (xval[:,i], yval[i]), softloss; gclip = 10, losscnt = fill!(l,0), maxnorm = fill!(m,0))
			test(lstm, (xval[:,i], yval[i]), softloss, int2word)
		end
	end

	# sent = "";
	# for i = 1:27
	# 	ypred = to_host(sforw(lstm, xval[i,:] ))
	# 	m = findmax(ypred,1)[2] % length(int2word)
	# 	sent = string(sent, int2word[m[1]], " ")
	# end

	# print(sent)

	# for epoch = 1:nepochs
	#
	# 	for batch = 0:numbatches-1
	# 		#Initialize x matrix
	# 		x = zeros( imageSize, imageSize, 3, batchsize )
	#
	# 		for i = 1:batchsize
	# 			index = batch * batchsize + i
	# 			print("$(index)\n")
	# 			#Read image file
	# 			imgPath = "$(fpath)$(image_names[index])"
	# 			img = load(imgPath)
	# 			#Resize image to 224x224
	# 			img = Images.imresize(img, (imageSize, imageSize))
	#
	# 			#Convert img to float values for RGB
	# 			r = map(Float32,red(img))
	# 			g = map(Float32,green(img))
	# 			b = map(Float32,blue(img))
	#
	# 			x[:,:,1,i] = r
	# 			x[:,:,2,i] = g
	# 			x[:,:,3,i] = b
	# 		end
	#
	# 		y = forw(vgg16, x)
	#
	# 	end
	#
	# end

end

reset_trn!(f;o...)=reset!(f, keepstate=true)

function reset_tst!(f; keepstate=false)
    for p in regs(f)
        p.out = keepstate && isdefined(p,:out0) ? p.out0 : nothing
    end
end

function seqbatch(seq, dict, batchsize)
    data = Any[]
    T = div(length(seq), batchsize)
    for t=1:T
        d=zeros(Float32, length(dict), batchsize)
        for b=1:batchsize
            c = dict[seq[t + (b-1) * T]]
            d[c,b] = 1
        end
        push!(data, d)
    end
    return data
end

function train(f, data, loss; gcheck=false, gclip=0, maxnorm=nothing, losscnt=nothing)
	info("training...")
	reset_trn!(f)
    ystack = Any[]
	x = data[1]
    for s in data[2]
		for y in s
			if y != 12594
				ygold = sparsevec([y],[map(Float64,1)],12594)
		        ypred = sforw(f, x)
		        # Knet.netprint(f); error(:ok)
		        losscnt != nothing && (losscnt[1] += loss(ypred, ygold); losscnt[2] += 1)
		        push!(ystack, copy(ygold))
			else
				while !isempty(ystack)
					ygold = pop!(ystack)
					sback(f, ygold, loss)
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
				continue
			end
		end
    end
    # losscnt[1]/losscnt[2]       # this will give per-token loss, should we do per-sequence instead?
end

function test(f, data, loss, int2word; gcheck=false)
    info("testing...")
    sumloss = numloss = 0.0
    reset_tst!(f)
	x = data[1]
    for s in data[2]
		sent = ""
		for y in s
			if y != 12594
				ygold = sparsevec([y],[map(Float64,1)],12594)
		        ypred = sforw(f, x)
				m = findmax(to_host(ypred),1)[2] % length(int2word) + 1
				sent = string(sent, int2word[m[1]], " ")
		        # @show (hash(x),hash(ygold),vecnorm0(ypred))
				l = loss(ypred,ygold)
		        sumloss += l
		        numloss += 1
			else
				gcheck && return sumloss
				reset_tst!(f; keepstate=true)
				continue
			end
		end
		@show sent
    end
	@show sumloss/numloss
    return sumloss/numloss
end

main()
