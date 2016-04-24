using Knet
using JLD
using Images
using Colors

function main()
	fpath = "E:/Datasets/Flickr30k/flickr30k-images/"
	cpath = "E:/Datasets/COCO2014/train2014/"
	key = ".jpg"
	imageSize = 224

	nepochs = 1
	batchsize = 10

	image_names = filter(x->contains(x,key), readdir(fpath))
	numbatches = 10#div(size(image_names,1), batchsize)

	# Load VGG-16 Model
	# vgg16 = JLD.load("vgg16.jld", "model")
	# Load the LSTM Model
	lstm = JLD.load("lstm.jld", "model")
	# Load caption vocabulary
	vocabulary = open("lstm_data/vocabulary.txt")
	word2int = Dict{Any,Int32}()
	for (n, s) in enumerate(eachline(vocabulary))
		word2int[chomp(s)] = n
	end
	close(vocabulary)

	yt_train = readdlm("lstm_data/yt_pooled_vgg_fc7_mean_train.txt", ',', '\n'; use_mmap = true)
	yt_test = readdlm("lstm_data/yt_pooled_vgg_fc7_mean_test.txt", ',', '\n'; use_mmap = true)
	yt_val = readdlm("lstm_data/yt_pooled_vgg_fc7_mean_val.txt", ',', '\n'; use_mmap = true)

	sents_train = readdlm("lstm_data/sents_train_lc_nopunc.txt", '\t', '\n'; use_mmap = true)
	sents_test = readdlm("lstm_data/sents_test_lc_nopunc.txt", '\t', '\n'; use_mmap = true)
	sents_val = readdlm("lstm_data/sents_val_lc_nopunc.txt", '\t', '\n'; use_mmap = true)

	sforw(lstm, yt_val[1,:])

	# output = generate(lstm,x)

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
	#
	# 	end
	#
	# end

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
    reset!(f, keepstate = true)
    ystack = Any[]
    for item in data
        if item != nothing
            (x,ygold) = item
            ypred = sforw(f, x)
            # Knet.netprint(f); error(:ok)
            losscnt != nothing && (losscnt[1] += loss(ypred, ygold); losscnt[2] += 1)
            push!(ystack, copy(ygold))
        else                    # end of sequence
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
            reset_trn!(f; keepstate=true)
        end
    end
    # losscnt[1]/losscnt[2]       # this will give per-token loss, should we do per-sequence instead?
end

function test(f, data, loss; gcheck=false)
    #info("testing")
    sumloss = numloss = 0.0
    reset_tst!(f)
    for item in data
        if item != nothing
            (x,ygold) = item
            ypred = forw(f, x)
            # @show (hash(x),hash(ygold),vecnorm0(ypred))
            sumloss += loss(ypred, ygold)
            numloss += 1
        else
            gcheck && return sumloss
            reset_tst!(f; keepstate=true)
        end
    end
    return sumloss/numloss
end

main()
