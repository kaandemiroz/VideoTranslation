using JLD

function main()
	# # Load caption vocabulary
	vocabulary = open("lstm_data/vocabulary.txt")
	word2onehot = Dict{Any,Any}()
	int2word = Dict{Any,Any}()
	for (n, s) in enumerate(eachline(vocabulary))
		sp = sparsevec([n],[1],12594)
		word2onehot[chomp(s)] = sp
		int2word[n] = chomp(s);
	end
	# Manually add <eos> tag
	sp = sparsevec([12594],[1],12594)
	word2onehot[""] = sp
	int2word[12594] = "";
	close(vocabulary)

	yt_train = readdlm("lstm_data/yt_pooled_vgg_fc7_mean_train.txt", ',', '\n'; use_mmap = true)
	yt_test = readdlm("lstm_data/yt_pooled_vgg_fc7_mean_test.txt", ',', '\n'; use_mmap = true)
	yt_val = readdlm("lstm_data/yt_pooled_vgg_fc7_mean_val.txt", ',', '\n'; use_mmap = true)

	sents_train = readdlm("lstm_data/sents_train_lc_nopunc.txt"; use_mmap = true)
	sents_test = readdlm("lstm_data/sents_test_lc_nopunc.txt"; use_mmap = true)
	sents_val = readdlm("lstm_data/sents_val_lc_nopunc.txt"; use_mmap = true)

	xtrn = transpose(yt_train)
	xtst = transpose(yt_test)
	xval = transpose(yt_val)

	ytrn = Array(Any,1200,66)
	lastVid = 0
	index = 0
	for i = 1:size(sents_train,1)
		vid = parse(Int,lstrip(sents_train[i,1], ['v','i','d']))
		if vid != lastVid
			index = 0
		end
		index+=1
		ytrn[vid,index] = [word2onehot[string(word)] for word in sents_train[i,2:46] ]
		lastVid = vid
	end

	ytst = Array(Any,670,81)
	lastVid = 0
	index = 0
	for i = 1:size(sents_test,1)
		vid = parse(Int,lstrip(sents_test[i,1], ['v','i','d'])) - 1300
		if vid != lastVid
			index = 0
		end
		index+=1
		ytst[vid,index] = [word2onehot[string(word)] for word in sents_test[i,2:42] ]
		lastVid = vid
	end

	yval = Array(Any,100,62)
	lastVid = 0
	index = 0
	for i = 1:size(sents_val,1)
		vid = parse(Int,lstrip(sents_val[i,1], ['v','i','d'])) - 1200
		if vid != lastVid
			index = 0
		end
		index+=1
		yval[vid,index] = [word2onehot[string(word)] for word in sents_val[i,2:28] ]
		lastVid = vid
	end

	JLD.save("lstm_data.jld", "word2onehot", word2onehot, "int2word", int2word, "xtrn", xtrn, "xtst", xtst, "xval", xval, "ytrn", ytrn, "ytst", ytst, "yval", yval)

end

main()
