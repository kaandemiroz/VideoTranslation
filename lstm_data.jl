using JLD

function main()

	vocab_path = "lstm_data/vocabulary.txt"
	xtrn_path = "lstm_data/yt_pooled_vgg_fc7_mean_train.txt"
	xtst_path = "lstm_data/yt_pooled_vgg_fc7_mean_test.txt"
	xval_path = "lstm_data/yt_pooled_vgg_fc7_mean_val.txt"
	ytrn_path = "lstm_data/sents_train_lc_nopunc.txt"
	ytst_path = "lstm_data/sents_test_lc_nopunc.txt"
	yval_path = "lstm_data/sents_val_lc_nopunc.txt"

	sents_trn = readdlm(ytrn_path; use_mmap = true)
	sents_tst = readdlm(ytst_path; use_mmap = true)
	sents_val = readdlm(yval_path; use_mmap = true)

	# Load caption vocabulary
	vocabulary = open(vocab_path)
	word2int = Dict{Any,Int32}()
	int2word = Array(Any,12594)
	for (n, s) in enumerate(eachline(vocabulary))
		word2int[chomp(s)] = n
		int2word[n] = chomp(s)
	end
	close(vocabulary)
	# Manually add <eos> tag
	word2int[""] = 12594
	int2word[12594] = "";

	# Read x data easily
	xtrn = transpose( readdlm(xtrn_path, ',', '\n'; use_mmap = true) )
	xtst = transpose( readdlm(xtst_path, ',', '\n'; use_mmap = true) )
	xval = transpose( readdlm(xval_path, ',', '\n'; use_mmap = true) )

	# Parse y data by using the dictionary
	ytrn = Array(Any,1200)
	lastVid = 1
	data = Any[]
	for i = 1:size(sents_trn,1)
		vid = parse(Int,lstrip(sents_trn[i,1], ['v','i','d']))
		push!(data, [word2int[string(word)] for word in sents_trn[i,2:46] ] )
		if vid != lastVid
			ytrn[lastVid] = data
			data = Any[]
		end
		lastVid = vid
	end
	ytrn[lastVid] = data

	# ytrn = Array(Any,1200)
	# open(ytrn_path) do f
	# 	for l in eachline(f)
	# 		data = Int32[]
	# 		s = split(l,'\t')
	# 		vid = parse(Int,lstrip(s[1], ['v','i','d']))
	# 		for w in split(s[2])
	# 			push!(data, word2int[w])
	# 		end
	# 		push!(data,word2int["<eos>"])
	# 		ytrn[vid] = data
	# 	end
	# end

	ytst = Array(Any,670)
	lastVid = 1
	data = Any[]
	for i = 1:size(sents_tst,1)
		vid = parse(Int,lstrip(sents_tst[i,1], ['v','i','d'])) - 1300
		push!(data, [word2int[string(word)] for word in sents_tst[i,2:42] ] )
		if vid != lastVid
			ytst[lastVid] = data
			data = Any[]
		end
		lastVid = vid
	end
	ytst[lastVid] = data

	# ytst = Array(Any,670)
	# open(ytst_path) do f
	# 	for l in eachline(f)
	# 		data = Int32[]
	# 		s = split(l,'\t')
	# 		vid = parse(Int,lstrip(s[1], ['v','i','d'])) - 1300
	# 		for w in split(s[2])
	# 			push!(data, word2int[w])
	# 		end
	# 		push!(data,word2int["<eos>"])
	# 		ytst[vid] = data
	# 	end
	# end

	yval = Array(Any,100)
	lastVid = 1
	data = Any[]
	for i = 1:size(sents_val,1)
		vid = parse(Int,lstrip(sents_val[i,1], ['v','i','d'])) - 1200
		push!(data, [word2int[string(word)] for word in sents_val[i,2:28] ])
		if vid != lastVid
			yval[lastVid] = data
			data = Any[]
		end
		lastVid = vid
	end
	yval[lastVid] = data

	# yval = Array(Any,100)
	# open(yval_path) do f
	# 	for l in eachline(f)
	# 		data = Int32[]
	# 		s = split(l,'\t')
	# 		vid = parse(Int,lstrip(s[1], ['v','i','d'])) - 1200
	# 		for w in split(s[2])
	# 			push!(data, word2int[w])
	# 		end
	# 		push!(data,word2int["<eos>"])
	# 		yval[vid] = data
	# 	end
	# end

	JLD.save("lstm_data.jld", "word2int", word2int, "int2word", int2word, "xtrn", xtrn, "xtst", xtst, "xval", xval, "ytrn", ytrn, "ytst", ytst, "yval", yval )

end

main()
