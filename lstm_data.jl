using JLD, JSON

function main()

	vocab_path = "lstm_data/vocabulary.txt"
	xtrn_path = "lstm_data/yt_pooled_vgg_fc7_mean_train.txt"
	xtst_path = "lstm_data/yt_pooled_vgg_fc7_mean_test.txt"
	xval_path = "lstm_data/yt_pooled_vgg_fc7_mean_val.txt"
	ytrn_path = "lstm_data/sents_train_lc_nopunc.txt"
	ytst_path = "lstm_data/sents_test_lc_nopunc.txt"
	yval_path = "lstm_data/sents_val_lc_nopunc.txt"
	fdesc_path = "lstm_data/results_20130124.token"

	info("Preparing YouTube Vocabulary...")

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

	info("Preparing YouTube Data...")

	# Read x data easily
	xtrn = transpose( readdlm(xtrn_path, ',', '\n'; use_mmap = true) )
	xtst = transpose( readdlm(xtst_path, ',', '\n'; use_mmap = true) )
	xval = transpose( readdlm(xval_path, ',', '\n'; use_mmap = true) )

	# Parse y data by using the dictionary
	sents_trn = readdlm(ytrn_path; use_mmap = true)
	sents_tst = readdlm(ytst_path; use_mmap = true)
	sents_val = readdlm(yval_path; use_mmap = true)
	sents_fdesc = readdlm(fdesc_path; use_mmap = true)

	info("ytrn...")
	ytrn = Array( Int32, size(sents_trn,2), size(sents_trn,1) )
	for i = 1:size(ytrn,2)
		ytrn[1,i] = parse(Int,lstrip(sents_trn[i,1], ['v','i','d']))
		ytrn[2:end,i] = [word2int[string(word)] for word in sents_trn[i,2:end]]
	end

	info("ytst...")
	ytst = Array( Int32, size(sents_tst,2), size(sents_tst,1) )
	for i = 1:size(ytst,2)
		ytst[1,i] = parse(Int,lstrip(sents_tst[i,1], ['v','i','d'])) - 1300
		ytst[2:end,i] = [word2int[string(word)] for word in sents_tst[i,2:end]]
	end

	info("yval...")
	yval = Array( Int32, size(sents_val,2), size(sents_val,1) )
	for i = 1:size(yval,2)
		yval[1,i] = parse(Int,lstrip(sents_val[i,1], ['v','i','d'])) - 1200
		yval[2:end,i] = [word2int[string(word)] for word in sents_val[i,2:end]]
	end

	info("Preparing Flickr30k Data & Vocabulary...")

	word2intFlickr = copy(word2int)
	ftrn_names = Array( Int64, length(find(x -> x .== 0, sents_fdesc[:,2])) - 1000 )
	ftrn = Array( Int64, size(sents_fdesc,2) - 1, size(sents_fdesc,1) - 5000 )
	fval_names = Array( Int64, 1000 )
	fval = Array( Int64, size(sents_fdesc,2) - 1, 5000 )
	index = 0
	for i = 1:size(sents_fdesc,1)
		if i > 5000
			if sents_fdesc[i,2] == 0
				index += 1
				ftrn_names[index - 1000] = parse(Int64,rstrip(sents_fdesc[i,1],('.','j','p','g')))
			end
			ftrn[1,i-5000] = index - 1000
			ftrn[2:end,i-5000] = [get!(word2intFlickr, lowercase(string(word)), 1+length(word2intFlickr)) for word in sents_fdesc[i,3:end]]
		else
			if sents_fdesc[i,2] == 0
				index += 1
				fval_names[index] = parse(Int64,rstrip(sents_fdesc[i,1],('.','j','p','g')))
			end
			fval[1,i] = index
			fval[2:end,i] = [get!(word2intFlickr, lowercase(string(word)), 1+length(word2intFlickr)) for word in sents_fdesc[i,3:end]]
		end
	end

	int2wordFlickr = Array(Any,length(word2intFlickr))
	for m in keys(word2intFlickr)
		int2wordFlickr[word2intFlickr[m]] = m
	end

	info("Reading COCO2014 JSON...")

	ctrn_json = JSON.parsefile("lstm_data/captions_train2014.json"; use_mmap=true)
	cval_json = JSON.parsefile("lstm_data/captions_val2014.json"; use_mmap=true)

	info("Preparing COCO2014 Data & Vocabulary...")

	ctrn_names = Dict{Int32,Any}()
	for json in ctrn_json["images"]
		get!(ctrn_names, json["id"], json["file_name"])
	end
	ctrn_keys = Dict{Int32,Int32}()
	for key in collect(keys(ctrn_names))
		get!(ctrn_keys, key, 1 + length(ctrn_keys))
	end

	cval_names = Dict{Int32,Any}()
	for json in cval_json["images"]
		get!(cval_names, json["id"], json["file_name"])
	end
	cval_keys = Dict{Int32,Int32}()
	for key in collect(keys(cval_names))
		get!(cval_keys, key, 1 + length(cval_keys))
	end

	punc = Set(".,';:!?()[]{}<>\"\'")

	info("Parsing COCO2014 Training Captions...")

	word2intCOCO = copy(word2int)
	word2intCOCOFlickr = copy(word2intFlickr)
	dataCOCO = Any[]
	dataCOCOFlickr = Any[]
	for json in ctrn_json["annotations"]
		push!( dataCOCO, [ctrn_keys[json["image_id"]] transpose( [get!(word2intCOCO, word, 1 + length(word2intCOCO)) for word in split( lowercase( replace( json["caption"], punc, "" ) ) ) ] ) ] )
		push!( dataCOCOFlickr, [ctrn_keys[json["image_id"]] transpose( [get!(word2intCOCOFlickr, word, 1 + length(word2intCOCOFlickr)) for word in split( lowercase( replace( json["caption"], punc, "" ) ) ) ] ) ] )
	end
	ctrn_COCO = fill(word2intCOCO[""], findmax([length(array) for array in dataCOCO])[1], length(dataCOCO) )
	ctrn_COCOFlickr = fill(word2intCOCOFlickr[""], findmax([length(array) for array in dataCOCOFlickr])[1], length(dataCOCOFlickr) )
	for i = 1:length(dataCOCO)
		ctrn_COCO[1:length(dataCOCO[i]),i] = dataCOCO[i]
		ctrn_COCOFlickr[1:length(dataCOCOFlickr[i]),i] = dataCOCOFlickr[i]
	end

	info("Parsing COCO2014 Validation Captions...")

	dataCOCO = Any[]
	dataCOCOFlickr = Any[]
	for json in cval_json["annotations"]
		push!( dataCOCO, [cval_keys[json["image_id"]] transpose( [get!(word2intCOCO, word, 1 + length(word2intCOCO)) for word in split( lowercase( replace( json["caption"], punc, "" ) ) ) ] ) ] )
		push!( dataCOCOFlickr, [cval_keys[json["image_id"]] transpose( [get!(word2intCOCOFlickr, word, 1 + length(word2intCOCOFlickr)) for word in split( lowercase( replace( json["caption"], punc, "" ) ) ) ] ) ] )
	end
	cval_COCO = fill(word2intCOCO[""], findmax([length(array) for array in dataCOCO])[1], length(dataCOCO) )
	cval_COCOFlickr = fill(word2intCOCOFlickr[""], findmax([length(array) for array in dataCOCOFlickr])[1], length(dataCOCOFlickr) )
	for i = 1:length(dataCOCO)
		cval_COCO[1:length(dataCOCO[i]),i] = dataCOCO[i]
		cval_COCOFlickr[1:length(dataCOCOFlickr[i]),i] = dataCOCOFlickr[i]
	end

	int2wordCOCO = Array(Any,length(word2intCOCO))
	int2wordCOCOFlickr = Array(Any,length(word2intCOCOFlickr))
	for m in keys(word2intCOCO)
		int2wordCOCO[word2intCOCO[m]] = m
	end
	for m in keys(word2intCOCOFlickr)
		int2wordCOCOFlickr[word2intCOCOFlickr[m]] = m
	end

	info("Saving Data...")

	info("Saving YouTube Data...")
	JLD.save(	"youtube_data.jld",
				"word2int", word2int,
				"int2word", int2word,
				"xtrn", xtrn, "xtst", xtst, "xval", xval,
				"ytrn", ytrn, "ytst", ytst, "yval", yval	)

	info("Saving Flickr30k Data...")
	JLD.save(	"flickr_data.jld",
				"word2intFlickr", word2intFlickr,
				"int2wordFlickr", int2wordFlickr,
				"ftrn_names", ftrn_names,
				"fval_names", fval_names,
				"ftrn", ftrn,
				"fval", fval	)

	info("Saving COCO2014 Data...")
	JLD.save(	"coco_data.jld",
				"word2intCOCO", word2intCOCO,
				"int2wordCOCO", int2wordCOCO,
				"word2intCOCOFlickr", word2intCOCOFlickr,
				"int2wordCOCOFlickr", int2wordCOCOFlickr,
				"ctrn_names", ctrn_names,
				"cval_names", cval_names,
				"ctrn_COCO", ctrn_COCO,
				"cval_COCO", cval_COCO,
				"ctrn_COCOFlickr", ctrn_COCOFlickr,
				"cval_COCOFlickr", cval_COCOFlickr	)

end

main()
