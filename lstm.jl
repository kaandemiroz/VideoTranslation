using Knet
using JLD

# Used for the LSTM operations with two inputs (Visual vector + word)
@knet function wbf3(x1, x2, x3; f=:sigm, o...)
    y1 = wdot(x1; o...)
    y2 = wdot(x2; o...)
    y3 = wdot(x3; o...)
    x4 = add(y2,y1)
	x5 = add(x4,y3)
    y4 = bias(x5; o...)
    return f(y4; o...)
end

# First LSTM layer where word embedding and concatenation occurs
@knet function lstm1(x1,word; winit = Uniform(-0.08, 0.08), binit = Constant(0), o...)
	x2 = wdot(word; init = winit, o...)
    input  = wbf3(x1,x2,h; o..., f=:sigm)
    forget = wbf3(x1,x2,h; o..., f=:sigm)
    output = wbf3(x1,x2,h; o..., f=:sigm)
    newmem = wbf3(x1,x2,h; o..., f=:tanh)
    cell = input .* newmem + cell .* forget
    h  = tanh(cell) .* output
    return h
end

@knet function lstm_network(x,word; layers = 2, size = 1000, vocab_size = 12594, o...)
	z1 = lstm1(x,word; out = size, winit = Uniform(-0.08, 0.08), binit = Constant(0), o...)
	z2 = lstm(z1; out = size, winit = Uniform(-0.08, 0.08), binit = Constant(0), fbias = 0, o...)
    return wbf(z2; o..., out = vocab_size, f = :soft)
end

# Compile the LSTM Network to lstm.jld
function main()
	info("Infering Vocabulary sizes...")
	vocab_size_YT = 			length( JLD.load("youtube_data.jld", "word2int") )
	vocab_size_YTFlickr =		length( JLD.load("flickr_data.jld", "word2intFlickr") )
	vocab_size_YTCOCO =			length( JLD.load("coco_data.jld", "word2intCOCO") )
	vocab_size_YTCOCOFlickr = 	length( JLD.load("coco_data.jld", "word2intCOCOFlickr") )

	info("Compiling LSTM Models...")
	lstmYT = 			compile(:lstm_network; vocab_size = vocab_size_YT )
	lstmYTFlickr = 		compile(:lstm_network; vocab_size = vocab_size_YTFlickr)
	lstmYTCOCO = 		compile(:lstm_network; vocab_size = vocab_size_YTCOCO)
	lstmYTCOCOFlickr = 	compile(:lstm_network; vocab_size = vocab_size_YTCOCOFlickr)

	info("Saving LSTM Models...")
	JLD.save(	"lstm.jld",
				"lstmYT", 			clean(lstmYT),
				"lstmYTFlickr", 	clean(lstmYTFlickr),
				"lstmYTCOCO", 		clean(lstmYTCOCO),
				"lstmYTCOCOFlickr",	clean(lstmYTCOCOFlickr)	)

end

main()
